#include <stddef.h>
#include <stdint.h>
#include <math.h>

static_assert(sizeof(size_t) == 8, "ya-gpt-cuda requires a 64-bit target");

// All kernels use grid-stride loops so launch dimensions never truncate tensors.
#define EACH(i, n) \
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += (size_t)blockDim.x * gridDim.x)

__device__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ float dropout_mask(uint64_t seed, uint64_t stream, size_t index, float p) {
    if (p == 0.0f) return 1.0f;
    // SplitMix64 used as a stateless counter hash. Hash the stream key separately
    // so adjacent streams cannot be short shifts of the same sequence at seed 0.
    uint64_t key = mix64(seed ^ ((stream + 1) * 0xd1b54a32d192ed03ULL));
    uint64_t z = mix64(key + ((uint64_t)index + 1) * 0x9e3779b97f4a7c15ULL);
    float uniform = (float)(z >> 40) * (1.0f / 16777216.0f);
    return uniform < p ? 0.0f : 1.0f / (1.0f - p);
}

extern "C" __global__ void fill(float* out, size_t n, float value) {
    EACH(i, n) out[i] = value;
}

extern "C" __global__ void copy(const float* x, float* out, size_t n) {
    EACH(i, n) out[i] = x[i];
}

extern "C" __global__ void add(float* out, const float* x, size_t n) {
    EACH(i, n) out[i] += x[i];
}

extern "C" __global__ void bias_add(float* out, const float* bias, size_t n, size_t width) {
    EACH(i, n) out[i] += bias[i % width];
}

extern "C" __global__ void bias_backward(
    const float* dy, float* db, size_t rows, size_t width
) {
    EACH(ch, width) {
        float sum = 0.0f;
        for (size_t r = 0; r < rows; ++r) sum += dy[r * width + ch];
        db[ch] = sum;
    }
}

extern "C" __global__ void relu(float* x, size_t n) {
    EACH(i, n) x[i] = fmaxf(x[i], 0.0f);
}

extern "C" __global__ void relu_backward(const float* x, float* dx, size_t n) {
    EACH(i, n) if (x[i] <= 0.0f) dx[i] = 0.0f;
}

extern "C" __global__ void norm_forward(
    const float* x, const float* gamma, const float* beta, float* out,
    float* normalized, float* inverse_std, size_t rows, size_t channels,
    float epsilon, unsigned int save
) {
    EACH(row, rows) {
        size_t start = row * channels;
        float mean = 0.0f;
        for (size_t ch = 0; ch < channels; ++ch) mean += x[start + ch];
        mean /= (float)channels;
        float variance = 0.0f;
        for (size_t ch = 0; ch < channels; ++ch) {
            float delta = x[start + ch] - mean;
            variance += delta * delta;
        }
        float inv = 1.0f / sqrtf(variance / (float)channels + epsilon);
        for (size_t ch = 0; ch < channels; ++ch) {
            float value = (x[start + ch] - mean) * inv;
            out[start + ch] = value * gamma[ch] + beta[ch];
            if (save) normalized[start + ch] = value;
        }
        if (save) inverse_std[row] = inv;
    }
}

extern "C" __global__ void norm_backward_input(
    const float* dy, const float* gamma, const float* normalized,
    const float* inverse_std, float* dx, size_t rows, size_t channels
) {
    EACH(row, rows) {
        size_t start = row * channels;
        float sum = 0.0f, product = 0.0f;
        for (size_t ch = 0; ch < channels; ++ch) {
            float gradient = dy[start + ch] * gamma[ch];
            sum += gradient;
            product += gradient * normalized[start + ch];
        }
        for (size_t ch = 0; ch < channels; ++ch) {
            dx[start + ch] = inverse_std[row] / (float)channels
                * ((float)channels * dy[start + ch] * gamma[ch]
                   - sum - normalized[start + ch] * product);
        }
    }
}

extern "C" __global__ void norm_backward_weights(
    const float* dy, const float* normalized, float* dg, float* db,
    size_t rows, size_t channels
) {
    EACH(ch, channels) {
        float gamma = 0.0f, beta = 0.0f;
        for (size_t row = 0; row < rows; ++row) {
            size_t i = row * channels + ch;
            gamma += dy[i] * normalized[i];
            beta += dy[i];
        }
        dg[ch] = gamma;
        db[ch] = beta;
    }
}

extern "C" __global__ void embedding_forward(
    const unsigned int* tokens, const float* token_table, const float* position_table,
    float* out, size_t n, size_t time, size_t channels
) {
    EACH(i, n) {
        size_t row = i / channels, ch = i % channels;
        out[i] = token_table[(size_t)tokens[row] * channels + ch]
            + position_table[(row % time) * channels + ch];
    }
}

extern "C" __global__ void embedding_backward(
    const unsigned int* tokens, const float* dy, float* dt, float* dp,
    size_t n, size_t time, size_t channels
) {
    EACH(i, n) {
        size_t row = i / channels, ch = i % channels;
        atomicAdd(&dt[(size_t)tokens[row] * channels + ch], dy[i]);
        atomicAdd(&dp[(row % time) * channels + ch], dy[i]);
    }
}

extern "C" __global__ void dropout_add(
    float* branch, const float* residual, float* mask, size_t n,
    float p, uint64_t seed, uint64_t stream
) {
    EACH(i, n) {
        float m = dropout_mask(seed, stream, i, p);
        mask[i] = m;
        branch[i] = branch[i] * m + residual[i];
    }
}

extern "C" __global__ void masked(
    const float* x, const float* mask, float* out, size_t n
) {
    EACH(i, n) out[i] = x[i] * mask[i];
}

// QKV layout: [batch, time, 3, heads, head_width].
// Attention layout: [batch, heads, query_time, key_time].
extern "C" __global__ void attention_scores(
    const float* qkv, float* scores, size_t count, size_t time,
    size_t channels, size_t heads
) {
    size_t width = channels / heads;
    float scale = 1.0f / sqrtf((float)width);
    EACH(i, count) {
        size_t key = i % time, query = (i / time) % time;
        size_t head = (i / time / time) % heads, batch = i / time / time / heads;
        float score = 0.0f;
        if (key <= query) {
            size_t q = (batch * time + query) * 3 * channels + head * width;
            size_t k = (batch * time + key) * 3 * channels + channels + head * width;
            for (size_t ch = 0; ch < width; ++ch) score += qkv[q + ch] * qkv[k + ch];
        }
        scores[i] = score * scale;
    }
}

extern "C" __global__ void attention_softmax(
    float* scores, float* mask, size_t rows, size_t time, float p, uint64_t seed
) {
    EACH(row, rows) {
        size_t query = row % time, start = row * time;
        float max_value = -INFINITY, total = 0.0f;
        for (size_t k = 0; k <= query; ++k) max_value = fmaxf(max_value, scores[start + k]);
        for (size_t k = 0; k <= query; ++k) {
            float value = expf(scores[start + k] - max_value);
            scores[start + k] = value;
            total += value;
        }
        for (size_t k = 0; k < time; ++k) {
            scores[start + k] = k <= query ? scores[start + k] / total : 0.0f;
            mask[start + k] = k <= query ? dropout_mask(seed, 0, start + k, p) : 0.0f;
        }
    }
}

extern "C" __global__ void attention_context(
    const float* qkv, const float* probabilities, const float* mask, float* out,
    size_t n, size_t time, size_t channels, size_t heads
) {
    size_t width = channels / heads;
    EACH(i, n) {
        size_t ch = i % channels, query = (i / channels) % time, batch = i / channels / time;
        size_t head = ch / width;
        size_t start = ((batch * heads + head) * time + query) * time;
        float sum = 0.0f;
        for (size_t k = 0; k <= query; ++k) {
            size_t value = (batch * time + k) * 3 * channels + 2 * channels + ch;
            sum += probabilities[start + k] * mask[start + k] * qkv[value];
        }
        out[i] = sum;
    }
}

extern "C" __global__ void attention_dprob(
    const float* qkv, const float* dy, const float* mask, float* ds,
    size_t count, size_t time, size_t channels, size_t heads
) {
    size_t width = channels / heads;
    EACH(i, count) {
        size_t key = i % time, query = (i / time) % time;
        size_t head = (i / time / time) % heads, batch = i / time / time / heads;
        float sum = 0.0f;
        if (key <= query) {
            size_t v = (batch * time + key) * 3 * channels + 2 * channels + head * width;
            size_t y = (batch * time + query) * channels + head * width;
            for (size_t ch = 0; ch < width; ++ch) sum += dy[y + ch] * qkv[v + ch];
        }
        ds[i] = sum * mask[i];
    }
}

extern "C" __global__ void attention_dsoftmax(
    const float* probabilities, float* ds, size_t rows, size_t time, float scale
) {
    EACH(row, rows) {
        size_t query = row % time, start = row * time;
        float dot = 0.0f;
        for (size_t k = 0; k <= query; ++k) dot += probabilities[start + k] * ds[start + k];
        for (size_t k = 0; k < time; ++k)
            ds[start + k] = k <= query ? probabilities[start + k] * (ds[start + k] - dot) * scale : 0.0f;
    }
}

extern "C" __global__ void attention_dqkv(
    const float* qkv, const float* probabilities, const float* mask,
    const float* dy, const float* ds, float* dqkv,
    size_t count, size_t time, size_t channels, size_t heads
) {
    size_t width = channels / heads;
    // Each Q, K, V element has a single writer, avoiding atomic attention reductions.
    EACH(i, count) {
        size_t ch = i % channels, part = (i / channels) % 3;
        size_t t = (i / (3 * channels)) % time, batch = i / (3 * channels) / time;
        size_t head = ch / width, base = (batch * heads + head) * time * time;
        float sum = 0.0f;
        if (part == 0) {
            for (size_t k = 0; k <= t; ++k)
                sum += ds[base + t * time + k] * qkv[(batch * time + k) * 3 * channels + channels + ch];
        } else if (part == 1) {
            for (size_t q = t; q < time; ++q)
                sum += ds[base + q * time + t] * qkv[(batch * time + q) * 3 * channels + ch];
        } else {
            for (size_t q = t; q < time; ++q) {
                size_t offset = base + q * time + t;
                sum += probabilities[offset] * mask[offset] * dy[(batch * time + q) * channels + ch];
            }
        }
        dqkv[i] = sum;
    }
}

extern "C" __global__ void cross_entropy(
    const float* logits, const unsigned int* targets, float* losses, float* gradient,
    size_t rows, size_t vocab, unsigned int with_grad
) {
    EACH(row, rows) {
        size_t start = row * vocab;
        float max_value = -INFINITY, total = 0.0f;
        bool valid = true;
        for (size_t j = 0; j < vocab; ++j) {
            valid = valid && isfinite(logits[start + j]);
            max_value = fmaxf(max_value, logits[start + j]);
        }
        for (size_t j = 0; j < vocab; ++j) total += expf(logits[start + j] - max_value);
        float loss = (max_value - logits[start + targets[row]]) + logf(total);
        losses[row] = valid ? loss / (float)rows : NAN;
        if (with_grad) {
            for (size_t j = 0; j < vocab; ++j)
                gradient[start + j] = (expf(logits[start + j] - max_value) / total
                    - (float)(j == targets[row])) / (float)rows;
        }
    }
}

extern "C" __global__ void reduce_loss(const float* losses, float* out, size_t rows) {
    EACH(i, (size_t)1) {
        double sum = 0.0;
        for (size_t r = 0; r < rows; ++r) sum += losses[r];
        out[0] = (float)sum;
    }
}

extern "C" __global__ void all_finite(const float* values, unsigned int* invalid, size_t n) {
    EACH(i, n) if (!isfinite(values[i])) atomicExch(invalid, 1U);
}

extern "C" __global__ void adamw(
    float* values, const float* gradients, float* first, float* second,
    unsigned int* invalid, size_t n, float lr, float decay,
    float beta1, float beta2, float epsilon, float correction1, float correction2
) {
    EACH(i, n) {
        float g = gradients[i];
        float m = beta1 * first[i] + (1.0f - beta1) * g;
        float v = beta2 * second[i] + (1.0f - beta2) * g * g;
        float value = values[i] * (1.0f - lr * decay)
            - lr * (m / correction1) / (sqrtf(v / correction2) + epsilon);
        if (!isfinite(m) || !isfinite(v) || !isfinite(value)) {
            atomicExch(invalid, 1U);
        } else {
            values[i] = value;
            first[i] = m;
            second[i] = v;
        }
    }
}

extern "C" __global__ void sample(
    const float* logits, unsigned int* out, size_t rows, size_t vocab, float uniform
) {
    EACH(i, (size_t)1) {
        size_t start = (rows - 1) * vocab;
        float max_value = -INFINITY, total = 0.0f;
        bool valid = true;
        for (size_t j = 0; j < vocab; ++j) {
            valid = valid && isfinite(logits[start + j]);
            max_value = fmaxf(max_value, logits[start + j]);
        }
        if (!valid) { out[0] = UINT32_MAX; continue; }
        for (size_t j = 0; j < vocab; ++j) total += expf(logits[start + j] - max_value);
        float threshold = uniform * total, cumulative = 0.0f;
        unsigned int selected = 0;
        for (size_t j = 0; j < vocab; ++j) {
            float mass = expf(logits[start + j] - max_value);
            if (mass > 0.0f) selected = (unsigned int)j;
            cumulative += mass;
            if (threshold < cumulative) break;
        }
        out[0] = selected;
    }
}
