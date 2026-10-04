// ABI training operation: eight distinct storage bindings and 16 uniform words.
// All reductions have a single writer, including repeated-token embeddings.
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<u32>;
@group(0) @binding(3) var<storage, read_write> d: array<f32>;
@group(0) @binding(4) var<storage, read_write> e: array<f32>;
@group(0) @binding(5) var<storage, read_write> f: array<f32>;
@group(0) @binding(6) var<storage, read_write> g: array<f32>;
@group(0) @binding(7) var<storage, read_write> h: array<f32>;
@group(0) @binding(8) var<uniform> uniforms: array<vec4<u32>, 4>;
fn read_b(i: u32) -> f32 { return bitcast<f32>(b[i]); }
fn read_c(i: u32) -> f32 { return bitcast<f32>(c[i]); }
fn write_b(i: u32, v: f32) { b[i] = bitcast<u32>(v); }
fn write_c(i: u32, v: f32) { c[i] = bitcast<u32>(v); }
fn word(i: u32) -> u32 { return uniforms[i / 4u][i % 4u]; }
fn p(i: u32) -> u32 { return word(i + 2u); }
fn fp(i: u32) -> f32 { return bitcast<f32>(p(i)); }
fn finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let i = id.x + id.y * groups.x * 64u;
  let count = word(1u);
  if (i >= count) { return; }
  switch word(0u) {
    case 1u: { a[i] = read_b(i); }
    case 2u: { a[i] = read_b(i) * read_c(i); }
    case 3u: { a[i] = read_b(i) + read_c(i); }
    case 4u: { a[i] = select(0.0, read_b(i), read_c(i) > 0.0); }
    case 5u: { // cached normalization, count = rows
      let channels = p(0u); let start = i * channels;
      var mean = 0.0;
      for (var j = 0u; j < channels; j++) { mean += read_b(start + j); }
      mean /= f32(channels);
      var variance = 0.0;
      for (var j = 0u; j < channels; j++) { let v = read_b(start + j) - mean; variance += v * v; }
      let inv = 1.0 / sqrt(variance / f32(channels) + fp(1u));
      f[i] = inv;
      for (var j = 0u; j < channels; j++) {
        let v = (read_b(start + j) - mean) * inv;
        e[start + j] = v; a[start + j] = v * read_c(j) + d[j];
      }
    }
    case 6u: { // normalization input gradient, count = rows
      let channels = p(0u); let start = i * channels;
      var sum = 0.0; var product = 0.0;
      for (var j = 0u; j < channels; j++) { let v = read_b(start + j) * read_c(j); sum += v; product += v * d[start + j]; }
      for (var j = 0u; j < channels; j++) { a[start + j] = e[i] / f32(channels) * (f32(channels) * read_b(start + j) * read_c(j) - sum - d[start + j] * product); }
    }
    case 7u: { // gamma/beta gradients, count = channels
      var dg = 0.0; var db = 0.0;
      for (var r = 0u; r < p(0u); r++) { let k = r * count + i; dg += read_c(k) * d[k]; db += read_c(k); }
      a[i] = dg; write_b(i, db);
    }
    case 8u: { // linear dx: rows, inputs, outputs
      let inputs = p(1u); let outputs = p(2u); let row = i / inputs; let input = i % inputs;
      var sum = 0.0;
      for (var j = 0u; j < outputs; j++) { sum += read_b(row * outputs + j) * read_c(input * outputs + j); }
      a[i] = sum;
    }
    case 9u: { // linear dw
      let inputs = p(1u); let outputs = p(2u); let input = i / outputs; let output = i % outputs;
      var sum = 0.0;
      for (var r = 0u; r < p(0u); r++) { sum += read_b(r * inputs + input) * read_c(r * outputs + output); }
      a[i] = sum;
    }
    case 10u: { // linear db, count = outputs
      var sum = 0.0;
      for (var r = 0u; r < p(0u); r++) { sum += read_b(r * count + i); }
      a[i] = sum;
    }
    case 11u: { // token embedding gradient: rows, time, channels
      let channels = p(2u); let token = i / channels; let ch = i % channels;
      var sum = 0.0;
      for (var r = 0u; r < p(0u); r++) { if (b[r] == token) { sum += read_c(r * channels + ch); } }
      a[i] = sum;
    }
    case 12u: { // position embedding gradient
      let channels = p(2u); let pos = i / channels; let ch = i % channels;
      var sum = 0.0;
      if (pos < p(1u)) { for (var r = pos; r < p(0u); r += p(1u)) { sum += read_b(r * channels + ch); } }
      a[i] = sum;
    }
    case 13u: { // attention dP including dropout: batch, time, channels, heads
      let time = p(1u); let channels = p(2u); let heads = p(3u); let width = channels / heads;
      let key = i % time; let query = (i / time) % time; let head = (i / (time * time)) % heads; let batch = i / (time * time * heads);
      var sum = 0.0;
      if (key <= query) {
        for (var ch = 0u; ch < width; ch++) { sum += read_b((batch * time + query) * channels + head * width + ch) * read_c((batch * time + key) * 3u * channels + 2u * channels + head * width + ch); }
        sum *= d[i];
      }
      a[i] = sum;
    }
    case 14u: { // attention softmax backward: time, scale; count = BHT
      let time = p(0u); let query = i % time; let start = i * time;
      var dot = 0.0;
      for (var k = 0u; k <= query; k++) { dot += read_b(start + k) * read_c(start + k); }
      for (var k = 0u; k <= query; k++) { a[start + k] = read_b(start + k) * (read_c(start + k) - dot) * fp(1u); }
      for (var k = query + 1u; k < time; k++) { a[start + k] = 0.0; }
    }
    case 15u: { // dQ, dK, dV, one writer per component
      let time = p(1u); let channels = p(2u); let heads = p(3u); let width = channels / heads;
      let ch = i % channels; let component = (i / channels) % 3u; let pos = (i / (3u * channels)) % time; let batch = i / (3u * channels * time); let head = ch / width;
      let start = (batch * heads + head) * time * time;
      var sum = 0.0;
      if (component == 0u) {
        for (var k = 0u; k <= pos; k++) { sum += read_c(start + pos * time + k) * read_b((batch * time + k) * 3u * channels + channels + ch); }
      } else if (component == 1u) {
        for (var t = pos; t < time; t++) { sum += read_c(start + t * time + pos) * read_b((batch * time + t) * 3u * channels + ch); }
      } else {
        for (var t = pos; t < time; t++) { let k = start + t * time + pos; sum += d[k] * e[k] * f[(batch * time + t) * channels + ch]; }
      }
      a[i] = sum;
    }
    case 16u: { // cross entropy: vocab, write gradient; count = rows
      let vocab = p(0u); let start = i * vocab; let target_index = c[i];
      var valid = target_index < vocab; var maximum = read_b(start);
      for (var j = 0u; j < vocab; j++) { valid = valid && finite(read_b(start + j)); maximum = max(maximum, read_b(start + j)); }
      if (!valid) { d[i] = 0.0; e[i] = 1.0; return; }
      var total = 0.0;
      for (var j = 0u; j < vocab; j++) { total += exp(read_b(start + j) - maximum); }
      let loss = (maximum - read_b(start + target_index)) + log(total);
      d[i] = loss; e[i] = select(1.0, 0.0, finite(loss));
      if (p(1u) != 0u) {
        for (var j = 0u; j < vocab; j++) { a[start + j] = (exp(read_b(start + j) - maximum) / total - select(0.0, 1.0, j == target_index)) / f32(count); }
      }
    }
    case 17u: { // sum a status buffer
      var total = 0.0;
      for (var j = 0u; j < p(0u); j++) { total += read_b(j); }
      a[0] = total;
    }
    case 18u: { // finite scan, one result per 256 elements
      var invalid = 0.0;
      for (var j = i * 256u; j < min((i + 1u) * 256u, p(0u)); j++) { if (!finite(read_b(j))) { invalid = 1.0; } }
      a[p(1u) + i] = invalid;
    }
    case 19u: { // AdamW; len, lr, decay, beta1, beta2, epsilon, corrections, status offset
      var invalid = 0.0;
      for (var j = i * 256u; j < min((i + 1u) * 256u, p(0u)); j++) {
        let m = fp(3u) * read_c(j) + (1.0 - fp(3u)) * read_b(j);
        let v = fp(4u) * d[j] + (1.0 - fp(4u)) * read_b(j) * read_b(j);
        let value = a[j] * (1.0 - fp(1u) * fp(2u)) - fp(1u) * (m / fp(6u)) / (sqrt(v / fp(7u)) + fp(5u));
        if (!finite(m) || !finite(v) || !finite(value)) { invalid = 1.0; }
        a[j] = value; write_c(j, m); d[j] = v;
      }
      e[p(8u) + i] = invalid;
    }
    case 20u: { // mean loss plus independent invalid flags
      var loss = 0.0; var invalid = 0.0;
      for (var j = 0u; j < p(0u); j++) { loss += read_b(j) / f32(p(0u)); }
      for (var j = 0u; j < p(1u); j++) { invalid += read_c(j); }
      for (var j = 0u; j < p(2u); j++) { invalid += d[j]; }
      a[0] = loss; a[1] = invalid;
    }
    // Keep all eight bindings in the layout; unknown kinds are rejected by the host.
    default: { a[i] = g[i] + h[i]; }
  }
}
