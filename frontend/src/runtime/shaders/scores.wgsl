@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>; // batch, time, channels, heads
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let linear_id = id.x + id.y * groups.x * 64u;
  let i = linear_id;
  if (i >= p.x * p.w * p.y * p.y) { return; }
  let k = i % p.y;
  let t = (i / p.y) % p.y;
  let h = (i / (p.y * p.y)) % p.w;
  let b = i / (p.y * p.y * p.w);
  if (k > t) { scores[i] = 0.0; return; }
  let width = p.z / p.w;
  let q = (b * p.y + t) * 3u * p.z + h * width;
  let key = (b * p.y + k) * 3u * p.z + p.z + h * width;
  var dot = 0.0;
  for (var c = 0u; c < width; c++) { dot += qkv[q + c] * qkv[key + c]; }
  scores[i] = dot / sqrt(f32(width));
}
