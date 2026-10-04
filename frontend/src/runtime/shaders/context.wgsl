@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> probabilities: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>; // batch, time, channels, heads
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let linear_id = id.x + id.y * groups.x * 64u;
  let i = linear_id;
  if (i >= p.x * p.y * p.z) { return; }
  let c = i % p.z;
  let t = (i / p.z) % p.y;
  let b = i / (p.z * p.y);
  let h = c / (p.z / p.w);
  let start = ((b * p.w + h) * p.y + t) * p.y;
  var sum = 0.0;
  for (var k = 0u; k <= t; k++) { sum += probabilities[start + k] * qkv[(b * p.y + k) * 3u * p.z + 2u * p.z + c]; }
  output[i] = sum;
}
