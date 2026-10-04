@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> p: vec4<u32>; // batch, time, heads
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let linear_id = id.x + id.y * groups.x * 64u;
  let row = linear_id;
  if (row >= p.x * p.y * p.z) { return; }
  let t = row % p.y;
  let start = row * p.y;
  var maximum = scores[start];
  for (var k = 1u; k <= t; k++) { maximum = max(maximum, scores[start + k]); }
  var total = 0.0;
  for (var k = 0u; k <= t; k++) { let mass = exp(scores[start + k] - maximum); scores[start + k] = mass; total += mass; }
  for (var k = 0u; k <= t; k++) { scores[start + k] /= total; }
  for (var k = t + 1u; k < p.y; k++) { scores[start + k] = 0.0; }
}
