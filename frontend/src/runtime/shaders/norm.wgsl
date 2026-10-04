@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: vec4<u32>; // rows, channels, epsilon bits
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let linear_id = id.x + id.y * groups.x * 64u;
  let row = linear_id;
  if (row >= p.x) { return; }
  let start = row * p.y;
  var mean = 0.0;
  for (var c = 0u; c < p.y; c++) { mean += input[start + c]; }
  mean /= f32(p.y);
  var variance = 0.0;
  for (var c = 0u; c < p.y; c++) { let d = input[start + c] - mean; variance += d * d; }
  let inverse = inverseSqrt(variance / f32(p.y) + bitcast<f32>(p.z));
  for (var c = 0u; c < p.y; c++) { output[start + c] = (input[start + c] - mean) * inverse * gamma[c] + beta[c]; }
}
