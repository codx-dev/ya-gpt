@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: vec4<u32>; // rows, inputs, outputs, bias enabled
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let linear_id = id.x + id.y * groups.x * 64u;
  let index = linear_id;
  if (index >= p.x * p.z) { return; }
  let row = index / p.z;
  let column = index % p.z;
  var sum = 0.0;
  if (p.w != 0u) { sum = bias[column]; }
  for (var i = 0u; i < p.y; i++) { sum += input[row * p.y + i] * weight[i * p.z + column]; }
  output[index] = sum;
}
