@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> token_table: array<f32>;
@group(0) @binding(2) var<storage, read> position_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: vec4<u32>; // count, time, channels
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let linear_id = id.x + id.y * groups.x * 64u;
  let i = linear_id;
  if (i >= p.x) { return; }
  let row = i / p.z;
  let c = i % p.z;
  output[i] = token_table[tokens[row] * p.z + c] + position_table[(row % p.y) * p.z + c];
}
