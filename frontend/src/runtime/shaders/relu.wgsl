@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> p: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let linear_id = id.x + id.y * groups.x * 64u;
  if (linear_id < p.x) { values[linear_id] = max(values[linear_id], 0.0); }
}
