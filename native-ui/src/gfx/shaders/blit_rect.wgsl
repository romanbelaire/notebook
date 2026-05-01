// Blit a texture region to a screen rect. Used for text layers with scroll.
// Uniform buffer: uv_min.xy, uv_max.xy (for scroll/clipping)
// Viewport is set per-draw to the dest rect.

@group(0) @binding(0)
var vello_texture: texture_2d<f32>;
@group(0) @binding(1)
var vello_sampler: sampler;
@group(0) @binding(2)
var<uniform> uv_params: UVParams;

struct UVParams {
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.tex_coord = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = uv_params.uv_min + (uv_params.uv_max - uv_params.uv_min) * in.tex_coord;
    let tex_color = textureSample(vello_texture, vello_sampler, uv);
    let a = tex_color.a;
    return vec4<f32>(tex_color.rgb * a, a);
}
