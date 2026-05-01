// Full-screen blit (same as blit.wgsl) — DEBUG: tints non-transparent pixels magenta
// Enable with NOTEBOOK_DEBUG_TEXT_TINT=1

@group(0) @binding(0)
var vello_texture: texture_2d<f32>;
@group(0) @binding(1)
var vello_sampler: sampler;

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
    let tex_color = textureSample(vello_texture, vello_sampler, in.tex_coord);
    if tex_color.a > 0.001 {
        return vec4<f32>(1.0, 0.0, 1.0, tex_color.a);
    }
    return tex_color;
}
