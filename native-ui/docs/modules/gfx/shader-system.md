# UI Shader System

The native-ui uses a **single wgpu pipeline** for all UI quads. The fragment shader branches on vertex flags and uniform data to produce plain rounded rects, borders, glows, **bubble** (velocity-driven border wobble), and **liquid slider** (stretch in direction of motion). This document is the reference for the shader system and how to extend it.

## Overview

- **Shader**: [native-ui/src/gfx/shaders/ui_shader.wgsl](native-ui/src/gfx/shaders/ui_shader.wgsl) (WGSL).
- **Vertex type**: [native-ui/src/gfx/types.rs](native-ui/src/gfx/types.rs) — `Vertex` and `Quad`.
- **Uniform writes**: [native-ui/src/gfx/renderer.rs](native-ui/src/gfx/renderer.rs) — each frame before building draw batches.

All quads are batched into one vertex buffer and drawn in one pass. Effect selection is per-quad via vertex attributes (`bubble`, `slider`) and by convention (e.g. negative `corner_radius` for borders, `corner_radius ≈ half size` for glow).

## Uniforms

Layout (must match WGSL and Rust write):

| Offset (bytes) | Content              | Type        | Used in   |
|----------------|----------------------|------------|-----------|
| 0              | `projection`         | `mat4x4<f32>` | Vertex   |
| 64             | `time`               | `f32`      | Fragment  |
| 68             | `scroll_velocity`    | `f32`      | Fragment (bubble) |
| 72             | `cursor`             | `vec2<f32>`| Fragment (bubble) |
| 80             | `slider_velocity`    | `f32`      | Fragment (slider) |

**Total**: 84 bytes (21 × 4). Written each frame in `Renderer::render()`:

- **Projection**: full 4×4 orthographic matrix at offset 0.
- **Fragment block** (offset 64): `[time, scroll_velocity, cursor.x, cursor.y, slider_velocity]`.  
  - `time`: elapsed seconds from renderer start.
  - `scroll_velocity`: chat message list scroll velocity (when Chat tab active); otherwise 0.
  - `cursor`: mouse position in UI coordinates.
  - `slider_velocity`: **trailing** edge velocity in **pixels per second** (positive = moving right), from `tab_bar.slider_trailing_animation.velocity * tab_width`; the tail catch-up speed drives the stretch.

## Vertex format

`Vertex` is a single layout used for every UI quad. Attributes:

| Location | Name           | Type        | Source / meaning |
|----------|----------------|------------|------------------|
| 0        | `position`     | `vec2<f32>`| Quad vertex position (world) |
| 1        | `color`        | `vec4<f32>`| RGBA |
| 2        | `quad_pos`     | `vec2<f32>`| Quad top-left (same for all 6 vertices of the quad) |
| 3        | `quad_size`    | `vec2<f32>`| Quad size (same for all 6) |
| 4        | `corner_radius`| `f32`      | Radius for SDF; **negative** = border mode (width in color.a) |
| 5        | `bubble`       | `f32`      | `1.0` = apply bubble border displacement; `0.0` = normal |
| 6        | `slider`       | `f32`      | `1.0` = apply liquid slider stretch; `0.0` = normal |

Rust: `Quad` has `bubble_effect: bool` and `slider_effect: bool`; they are converted to `bubble` and `slider` in `to_vertices()` / `push_vertices_to()`.

## Fragment shader branches (order matters)

The fragment shader resolves the shape and effect in this order:

1. **Border mode** — `corner_radius < 0`: draw a rounded-rect border (outer minus inner SDF). Border width from `color.a`. No effect flags used.
2. **No SDF** — `corner_radius <= 0` (and not border): solid `color`.
3. **Glow** — `corner_radius` ≈ half of min(quad_size): elliptical glow with cubic falloff and soft edge. No effect flags.
4. **Regular rounded rect** — everything else:
   - **Liquid slider** (if `in.slider > 0.5`): stretch SDF in direction of `slider_velocity` (see below).
   - **Bubble** (if `in.bubble > 0.5`): add noise-based displacement to the SDF only near the border, driven by `time`, `scroll_velocity`, and `cursor`.
   - Final alpha from SDF smoothstep.

So: one quad can be **either** bubble **or** slider (they use different flags); the slider quad should have `bubble_effect: false` and `slider_effect: true`.

## Effects in detail

### Rounded rectangle (base)

- SDF: `rounded_box_sdf(pos, size, radius)` in quad-local space.
- Alpha: `1.0 - smoothstep(-1.0, 0.0, dist)` for 1px antialiasing.

### Border mode

- `corner_radius < 0`: outer radius = `-corner_radius`, inner rect inset by `color.a`, inner radius reduced.
- Fragment is inside border if inside outer SDF and outside inner SDF.

### Glow

- Detected when `corner_radius` is close to half the smaller side of `quad_size`.
- Elliptical distance, cubic falloff, soft edge; pixels outside ellipse are discarded.

### Bubble (chat bubbles)

- **Where**: Chat message bubbles (and optional “generating” bubble); set `bubble_effect: true` on those quads only.
- **What**: Only the **edge** wobbles; interior stays solid.
- **How**: Noise from `hash12(rel_pos * frequency + motion_offset + cursor_influence)`. Motion uses `time` and `scroll_velocity`; cursor adds a small position offset. Displacement is applied only near the SDF zero: `border_factor = 1.0 - smoothstep(0, border_thickness, abs(dist))`, then `dist += displacement * amplitude * border_factor`.
- **Tunables** (constants in shader): `frequency`, `border_thickness`, `amplitude`, `scroll_strength`; cursor scale.

### Liquid slider (nav bar)

- **Where**: Header tab bar slider only; set `slider_effect: true` on that single quad.
- **What**: **Trailing SDF**: a pill that follows the leading edge; tail width (extent along the normal to motion) tapers with distance from the head. Rounded pill shape with radius that depends on distance from head (full at head, smaller toward the tail).
- **How**: From `uniforms.slider_velocity` (pixels/sec): `stretch_amount = min(abs(vel) * stretch_scale, max_stretch)`. Logical pill runs along the movement axis; head = leading edge (right when vel ≥ 0, left when vel &lt; 0). For each fragment, closest point on the segment (head–tail) is found; `distance_from_head` is the distance along the segment from that point to the head. Tail radius (half-height, i.e. magnitude along the normal to motion) = `base_radius * max(0.2, 1.0 - distance_from_head / total_length)`. SDF = distance to segment − radius at closest point. Same AA as base rounded rect.
- **Tunables**: `stretch_scale`, `max_stretch`, tail falloff min (0.2) in the shader.

## Adding a new effect

1. **Decide data source**
   - **Global (all frames)**: Add a new field to the uniform block (WGSL + renderer write). Keep alignment (e.g. 4-byte for f32, 8-byte for vec2).
   - **Per quad**: Add a new vertex attribute (e.g. `effect_id` or another float flag) and a corresponding field on `Quad`; update `Vertex::desc()` and all `to_vertices`/`push_vertices_to` sites to set it.

2. **Shader**
   - Extend `VertexInput` / `VertexOutput` if you added a vertex attribute.
   - In the fragment shader, add a branch **in the correct order** (see “Fragment shader branches” above). For border-only effects, restrict displacement to the edge (e.g. with a `border_factor` like the bubble).

3. **Rust**
   - If new uniform: extend the buffer size and the block written at 64 (or wherever the fragment block starts); keep layout identical to WGSL.
   - If new vertex attribute: add the field to `Vertex`, update `Vertex::desc()` offsets, add the field to `Quad`, set it in every place that builds quads (or in `to_vertices`/`push_vertices_to` from the new Quad field).

4. **Single pipeline**: All quads still go through the same pipeline; the new logic is selected by the new uniform or vertex flag.

## Diagnosing fuzzy or pixelated bubble borders

- **Projection vs resolution**: The SDF and smoothstep are in the same space as the projection (world units = pixels). If the window/surface is at logical resolution and the OS scales the backbuffer, edges can look soft or pixelated. Ensure the renderer uses physical pixel size for the projection and surface (no extra scaling).
- **AA width**: Edge alpha uses `smoothstep(-aa_width, 0.0, dist)` with `aa_width = 0.5` for a crisp ~0.5px transition. A wider band (e.g. 1.0) looks softer. The constant is in [ui_shader.wgsl](native-ui/src/gfx/shaders/ui_shader.wgsl) in the "Anti-aliased edge" comment block.
- **Bubble noise**: The bubble uses **smooth value noise** (bilinear interpolation of `hash12` at cell corners), not raw hash, so the displacement varies smoothly and the edge doesn’t look pixelated. If the edge still looks blocky, the cause is likely resolution/scale (see above). Tunables: `border_thickness`, `amplitude`, `frequency`.

## File reference

| File | Role |
|------|------|
| [native-ui/src/gfx/shaders/ui_shader.wgsl](native-ui/src/gfx/shaders/ui_shader.wgsl) | Vertex + fragment shader; Uniforms; SDF, noise, and all effect branches |
| [native-ui/src/gfx/types.rs](native-ui/src/gfx/types.rs) | `Vertex`, `Quad`, `Vertex::desc()`, `to_vertices`, `push_vertices_to` |
| [native-ui/src/gfx/renderer.rs](native-ui/src/gfx/renderer.rs) | Uniform buffer creation and per-frame write (projection + fragment block) |
| [native-ui/src/gfx/components/header.rs](native-ui/src/gfx/components/header.rs) | Slider quad with `slider_effect: true` |
| [native-ui/src/gfx/components/chat.rs](native-ui/src/gfx/components/chat.rs) | Bubble quads with `bubble_effect: true` |

## Related docs

- [Rendering pipeline](../../architecture/rendering.md) — overall frame flow and where the UI shader fits.
- [Graphics types](types.md) — `Vertex` and `Quad` API (this doc focuses on shader semantics and extension).
- [bubble-UI.md](../../../../bubble-UI.md) — design notes for SDF + displacement (bubble-style edges).
