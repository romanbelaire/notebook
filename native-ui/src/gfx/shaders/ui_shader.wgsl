struct Uniforms {
    projection: mat4x4<f32>,
    time: f32,
    scroll_velocity: f32,
    cursor: vec2<f32>,
    slider_velocity: f32,  // pixels per second, positive = moving right
    slider_tab_bar_x: f32,
    slider_tab_width: f32,
    slider_from_index: f32,  // floor(min(leading, following))
    slider_to_index: f32,    // ceil(max(leading, following))
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) quad_pos: vec2<f32>,
    @location(3) quad_size: vec2<f32>,
    @location(4) corner_radius: f32,
    @location(5) bubble: f32,
    @location(6) slider: f32,
    @location(7) shadow_sigma: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) frag_pos: vec2<f32>,
    @location(2) quad_pos: vec2<f32>,
    @location(3) quad_size: vec2<f32>,
    @location(4) corner_radius: f32,
    @location(5) bubble: f32,
    @location(6) slider: f32,
    @location(7) shadow_sigma: f32,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.projection * vec4<f32>(model.position, 0.0, 1.0);
    out.color = model.color;
    out.frag_pos = model.position;
    out.quad_pos = model.quad_pos;
    out.quad_size = model.quad_size;
    out.corner_radius = model.corner_radius;
    out.bubble = model.bubble;
    out.slider = model.slider;
    out.shadow_sigma = model.shadow_sigma;
    return out;
}

fn rounded_box_sdf(pos: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let half_size = size * 0.5;
    let center = half_size;
    let q = abs(pos - center) - half_size + radius;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0, 0.0))) - radius;
}

// Abramowitz & Stegun 7.1.26 rational approximation of erf (max abs error ~1.5e-7).
// Closed-form approximation that lets us evaluate a gaussian-blurred rounded box in O(1) per pixel.
fn erf7(x: f32) -> f32 {
    let a1: f32 =  0.254829592;
    let a2: f32 = -0.284496736;
    let a3: f32 =  1.421413741;
    let a4: f32 = -1.453152027;
    let a5: f32 =  1.061405429;
    let p: f32  =  0.3275911;
    let sign_x = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign_x * y;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // ============================================================================
    // Lighting primitives (bubble sentinels 3.0–6.0). Shared -45° key light.
    // All O(1) per pixel, no extra passes, no textures.
    // ============================================================================

    // bubble == 3.0 — Outer drop shadow.
    // Analytical gaussian-blurred rounded rectangle: 0.5 - 0.5 * erf(sdf / (sigma * sqrt(2))).
    if (in.bubble > 2.5 && in.bubble < 3.5) {
        let sigma = max(in.shadow_sigma, 0.0001);
        let rel = in.frag_pos - in.quad_pos;
        let d = rounded_box_sdf(rel, in.quad_size, max(in.corner_radius, 0.0));
        let cov = 0.5 - 0.5 * erf7(d / (sigma * 1.41421356));
        return vec4<f32>(in.color.rgb, in.color.a * cov);
    }

    // bubble == 4.0 — Inner drop shadow.
    // Same -45° light, but the shadow lives INSIDE the shape. The source rect is shifted by
    // +offset (bottom-right) and we take the erf of its SDF evaluated at the fragment — that
    // gives full shadow at the top-left interior (outside the offset shape but still inside
    // the real shape) and no shadow at the bottom-right interior. Clipped to the shape.
    // `in.slider` carries the scalar offset magnitude (applied at 45° diagonal).
    if (in.bubble > 3.5 && in.bubble < 4.5) {
        let sigma = max(in.shadow_sigma, 0.0001);
        let u: f32 = 0.70710678;
        let off = vec2<f32>(in.slider * u, in.slider * u);
        let r = max(in.corner_radius, 0.0);
        let rel = in.frag_pos - in.quad_pos;
        let d_shape = rounded_box_sdf(rel, in.quad_size, r);
        // Equivalent to shifting the source rect by +off: evaluate SDF at `rel - off`.
        let d_offset = rounded_box_sdf(rel - off, in.quad_size, r);
        let inside_mask = 1.0 - smoothstep(-0.5, 0.0, d_shape);
        if (inside_mask <= 0.0) {
            discard;
        }
        // 0 deep inside the offset shape, 1 far outside → shadow in the TL annulus.
        let cov = 0.5 + 0.5 * erf7(d_offset / (sigma * 1.41421356));
        return vec4<f32>(in.color.rgb, in.color.a * cov * inside_mask);
    }

    // bubble == 5.0 — Specular border highlight.
    // A bright rim along the inside of the shape's edge, brightest where the edge faces the
    // key light. Surface normal reconstructed from finite differences of the rounded-box SDF;
    // Lambert term is max(0, dot(-L, n)) with L = (+u, +u). A gaussian pulse peaking just
    // inside the edge confines the highlight to a rim of thickness `in.slider`.
    if (in.bubble > 4.5 && in.bubble < 5.5) {
        let width = max(in.slider, 0.5);
        let feather = max(in.shadow_sigma, 0.0001);
        let r = max(in.corner_radius, 0.0);
        let rel = in.frag_pos - in.quad_pos;
        let d_shape = rounded_box_sdf(rel, in.quad_size, r);

        // Outward-pointing surface normal via central differences on the SDF.
        let eps: f32 = 1.0;
        let gx = rounded_box_sdf(rel + vec2<f32>(eps, 0.0), in.quad_size, r)
               - rounded_box_sdf(rel - vec2<f32>(eps, 0.0), in.quad_size, r);
        let gy = rounded_box_sdf(rel + vec2<f32>(0.0, eps), in.quad_size, r)
               - rounded_box_sdf(rel - vec2<f32>(0.0, eps), in.quad_size, r);
        let grad_len = max(length(vec2<f32>(gx, gy)), 1e-6);
        let normal = vec2<f32>(gx, gy) / grad_len;

        // Lambert: 1.0 at the TL corner (normal ~ (-u,-u)), 0.0 at BR (normal ~ (+u,+u)).
        let u: f32 = 0.70710678;
        let lambert = max(0.0, -u * (normal.x + normal.y));

        // Inside-rim pulse: peak at d_shape = 0 (edge), decay inward over `width`.
        let dist_inside = max(0.0, -d_shape);
        let rim = exp(-(dist_inside * dist_inside) / (width * width));

        // AA at the outer edge so the rim doesn't bleed outside the shape.
        let inside_mask = 1.0 - smoothstep(-feather, 0.0, d_shape);

        let alpha = in.color.a * lambert * rim * inside_mask;
        if (alpha <= 0.0) {
            discard;
        }
        return vec4<f32>(in.color.rgb, alpha);
    }

    // bubble == 6.0 — Specular surface highlight.
    // A diagonal sheen across the interior: brightest at top-left, fading to zero at
    // bottom-right. Fakes a subtle bevel under the -45° key light. Clipped to inside the
    // shape; `in.slider` carries the falloff curve exponent.
    if (in.bubble > 5.5 && in.bubble < 6.5) {
        let curve = max(in.slider, 0.1);
        let feather = max(in.shadow_sigma, 0.0001);
        let r = max(in.corner_radius, 0.0);
        let rel = in.frag_pos - in.quad_pos;
        let d_shape = rounded_box_sdf(rel, in.quad_size, r);
        let inside_mask = 1.0 - smoothstep(-feather, 0.0, d_shape);
        if (inside_mask <= 0.0) {
            discard;
        }
        let inv_size = vec2<f32>(
            1.0 / max(in.quad_size.x, 1.0),
            1.0 / max(in.quad_size.y, 1.0),
        );
        let uv = rel * inv_size;
        // 0 at TL, 1 at BR — invert and curve for a light-facing gradient.
        let diag = clamp((uv.x + uv.y) * 0.5, 0.0, 1.0);
        let lambert = pow(1.0 - diag, curve);
        return vec4<f32>(in.color.rgb, in.color.a * lambert * inside_mask);
    }

    // Check if this is a border (encoded as negative corner_radius) - check this FIRST
    // Border width is encoded in color.a when corner_radius is negative
    if (in.corner_radius < 0.0) {
        let border_width = in.color.a;
        let outer_radius = -in.corner_radius;
        let inner_size = in.quad_size - vec2<f32>(border_width * 2.0, border_width * 2.0);
        let inner_radius = max(0.0, outer_radius - border_width);
        
        // Calculate position relative to quad
        let rel_pos = in.frag_pos - in.quad_pos;
        
        // Calculate SDF for outer and inner rounded boxes
        // Inner box is centered within outer box (offset by border_width)
        let outer_dist = rounded_box_sdf(rel_pos, in.quad_size, outer_radius);
        let inner_rel_pos = rel_pos - vec2<f32>(border_width, border_width);
        let inner_dist = rounded_box_sdf(inner_rel_pos, inner_size, inner_radius);
        
        // Border is where we're inside outer but outside inner (tight AA for crisp edges)
        let aa: f32 = 0.5;
        let outer_alpha = 1.0 - smoothstep(-aa, 0.0, outer_dist);
        let inner_alpha = smoothstep(-aa, 0.0, inner_dist);
        let border_alpha = outer_alpha * inner_alpha;
        
        return vec4<f32>(in.color.rgb, border_alpha);
    }
    
    // Skip SDF if corner_radius is 0
    if (in.corner_radius <= 0.0) {
        return in.color;
    }
    
    // Calculate position relative to quad center
    let center = in.quad_pos + in.quad_size * 0.5;
    let to_center = in.frag_pos - center;
    
    // Check if this is a glow effect (corner_radius == half of smaller dimension)
    let min_dim = min(in.quad_size.x, in.quad_size.y);
    let is_glow = abs(in.corner_radius - min_dim * 0.5) < 0.5;
    
    if (is_glow) {
        // Elliptical glow with radial falloff
        let semi_axes = in.quad_size * 0.5;
        
        // Normalized distance to center (elliptical distance)
        let normalized_offset = to_center / semi_axes;
        let ellipse_dist = length(normalized_offset);
        
        if (ellipse_dist <= 1.0) {
            // Inside ellipse - radial falloff. Phosphor glow uses strong center weight (cubic).
            // Contact shadows (`bubble == 2`) need visible alpha near the outer rim (cubic would crush it).
            let normalized = 1.0 - ellipse_dist;
            var glow_intensity: f32;
            var edge_fade: f32;
            if (in.bubble > 1.5) {
                glow_intensity = pow(max(normalized, 0.0), 0.52);
                edge_fade = 1.0 - smoothstep(0.82, 1.0, ellipse_dist);
            } else {
                glow_intensity = normalized * normalized * normalized;
                edge_fade = 1.0 - smoothstep(0.9, 1.0, ellipse_dist);
            }
            return vec4<f32>(in.color.rgb, in.color.a * glow_intensity * edge_fade);
        } else {
            // Outside ellipse
            discard;
        }
    } else {
        // Regular filled rounded rectangle
        var rel_pos = in.frag_pos - in.quad_pos;
        var quad_size = in.quad_size;
        // Liquid slider: trailing SDF. Pill follows the leading edge; tail width (radius along normal to motion)
        // tapers with distance from head. No peanut pinch.
        var dist: f32;
        if (in.slider > 0.5) {
            let stretch_scale: f32 = 0.14;
            let max_stretch: f32 = 48.0;     // must match SLIDER_MAX_STRETCH in header.rs
            let logical_size = vec2<f32>(in.quad_size.x - 2.0 * max_stretch, in.quad_size.y);
            let logical_origin = in.quad_pos + vec2<f32>(max_stretch, 0.0);
            rel_pos = in.frag_pos - logical_origin;
            let vel = uniforms.slider_velocity;
            let stretch_amount = min(abs(vel) * stretch_scale, max_stretch);
            let total_length = logical_size.x + stretch_amount;
            if (vel < 0.0) {
                rel_pos = rel_pos + vec2<f32>(stretch_amount, 0.0);
            }
            let center_y = logical_size.y * 0.5;
            let base_radius = logical_size.y * 0.5;   // full pill half-height at head (normal to motion)
            let head_x = select(0.0, total_length, vel >= 0.0);
            let closest_x = clamp(rel_pos.x, 0.0, total_length);
            let distance_from_head = abs(closest_x - head_x);
            let d_norm = distance_from_head / total_length;  // 0 at head, 1 at tail
            let inv_sq = 1.0 / (1.0 + d_norm * d_norm);     // slight inverse-square convexity (1 at head, ~0.5 at tail)
            let tail_falloff_raw = 0.2 + 1.6 * (inv_sq - 0.5);  // remap to [0.2, 1], min at tail
            let velocity_ratio = stretch_amount / max_stretch;  // 0 at rest, 1 at max stretch → no tail when at rest
            let tail_falloff = mix(1.0, tail_falloff_raw, velocity_ratio);
            let radius_at_closest = base_radius * tail_falloff;
            let to_segment = rel_pos - vec2<f32>(closest_x, center_y);
            dist = length(to_segment) - radius_at_closest;
        } else {
            dist = rounded_box_sdf(rel_pos, quad_size, in.corner_radius);
        }
        
        // Anti-aliased edge. Use a tight band (-0.5..0) for crisp edges; wider (-1..0) can look fuzzy when scaled.
        // To diagnose blurriness: ensure projection is 1:1 with physical pixels; avoid scaling the backbuffer.
        let aa_width: f32 = 0.5;
        let alpha = 1.0 - smoothstep(-aa_width, 0.0, dist);
        
        return vec4<f32>(in.color.rgb, in.color.a * alpha);
    }
}

