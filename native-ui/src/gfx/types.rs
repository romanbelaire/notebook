use glam::{Vec2, Vec4};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub quad_pos: [f32; 2],
    pub quad_size: [f32; 2],
    pub corner_radius: f32,
    // `bubble` selects the fragment-shader branch:
    //   0.0 = normal fill / border
    //   1.0 = velocity-driven bubble border
    //   2.0 = contact shadow (elliptical glow)
    //   3.0 = outer drop shadow (erf SDF)
    //   4.0 = inner drop shadow  (erf SDF, clipped inside shape)
    //   5.0 = specular border highlight (SDF-gradient Lambert on the rim)
    //   6.0 = specular surface highlight (diagonal sheen across the interior)
    pub bubble: f32,
    /// For `bubble == 0/1`, a boolean (`> 0.5`) enabling the liquid-slider stretch path.
    /// For `bubble == 4/5/6`, repurposed as a scalar parameter (offset magnitude / border
    /// width / curve exponent — see [`ui_shader.wgsl`](../shaders/ui_shader.wgsl)).
    pub slider: f32,
    /// Gaussian feather / blur used by the `bubble == 3/4/5/6` lighting branches.
    /// Unused (0.0) for other branches. Also keeps the struct at 16-byte alignment.
    pub shadow_sigma: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextVertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // quad_pos
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // quad_size
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>() + std::mem::size_of::<[f32; 2]>()) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // corner_radius
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 2]>()) as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
                // bubble
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
                // slider
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<f32>() * 2) as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32,
                },
                // shadow_sigma (gaussian feather; used by bubble == 3/4/5/6 lighting branches in ui_shader.wgsl)
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 4]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<[f32; 2]>() + std::mem::size_of::<f32>() * 3) as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

pub struct Quad {
    pub position: Vec2,
    pub size: Vec2,
    pub color: Vec4,
    pub corner_radius: f32,
    /// If true, fragment shader applies velocity-driven border displacement (bubble effect).
    pub bubble_effect: bool,
    /// If true, fragment shader applies liquid stretch in direction of motion (nav slider).
    pub slider_effect: bool,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            size: Vec2::ZERO,
            color: Vec4::ZERO,
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        }
    }
}

/// Emit 6 vertices for a line segment from a to b with given thickness and color (for edges).
#[inline]
pub fn segment_to_vertices(a: Vec2, b: Vec2, thickness: f32, color: Vec4, out: &mut Vec<Vertex>) {
    let d = b - a;
    let len = d.length();
    if len < 1e-6 {
        return;
    }
    let perp = Vec2::new(-d.y / len, d.x / len) * (thickness * 0.5f32);
    let p0 = a - perp;
    let p1 = a + perp;
    let p2 = b + perp;
    let p3 = b - perp;
    let min_x = p0.x.min(p1.x).min(p2.x).min(p3.x);
    let min_y = p0.y.min(p1.y).min(p2.y).min(p3.y);
    let max_x = p0.x.max(p1.x).max(p2.x).max(p3.x);
    let max_y = p0.y.max(p1.y).max(p2.y).max(p3.y);
    let quad_pos = Vec2::new(min_x, min_y);
    let quad_size = Vec2::new(max_x - min_x, max_y - min_y);
    let c = [color.x, color.y, color.z, color.w];
    out.push(Vertex { position: [p0.x, p0.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, shadow_sigma: 0.0 });
    out.push(Vertex { position: [p1.x, p1.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, shadow_sigma: 0.0 });
    out.push(Vertex { position: [p2.x, p2.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, shadow_sigma: 0.0 });
    out.push(Vertex { position: [p1.x, p1.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, shadow_sigma: 0.0 });
    out.push(Vertex { position: [p2.x, p2.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, shadow_sigma: 0.0 });
    out.push(Vertex { position: [p3.x, p3.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, shadow_sigma: 0.0 });
}

/// Build a seamless thick ribbon from a polyline.
///
/// At each interior point the perpendicular is computed from the bisector tangent
/// `pts[i+1] − pts[i-1]`, giving smooth mitered joins with no gaps between segments.
fn ribbon_from_pts(pts: &[Vec2], thickness: f32, color: Vec4, out: &mut Vec<Vertex>) {
    let n = match pts.len().checked_sub(1) {
        Some(n) if n > 0 => n,
        _ => return,
    };
    let half = thickness * 0.5;
    let c = [color.x, color.y, color.z, color.w];

    let perp_unit = |a: Vec2, b: Vec2| -> Vec2 {
        let d = b - a;
        let len = d.length();
        if len < 1e-6 {
            return Vec2::new(0.0, 1.0);
        }
        Vec2::new(-d.y / len, d.x / len)
    };

    let mut left = Vec::with_capacity(n + 1);
    let mut right = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let pa = if i == 0 { pts[0] } else { pts[i - 1] };
        let pb = if i == n { pts[n] } else { pts[i + 1] };
        let perp = perp_unit(pa, pb) * half;
        left.push(pts[i] - perp);
        right.push(pts[i] + perp);
    }

    for i in 0..n {
        let (l0, r0, l1, r1) = (left[i], right[i], left[i + 1], right[i + 1]);
        let mk = |p: Vec2| Vertex {
            position: [p.x, p.y],
            color: c,
            quad_pos: [0.0, 0.0],
            quad_size: [0.0, 0.0],
            corner_radius: 0.0,
            bubble: 0.0,
            slider: 0.0,
            shadow_sigma: 0.0,
        };
        out.push(mk(l0));
        out.push(mk(r0));
        out.push(mk(l1));
        out.push(mk(r0));
        out.push(mk(r1));
        out.push(mk(l1));
    }
}

/// Tessellate a cubic bezier curve into a seamless thick ribbon.
pub fn bezier_to_vertices(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    thickness: f32,
    color: Vec4,
    steps: usize,
    out: &mut Vec<Vertex>,
) {
    let n = steps.max(1);
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let mt = 1.0 - t;
        pts.push(
            p0 * (mt * mt * mt)
                + p1 * (3.0 * mt * mt * t)
                + p2 * (3.0 * mt * t * t)
                + p3 * (t * t * t),
        );
    }
    ribbon_from_pts(&pts, thickness, color, out);
}

/// Tessellate a "Manhattan elbow" connector as a seamless thick ribbon.
///
/// The path exits `a` vertically, makes a rounded 90° corner into a horizontal segment,
/// makes a second rounded 90° corner, then arrives at `b` vertically.  `r` is the desired
/// corner radius in screen pixels; it is automatically clamped so that:
/// - the horizontal segment never goes negative  (`r ≤ |dx| / 2`)
/// - the vertical run-in segments never go negative (`r ≤ |dy| / 4`)
///
/// When the horizontal offset is zero the path degenerates to a straight vertical segment.
/// Each quarter-circle arc is approximated by a cubic bezier (K = 0.5523, max radial error
/// < 0.03 %), giving C1-continuous joins with the adjacent straight segments.
pub fn elbow_to_vertices(
    a: Vec2,
    b: Vec2,
    r_max: f32,
    arc_steps: usize,
    thickness: f32,
    color: Vec4,
    out: &mut Vec<Vertex>,
) {
    let dx = b.x - a.x;
    let dy = b.y - a.y;

    let sign_y = if dy >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let mid_y = (a.y + b.y) / 2.0;

    // Effective corner radius, clamped by available horizontal and vertical space.
    // Using the same constraints in both modes keeps `knee_y` continuous across the
    // threshold so there is no shape discontinuity when the connector switches modes.
    let effective_r = r_max
        .min(dx.abs() / 2.0)   // room for two arcs side-by-side
        .min(dy.abs() / 4.0);  // room for two vertical run-in segments

    // `knee_y`: the Y level at which the path transitions to horizontal.
    // Matches the elbow's horizontal segment height, and degrades to mid_y (symmetric
    // S-curve) when effective_r → 0, which in turn degrades to a straight line at dx = 0.
    let knee_y = mid_y + effective_r * sign_y;

    // When the horizontal span is smaller than the two corner arc diameters the
    // exact elbow would produce a zero-length horizontal segment and a visible kink.
    // Use a smooth bezier whose control points track the same knee_y, so the
    // transition at the threshold (|dx| = 2·r_max) is visually continuous.
    if dx.abs() < 2.0 * r_max {
        let cp1 = Vec2::new(a.x, knee_y);
        let cp2 = Vec2::new(b.x, knee_y);
        bezier_to_vertices(a, cp1, cp2, b, thickness, color, arc_steps, out);
        return;
    }

    // Constrain r so all five path segments have non-negative length.
    let r = r_max
        .min(dx.abs() / 2.0)         // horizontal segment ≥ 0
        .min(dy.abs() / 4.0);        // both vertical segments ≥ 0

    if r < 0.5 {
        segment_to_vertices(a, b, thickness, color, out);
        return;
    }

    let sign_x = if dx >= 0.0 { 1.0_f32 } else { -1.0_f32 };

    // Cubic bezier approximation constant for a quarter circle arc.
    // Gives < 0.03 % radial error — far below any visible tessellation stepping.
    const K: f32 = 0.5523;

    // Arc 1: exits a going vertically, enters the horizontal going horizontally.
    // Arc 1 control points:
    //   P0 tangent = (0, sign_y)  →  P1 = P0 + K·r·(0, sign_y)
    //   P3 tangent = (sign_x, 0) →  P2 = P3 − K·r·(sign_x, 0)
    let a1_p0 = Vec2::new(a.x, mid_y);
    let a1_p1 = Vec2::new(a.x, mid_y + K * r * sign_y);
    let a1_p2 = Vec2::new(a.x + r * sign_x * (1.0 - K), mid_y + r * sign_y);
    let a1_p3 = Vec2::new(a.x + r * sign_x, mid_y + r * sign_y);

    // Arc 2: exits the horizontal going horizontally, arrives at b going vertically.
    //   P0 tangent = (sign_x, 0)  →  P1 = P0 + K·r·(sign_x, 0)
    //   P3 tangent = (0, sign_y)  →  P2 = P3 − K·r·(0, sign_y)
    let a2_p0 = Vec2::new(b.x - r * sign_x, mid_y + r * sign_y);
    let a2_p1 = Vec2::new(b.x - r * sign_x * (1.0 - K), mid_y + r * sign_y);
    let a2_p2 = Vec2::new(b.x, mid_y + r * sign_y * (2.0 - K));
    let a2_p3 = Vec2::new(b.x, mid_y + 2.0 * r * sign_y);

    let eval = |p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32| -> Vec2 {
        let mt = 1.0 - t;
        p0 * (mt * mt * mt)
            + p1 * (3.0 * mt * mt * t)
            + p2 * (3.0 * mt * t * t)
            + p3 * (t * t * t)
    };

    let n = arc_steps.max(2);

    // Collect the full polyline: two straight segments + two arcs + horizontal segment,
    // all in a single contiguous Vec so ribbon_from_pts produces one seamless ribbon.
    let capacity = 2 + (n + 1) + 2 + (n + 1) + 2;
    let mut pts: Vec<Vec2> = Vec::with_capacity(capacity);

    // Segment 1: a → arc1 start (vertical run-in)
    pts.push(a);
    pts.push(a1_p0);

    // Arc 1 (skip t=0 which equals a1_p0 already pushed)
    for i in 1..=n {
        pts.push(eval(a1_p0, a1_p1, a1_p2, a1_p3, i as f32 / n as f32));
    }

    // Horizontal segment: arc1 end → arc2 start (only add arc2_start; arc1 end = last arc pt)
    if (a2_p0 - a1_p3).length() > 0.5 {
        pts.push(a2_p0);
    }

    // Arc 2 (skip t=0 which equals a2_p0)
    for i in 1..=n {
        pts.push(eval(a2_p0, a2_p1, a2_p2, a2_p3, i as f32 / n as f32));
    }

    // Segment 5: arc2 end → b (vertical run-out)
    pts.push(b);

    ribbon_from_pts(&pts, thickness, color, out);
}

impl Quad {
    pub fn to_vertices(&self) -> [Vertex; 6] {
        let p = self.position;
        let s = self.size;
        let c = self.color;
        let r = self.corner_radius;
        let b = if self.bubble_effect { 1.0f32 } else { 0.0f32 };
        let sl = if self.slider_effect { 1.0f32 } else { 0.0f32 };

        [
            Vertex {
                position: [p.x, p.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                shadow_sigma: 0.0,
            },
            Vertex {
                position: [p.x + s.x, p.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                shadow_sigma: 0.0,
            },
            Vertex {
                position: [p.x, p.y + s.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                shadow_sigma: 0.0,
            },
            Vertex {
                position: [p.x + s.x, p.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                shadow_sigma: 0.0,
            },
            Vertex {
                position: [p.x + s.x, p.y + s.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                shadow_sigma: 0.0,
            },
            Vertex {
                position: [p.x, p.y + s.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                shadow_sigma: 0.0,
            },
        ]
    }
    
    /// Push vertices directly to a vector to avoid temporary array allocation
    /// This is more efficient than to_vertices() when you have a mutable vector
    #[inline]
    pub fn push_vertices_to(&self, vertices: &mut Vec<Vertex>) {
        let p = self.position;
        let s = self.size;
        let c = self.color;
        let r = self.corner_radius;
        let b = if self.bubble_effect { 1.0f32 } else { 0.0f32 };
        let sl = if self.slider_effect { 1.0f32 } else { 0.0f32 };

        vertices.push(Vertex {
            position: [p.x, p.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            shadow_sigma: 0.0,
        });
        vertices.push(Vertex {
            position: [p.x + s.x, p.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            shadow_sigma: 0.0,
        });
        vertices.push(Vertex {
            position: [p.x, p.y + s.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            shadow_sigma: 0.0,
        });
        vertices.push(Vertex {
            position: [p.x + s.x, p.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            shadow_sigma: 0.0,
        });
        vertices.push(Vertex {
            position: [p.x + s.x, p.y + s.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            shadow_sigma: 0.0,
        });
        vertices.push(Vertex {
            position: [p.x, p.y + s.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            shadow_sigma: 0.0,
        });
    }
}

impl TextVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TextVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

