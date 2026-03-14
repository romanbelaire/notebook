use glam::{Vec2, Vec4};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub quad_pos: [f32; 2],
    pub quad_size: [f32; 2],
    pub corner_radius: f32,
    pub bubble: f32,  // 1.0 = apply bubble border displacement, 0.0 = normal
    pub slider: f32,   // 1.0 = apply liquid slider stretch, 0.0 = normal
    pub _padding: [f32; 1],  // Pad to 16-byte alignment
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
    out.push(Vertex { position: [p0.x, p0.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, _padding: [0.0; 1] });
    out.push(Vertex { position: [p1.x, p1.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, _padding: [0.0; 1] });
    out.push(Vertex { position: [p2.x, p2.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, _padding: [0.0; 1] });
    out.push(Vertex { position: [p1.x, p1.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, _padding: [0.0; 1] });
    out.push(Vertex { position: [p2.x, p2.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, _padding: [0.0; 1] });
    out.push(Vertex { position: [p3.x, p3.y], color: c, quad_pos: [quad_pos.x, quad_pos.y], quad_size: [quad_size.x, quad_size.y], corner_radius: 0.0, bubble: 0.0, slider: 0.0, _padding: [0.0; 1] });
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
                _padding: [0.0; 1],
            },
            Vertex {
                position: [p.x + s.x, p.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                _padding: [0.0; 1],
            },
            Vertex {
                position: [p.x, p.y + s.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                _padding: [0.0; 1],
            },
            Vertex {
                position: [p.x + s.x, p.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                _padding: [0.0; 1],
            },
            Vertex {
                position: [p.x + s.x, p.y + s.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                _padding: [0.0; 1],
            },
            Vertex {
                position: [p.x, p.y + s.y],
                color: [c.x, c.y, c.z, c.w],
                quad_pos: [p.x, p.y],
                quad_size: [s.x, s.y],
                corner_radius: r,
                bubble: b,
                slider: sl,
                _padding: [0.0; 1],
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
            _padding: [0.0; 1],
        });
        vertices.push(Vertex {
            position: [p.x + s.x, p.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            _padding: [0.0; 1],
        });
        vertices.push(Vertex {
            position: [p.x, p.y + s.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            _padding: [0.0; 1],
        });
        vertices.push(Vertex {
            position: [p.x + s.x, p.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            _padding: [0.0; 1],
        });
        vertices.push(Vertex {
            position: [p.x + s.x, p.y + s.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            _padding: [0.0; 1],
        });
        vertices.push(Vertex {
            position: [p.x, p.y + s.y],
            color: [c.x, c.y, c.z, c.w],
            quad_pos: [p.x, p.y],
            quad_size: [s.x, s.y],
            corner_radius: r,
            bubble: b,
            slider: sl,
            _padding: [0.0; 1],
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

