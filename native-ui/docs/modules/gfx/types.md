# Graphics Types

The `gfx/types.rs` module defines core graphics data structures.

## Vertex

The `Vertex` struct represents a vertex in the UI shader:

```rust
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 2],      // Vertex position
    pub color: [f32; 4],          // RGBA color
    pub quad_pos: [f32; 2],       // Quad position (for rounded corners)
    pub quad_size: [f32; 2],      // Quad size (for rounded corners)
    pub corner_radius: f32,       // Corner radius
    pub _padding: [f32; 3],       // Padding for alignment
}
```

### Vertex Buffer Layout

```rust
impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static>
}
```

Defines the vertex buffer layout for wgpu.

## TextVertex

The `TextVertex` struct represents a vertex for text rendering:

```rust
#[repr(C)]
pub struct TextVertex {
    pub position: [f32; 2],        // Vertex position
    pub tex_coords: [f32; 2],     // Texture coordinates
    pub color: [f32; 4],         // RGBA color
}
```

## Quad

The `Quad` struct represents a rectangular UI element:

```rust
pub struct Quad {
    pub position: Vec2,          // Top-left position
    pub size: Vec2,              // Width and height
    pub color: Vec4,            // RGBA color
    pub corner_radius: f32,      // Corner radius for rounded corners
}
```

### Methods

#### to_vertices()

Converts quad to vertex array:

```rust
pub fn to_vertices(&self) -> [Vertex; 6]
```

Returns 6 vertices (2 triangles).

#### push_vertices_to()

Pushes vertices directly to vector:

```rust
pub fn push_vertices_to(&self, vertices: &mut Vec<Vertex>)
```

More efficient than `to_vertices()` when you have a mutable vector.

## ScissorRect

The `ScissorRect` struct represents a clipping rectangle:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScissorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
```

### Methods

#### from_rect()

Creates scissor rect from UI rect:

```rust
pub fn from_rect(rect: &Rect, viewport_height: f32) -> Self
```

#### intersect()

Intersects two scissor rects:

```rust
pub fn intersect(&self, other: &ScissorRect) -> ScissorRect
```

Used for nested clipping.

## Color Types

Colors are represented as `Vec4` (RGBA):
- `Vec4::new(r, g, b, a)` where values are 0.0 to 1.0
- Or use `glam::Vec4` directly

## Related Documentation

- [Renderer](renderer.md)
- [Renderable Trait](renderable.md)
- [UI Core Module](../ui/core.md)

