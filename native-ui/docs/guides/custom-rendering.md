# Custom Rendering

This guide explains how to add custom rendering logic to components.

## Rendering Approaches

### Quad Rendering

For simple UI elements, use quads:

```rust
let quad = Quad {
    position: Vec2::new(x, y),
    size: Vec2::new(width, height),
    color: Vec4::new(r, g, b, a),
    corner_radius: 8.0,
};
quad.push_vertices_to(vertices);
```

### Text Rendering

Queue text for vello rendering:

```rust
renderer.queue_text(
    &text,
    position,
    color,
    font_size,
);
```

### Icon Rendering

Queue icons for vello rendering:

```rust
renderer.queue_icon(
    &icon_name,
    position,
    size,
    color,
);
```

## Custom Vertex Generation

For complex shapes, generate vertices manually:

```rust
// Generate triangle vertices
vertices.push(Vertex {
    position: [x1, y1],
    color: [r, g, b, a],
    quad_pos: [x, y],
    quad_size: [width, height],
    corner_radius: 0.0,
    _padding: [0.0; 3],
});
// ... more vertices
```

## Scissor Rects

Use scissor rects for clipping:

```rust
renderer.push_scissor(ScissorRect::from_rect(&rect, viewport_height));
// Render content (clipped)
renderer.pop_scissor();
```

## Related Documentation

- [Renderer](../modules/gfx/renderer.md)
- [Rendering Pipeline](../architecture/rendering.md)
- [Types](../modules/gfx/types.md)

