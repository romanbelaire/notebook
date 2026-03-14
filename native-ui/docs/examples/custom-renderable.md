# Custom Renderable Example

This example shows how to create a custom renderable with complex rendering.

## Component Definition

```rust
pub struct CustomShape {
    rect: Rect,
    points: Vec<Vec2>,
    color: Vec4,
}

impl CustomShape {
    pub fn new(points: Vec<Vec2>) -> Self {
        // Calculate bounding rect from points
        let min_x = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let min_y = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let max_x = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let max_y = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        
        Self {
            rect: Rect::new(min_x, min_y, max_x - min_x, max_y - min_y),
            points,
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        }
    }
}
```

## Custom Rendering

```rust
impl Renderable for CustomShape {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
        renderer.validate_component("custom_shape", None, "CustomShape");
        
        // Generate triangle fan vertices
        if self.points.len() >= 3 {
            let center = self.rect.center();
            for i in 0..self.points.len() - 1 {
                // Triangle: center, point[i], point[i+1]
                vertices.push(Vertex {
                    position: [center.x, center.y],
                    color: [self.color.x, self.color.y, self.color.z, self.color.w],
                    quad_pos: [0.0, 0.0],
                    quad_size: [0.0, 0.0],
                    corner_radius: 0.0,
                    _padding: [0.0; 3],
                });
                vertices.push(Vertex {
                    position: [self.points[i].x, self.points[i].y],
                    color: [self.color.x, self.color.y, self.color.z, self.color.w],
                    quad_pos: [0.0, 0.0],
                    quad_size: [0.0, 0.0],
                    corner_radius: 0.0,
                    _padding: [0.0; 3],
                });
                vertices.push(Vertex {
                    position: [self.points[i + 1].x, self.points[i + 1].y],
                    color: [self.color.x, self.color.y, self.color.z, self.color.w],
                    quad_pos: [0.0, 0.0],
                    quad_size: [0.0, 0.0],
                    corner_radius: 0.0,
                    _padding: [0.0; 3],
                });
            }
        }
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect) {
        // Update layout if needed
    }
    
    fn min_size(&self) -> Vec2 {
        self.rect.size()
    }
}
```

## Related Documentation

- [Custom Rendering](../guides/custom-rendering.md)
- [Renderer](../modules/gfx/renderer.md)

