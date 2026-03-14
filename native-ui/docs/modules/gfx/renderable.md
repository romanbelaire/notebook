# Renderable Trait

The `Renderable` trait defines the interface for all renderable UI components.

## Trait Definition

```rust
pub trait Renderable {
    /// Render this component and append vertices to the provided vector
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>);
    
    /// Get the bounding rectangle of this component
    fn bounds(&self) -> Rect;
    
    /// Update layout based on parent constraints
    fn update_layout(&mut self, available_rect: Rect);
    
    /// Get the minimum size this component requires
    fn min_size(&self) -> Vec2;
}
```

## Methods

### render()

Renders the component:

```rust
fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>)
```

Should:
1. Validate component via `renderer.validate_component()`
2. Push scissor rect if needed
3. Generate vertices for backgrounds/borders
4. Queue text via `renderer.queue_text()`
5. Queue icons via `renderer.queue_icon()`
6. Render children recursively
7. Pop scissor rect

### bounds()

Returns the component's bounding rectangle:

```rust
fn bounds(&self) -> Rect
```

Used for:
- Hit testing
- Layout calculations
- Visibility culling

### update_layout()

Updates component layout when parent changes:

```rust
fn update_layout(&mut self, available_rect: Rect)
```

Should:
1. Update own bounds
2. Calculate child layouts
3. Call `update_layout()` on children

### min_size()

Returns minimum size:

```rust
fn min_size(&self) -> Vec2
```

Used for:
- Layout constraints
- Minimum window sizes

## Implementation Example

```rust
impl Renderable for MyComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
        renderer.validate_component("my_component", None, "MyComponent");
        
        renderer.push_scissor(self.rect);
        
        // Render background
        let bg = Quad {
            position: self.rect.position(),
            size: self.rect.size(),
            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            corner_radius: 8.0,
        };
        bg.push_vertices_to(vertices);
        
        // Render children
        for child in &self.children {
            child.render(renderer, app, vertices);
        }
        
        renderer.pop_scissor();
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect) {
        self.rect = available_rect;
        for child in &mut self.children {
            child.update_layout(available_rect);
        }
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(100.0, 50.0)
    }
}
```

## Best Practices

1. Always validate components during rendering
2. Use scissor rects for clipping
3. Update layout when parent changes
4. Return accurate bounds for hit testing
5. Specify minimum sizes for layout constraints

## Related Documentation

- [Component System](../architecture/components.md)
- [Renderer](renderer.md)
- [Creating Components Guide](../../guides/creating-components.md)

