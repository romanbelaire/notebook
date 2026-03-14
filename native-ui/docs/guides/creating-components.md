# Creating Components

This guide walks through creating a new UI component for the Notebook Native UI.

## Step 1: Define the Component

Create a new file or add to an existing module:

```rust
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;
use crate::ui::core::Rect;
use crate::ui::components::Renderable;
use crate::app::App;
use glam::Vec2;

pub struct MyComponent {
    rect: Rect,
    children: Vec<Box<dyn Renderable>>,
    // Component-specific fields
    text: String,
    color: Vec4,
}
```

## Step 2: Implement Renderable

Implement the `Renderable` trait:

```rust
impl Renderable for MyComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
        // 1. Validate component
        renderer.validate_component("my_component", None, "MyComponent");
        
        // 2. Push scissor for clipping
        renderer.push_scissor(ScissorRect::from_rect(&self.rect, renderer.viewport_height()));
        
        // 3. Render background
        let bg = Quad {
            position: self.rect.position(),
            size: self.rect.size(),
            color: self.color,
            corner_radius: 8.0,
        };
        bg.push_vertices_to(vertices);
        
        // 4. Queue text
        renderer.queue_text(
            &self.text,
            Vec2::new(self.rect.x + 10.0, self.rect.y + 10.0),
            Vec4::new(1.0, 1.0, 1.0, 1.0),
            16.0,
        );
        
        // 5. Render children
        for child in &self.children {
            child.render(renderer, app, vertices);
        }
        
        // 6. Pop scissor
        renderer.pop_scissor();
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect) {
        self.rect = available_rect;
        // Update children layout
        for child in &mut self.children {
            child.update_layout(available_rect);
        }
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(100.0, 50.0)
    }
}
```

## Step 3: Add Constructor

Add a constructor:

```rust
impl MyComponent {
    pub fn new(text: String) -> Self {
        Self {
            rect: Rect::new(0.0, 0.0, 100.0, 50.0),
            children: Vec::new(),
            text,
            color: Vec4::new(0.2, 0.2, 0.2, 1.0),
        }
    }
    
    pub fn add_child(&mut self, child: Box<dyn Renderable>) {
        self.children.push(child);
    }
}
```

## Step 4: Add to Component Tree

Add the component to the component tree in `app.rs`:

```rust
// In App::new() or component initialization
root.add_child(Box::new(MyComponent::new("Hello".to_string())));
```

## Best Practices

1. **Always validate**: Call `renderer.validate_component()` during rendering
2. **Use scissor rects**: Clip rendering to component bounds
3. **Update layout**: Implement `update_layout()` to handle parent changes
4. **Specify min size**: Return accurate `min_size()` for layout constraints
5. **Compose**: Prefer composition over complex monolithic components

## Related Documentation

- [Component System](../architecture/components.md)
- [Renderable Trait](../modules/gfx/renderable.md)
- [Examples](../examples/basic-component.md)

