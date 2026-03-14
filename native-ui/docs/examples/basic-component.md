# Basic Component Example

This example shows how to create a simple button component.

## Component Definition

```rust
use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::ui::core::Rect;
use crate::ui::components::Renderable;
use crate::app::App;
use glam::{Vec2, Vec4};

pub struct Button {
    rect: Rect,
    text: String,
    color: Vec4,
    hovered: bool,
}

impl Button {
    pub fn new(text: String) -> Self {
        Self {
            rect: Rect::new(0.0, 0.0, 100.0, 40.0),
            text,
            color: Vec4::new(0.2, 0.4, 0.8, 1.0),
            hovered: false,
        }
    }
}
```

## Renderable Implementation

```rust
impl Renderable for Button {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
        renderer.validate_component("button", None, "Button");
        
        renderer.push_scissor(ScissorRect::from_rect(&self.rect, renderer.viewport_height()));
        
        // Render background
        let bg_color = if self.hovered {
            Vec4::new(0.3, 0.5, 0.9, 1.0)
        } else {
            self.color
        };
        
        let bg = Quad {
            position: self.rect.position(),
            size: self.rect.size(),
            color: bg_color,
            corner_radius: 8.0,
        };
        bg.push_vertices_to(vertices);
        
        // Render text
        let text_pos = Vec2::new(
            self.rect.x + self.rect.width / 2.0,
            self.rect.y + self.rect.height / 2.0,
        );
        renderer.queue_text(
            &self.text,
            text_pos,
            Vec4::new(1.0, 1.0, 1.0, 1.0),
            16.0,
        );
        
        renderer.pop_scissor();
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect) {
        self.rect = available_rect;
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(100.0, 40.0)
    }
}
```

## Usage

```rust
// In app.rs or component initialization
let button = Button::new("Click Me".to_string());
root.add_child(Box::new(button));
```

## Related Documentation

- [Creating Components](../guides/creating-components.md)
- [Component System](../architecture/components.md)

