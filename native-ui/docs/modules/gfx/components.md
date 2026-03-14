# Graphics Components

The `gfx/components/` module contains rendering logic for specific UI components.

## Component Modules

### background.rs

Renders the application background.

### sidebar.rs

Renders the sidebar component.

### sidebar_content.rs

Renders sidebar content (conversations, documents, insights).

### header.rs

Renders the header bar.

### chat.rs

Renders the chat interface:
- Message bubbles
- Input field
- Scrollable message list

### glow.rs

Renders glow effects (e.g., sidebar edge glow).

### modals.rs

Renders modal dialogs:
- Insight modal
- PDF modal
- Notepad modal
- Chat info dialog

### toasts.rs

Renders toast notifications.

### library.rs

Renders the library window:
- Collections
- Papers
- Documents

### data.rs

Renders the data/insights panel.

### settings.rs

Renders the settings window.

### notepad.rs

Renders the notepad editor.

## Component Rendering Pattern

All components follow a similar pattern:

1. **Validate Component**: Call `renderer.validate_component()`
2. **Push Scissor**: Set clipping bounds
3. **Render Background**: Generate vertices for backgrounds
4. **Render Content**: Render text, icons, children
5. **Pop Scissor**: Remove clipping

## Example

```rust
pub fn render_sidebar(
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    renderer.validate_component("sidebar", Some("root"), "Sidebar");
    
    renderer.push_scissor(sidebar_rect);
    
    // Render background
    let bg = Quad { /* ... */ };
    bg.push_vertices_to(vertices);
    
    // Render content
    render_sidebar_content(renderer, app, vertices);
    
    renderer.pop_scissor();
}
```

## Related Documentation

- [Renderer](renderer.md)
- [Renderable Trait](renderable.md)
- [Component System](../architecture/components.md)

