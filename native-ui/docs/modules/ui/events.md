# UI Events

The `ui/events.rs` module provides event handling traits and utilities.

## Traits

### Hoverable

Components that handle hover events:

```rust
pub trait Hoverable {
    fn on_mouse_enter(&mut self, position: Vec2);
    fn on_mouse_leave(&mut self);
    fn contains(&self, pos: Vec2) -> bool;
}
```

### Draggable

Components that handle drag operations:

```rust
pub trait Draggable {
    fn on_drag_start(&mut self, position: Vec2, button: MouseButton);
    fn on_drag(&mut self, position: Vec2);
    fn on_drag_end(&mut self, position: Vec2);
}
```

### Focusable

Components that can receive focus:

```rust
pub trait Focusable {
    fn focus(&mut self);
    fn blur(&mut self);
    fn is_focused(&self) -> bool;
    fn focus_id(&self) -> String;
}
```

## State Types

### DragState

```rust
pub enum DragState {
    None,
    Starting { button: MouseButtonType, start_pos: Vec2 },
    Dragging { button: MouseButtonType, start_pos: Vec2 },
}
```

### HoverState

```rust
pub struct HoverState {
    pub hovered_component_id: Option<String>,
    pub last_hovered_component_id: Option<String>,
}
```

### FocusState

```rust
pub struct FocusState {
    pub focused_component_id: Option<String>,
    pub focusable_components: Vec<String>,
}
```

## Related Documentation

- [Event System](../architecture/events.md)
- [Component System](../architecture/components.md)

