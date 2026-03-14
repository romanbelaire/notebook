# Event Handling

This guide explains how to handle events in components.

## Event Flow

Events flow from winit → App → Components:

1. **winit** receives system events
2. **main.rs** routes to App
3. **app.rs** routes to components
4. **Components** handle events

## Mouse Events

### Mouse Button

```rust
// In app.rs
pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
    if let Some(component) = self.hit_test(self.mouse_pos) {
        component.on_mouse_button(button, state);
    }
}
```

### Mouse Movement

```rust
pub fn on_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
    self.mouse_pos = Vec2::new(position.x as f32, position.y as f32);
    // Update hover state
}
```

## Keyboard Events

### Key Events

```rust
pub fn on_keyboard(&mut self, event: &KeyEvent) {
    // Check shortcuts
    if self.shortcut_registry.handle_event(event) {
        return;
    }
    
    // Route to focused input
    if let Some(input) = self.focused_input {
        input.handle_keyboard(event);
    }
}
```

### Text Input

```rust
pub fn on_char_received(&mut self, ch: char) {
    if let Some(input) = self.focused_input {
        input.insert_char(ch);
    }
}
```

## Focus Management

Components can receive focus:

```rust
// Set focus
app.focused_input = Some(input_id);

// Handle focus events
if component.is_focused() {
    // Handle keyboard input
}
```

## Related Documentation

- [Event System](../architecture/events.md)
- [UI Events](../modules/ui/events.md)

