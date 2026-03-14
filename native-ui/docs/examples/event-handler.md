# Event Handler Example

This example shows how to handle events in a component.

## Component with Event Handling

```rust
pub struct InteractiveButton {
    rect: Rect,
    text: String,
    clicked: bool,
}

impl InteractiveButton {
    pub fn new(text: String) -> Self {
        Self {
            rect: Rect::new(0.0, 0.0, 100.0, 40.0),
            text,
            clicked: false,
        }
    }
    
    pub fn handle_click(&mut self) {
        self.clicked = true;
        // Perform action
    }
}
```

## Event Handling in App

```rust
// In app.rs
impl App {
    pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if state == ElementState::Pressed && button == MouseButton::Left {
            if let Some(component) = self.hit_test(self.mouse_pos) {
                // Check if it's our button
                if let Some(button) = component.downcast_ref::<InteractiveButton>() {
                    button.handle_click();
                }
            }
        }
    }
}
```

## Related Documentation

- [Event Handling](../guides/event-handling.md)
- [Event System](../architecture/events.md)

