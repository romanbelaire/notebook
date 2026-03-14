# App Module

The `app.rs` module is the central coordinator of the application. It manages all UI windows, components, state, and event handling.

## Overview

The `App` struct is the main application state container that:

- Manages all UI windows and components
- Handles event routing from winit
- Coordinates state updates
- Manages async API operations via channels
- Maintains the component hierarchy root

## App Structure

```rust
pub struct App {
    // Windows
    pub windows: Vec<SubWindow>,
    pub header: HeaderWindow,
    pub sidebar: SidebarWindow,
    pub chat_window: Option<ChatWindow>,
    pub library_window: Option<LibraryWindow>,
    // ... other windows
    
    // State
    pub chat_state: ChatState,
    pub ui_state: UIState,
    pub settings_state: SettingsState,
    pub insights_state: InsightsState,
    
    // API
    pub api_client: ApiClient,
    
    // Component hierarchy
    pub root: Root,
    
    // Event state
    pub mouse_pos: Vec2,
    pub modifiers: ModifiersState,
    pub focused_input: Option<usize>,
    // ... more event state
}
```

## Initialization

The `App` is created in `App::new()`:

```rust
pub fn new(viewport_size: (u32, u32)) -> Self {
    let viewport = Vec2::new(viewport_size.0 as f32, viewport_size.1 as f32);
    
    // Initialize state
    let ui_state = UIState::new();
    let settings_state = SettingsPersistence::load_settings()
        .unwrap_or_else(|_| SettingsState::new());
    
    // Create windows
    let header = HeaderWindow::new(/* ... */);
    let sidebar = SidebarWindow::new(/* ... */);
    
    // Build component tree
    let mut root = Root::new(viewport);
    root.add_child(Box::new(SidebarComponent::new()));
    // ... add more components
    
    App { /* ... */ }
}
```

## Event Handling

The `App` handles events from winit:

### Mouse Events

```rust
pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
    // Handle mouse button events
}

pub fn on_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
    self.mouse_pos = Vec2::new(position.x as f32, position.y as f32);
    // Update hover state
}
```

### Keyboard Events

```rust
pub fn on_keyboard(&mut self, event: &KeyEvent) {
    // Handle keyboard events
    // Check shortcuts
    // Route to focused input
}

pub fn on_char_received(&mut self, ch: char) {
    // Handle text input
    if let Some(input) = self.focused_input {
        // Insert character
    }
}
```

### Window Events

```rust
pub fn resize(&mut self, size: (u32, u32)) {
    self.viewport_size = Vec2::new(size.0 as f32, size.1 as f32);
    // Update window layouts
    self.root.update_layout(Rect::new(0.0, 0.0, size.0 as f32, size.1 as f32));
}
```

## Update Loop

The `App` updates each frame:

```rust
pub fn update(&mut self, dt: f32) {
    // Update animations
    self.cursor_blink_timer += dt;
    
    // Update cursor animation
    self.cursor_position_animation.update(dt);
    
    // Update component layouts if needed
}
```

## API Integration

The `App` manages async API operations via channels:

```rust
pub fn check_api_responses(&mut self) {
    while let Ok(result) = self.api_response_receiver.try_recv() {
        match result {
            Ok(response) => {
                // Update state with response
                self.chat_state.add_message(response.message);
            }
            Err(e) => {
                // Handle error
            }
        }
    }
}
```

## Component Management

The `App` maintains the component hierarchy:

```rust
pub root: Root,
```

Components are added to the root during initialization and rendered each frame.

## Window Management

The `App` manages multiple windows:

- **HeaderWindow**: Top header bar
- **SidebarWindow**: Left sidebar
- **ChatWindow**: Main chat interface
- **LibraryWindow**: Document library
- **SettingsWindow**: Settings panel
- **NotepadWindow**: Notepad editor

Windows are created during initialization and updated on resize.

## State Management

The `App` coordinates state updates:

- **ChatState**: Conversation state
- **UIState**: UI preferences
- **SettingsState**: Application settings
- **InsightsState**: Insights data

State is updated in response to user actions and API responses.

## Related Documentation

- [Architecture Overview](../architecture/overview.md)
- [Component System](../architecture/components.md)
- [Event System](../architecture/events.md)
- [State Management](../architecture/state.md)

