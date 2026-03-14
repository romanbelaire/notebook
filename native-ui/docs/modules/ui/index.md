# UI Module

The `ui/` module provides the UI component system for the Notebook Native UI.

## Module Structure

```
ui/
├── core.rs              # Core primitives (Rect, Alignment, etc.)
├── components/          # Container components
├── window.rs            # Window abstraction
├── button.rs            # Button component
├── text.rs              # Text component
├── text_input.rs        # Text input component
├── text_editor.rs       # Rich text editor
├── scroll_view.rs       # Scrollable container
├── section_list.rs      # Section list (scroll, highlight, selection, expand actions)
├── chat_window.rs       # Chat window
├── library_window.rs    # Library window
├── settings_window.rs   # Settings window
├── style.rs             # Styling system
└── events.rs            # Event handling
```

## Key Components

### Core Primitives

- **Rect**: Rectangle with position and size
- **Alignment**: Alignment options (Start, Center, End)
- **Direction**: Layout direction (Horizontal, Vertical)

See [Core Documentation](core.md) for details.

### Container Components

- **Root**: Top-level container
- **VStack**: Vertical stack layout
- **HStack**: Horizontal stack layout (if implemented)
- **SectionList**: Scrollable list per section (scroll, highlight, selection, collapsible row actions)

See [Components Documentation](components.md) for details.

### Window Components

- **SubWindow**: Base window abstraction
- **ChatWindow**: Chat interface
- **LibraryWindow**: Document library
- **SettingsWindow**: Settings panel

See [Windows Documentation](windows.md) for details.

### Input Components

- **Button**: Clickable button
- **TextInput**: Text input field
- **TextEditor**: Rich text editor
- **Dropdown**: Dropdown menu

See [Input Documentation](input.md) for details.

## Related Documentation

- [Component System](../architecture/components.md)
- [Layout System](../architecture/layout.md)
- [Event System](../architecture/events.md)

