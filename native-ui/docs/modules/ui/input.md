# Input Components

The `ui/` module provides input components.

## Button

Clickable button:

```rust
pub struct Button {
    pub text: String,
    pub rect: Rect,
    pub state: ButtonState,
}
```

### ButtonState

```rust
pub enum ButtonState {
    Normal,
    Hovered,
    Pressed,
    Disabled,
}
```

## TextInput

Text input field:

```rust
pub struct TextInput {
    pub text: String,
    pub placeholder: String,
    pub rect: Rect,
    pub focused: bool,
    pub cursor_position: usize,
    pub selection_start: Option<usize>,
    pub selection_end: Option<usize>,
}
```

## TextEditor

Rich text editor:

- Multi-line text editing
- Markdown support
- Cursor and selection

## Dropdown

Dropdown menu:

```rust
pub struct Dropdown {
    pub items: Vec<DropdownItem>,
    pub selected: Option<usize>,
    pub open: bool,
}
```

## Related Documentation

- [Component System](../architecture/components.md)
- [Event System](../architecture/events.md)

