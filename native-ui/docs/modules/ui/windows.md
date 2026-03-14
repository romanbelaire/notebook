# Window Components

The `ui/` module provides various window components.

## SubWindow

Base window abstraction:

```rust
pub struct SubWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub visible: bool,
}
```

## ChatWindow

Chat interface window:

- Message display
- Input field
- Scrollable message list

## LibraryWindow

Document library window:

- Collections list
- Papers list
- Documents list

## SettingsWindow

Settings panel:

- Settings sections
- Input fields
- Save/cancel buttons

## Related Documentation

- [Component System](../architecture/components.md)
- [App Module](../app.md)

