# Styling

This guide explains the styling system in the Notebook Native UI.

## Style Constants

Style constants are defined in `ui/style.rs`:

### Colors

```rust
pub mod bg {
    pub const PRIMARY: Vec4 = Vec4::new(0.1, 0.1, 0.1, 1.0);
    pub const SECONDARY: Vec4 = Vec4::new(0.2, 0.2, 0.2, 1.0);
    pub const INPUT: Vec4 = Vec4::new(0.15, 0.15, 0.15, 1.0);
    pub const INPUT_FOCUSED: Vec4 = Vec4::new(0.2, 0.2, 0.2, 1.0);
}

pub mod text {
    pub const PRIMARY: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0);
    pub const SECONDARY: Vec4 = Vec4::new(0.8, 0.8, 0.8, 1.0);
    pub const PLACEHOLDER: Vec4 = Vec4::new(0.5, 0.5, 0.5, 1.0);
}
```

### Font Sizes

```rust
pub mod font_size {
    pub const SMALL: f32 = 12.0;
    pub const NORMAL: f32 = 16.0;
    pub const LARGE: f32 = 20.0;
}
```

### Padding

```rust
pub mod padding {
    pub const SMALL: f32 = 8.0;
    pub const MEDIUM: f32 = 16.0;
    pub const LARGE: f32 = 24.0;
}
```

### Corner Radius

```rust
pub mod corner_radius {
    pub const SMALL: f32 = 4.0;
    pub const MEDIUM: f32 = 8.0;
    pub const LARGE: f32 = 12.0;
}
```

## Using Styles

Use style constants in components:

```rust
use crate::ui::style;

let bg = Quad {
    position: rect.position(),
    size: rect.size(),
    color: style::bg::PRIMARY,
    corner_radius: style::corner_radius::MEDIUM,
};
```

## Theming

Theming support may be added in the future. For now, use style constants directly.

## Related Documentation

- [Style Module](../modules/ui/style.md)

