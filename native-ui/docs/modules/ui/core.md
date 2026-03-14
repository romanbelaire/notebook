# UI Core

The `ui/core.rs` module provides core UI primitives and layout system.

## Rect

The `Rect` struct represents a rectangle:

```rust
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}
```

### Methods

- `new(x, y, width, height)`: Create new rect
- `from_pos_size(position, size)`: Create from position and size
- `position()`: Get position as Vec2
- `size()`: Get size as Vec2
- `center()`: Get center point
- `right()`: Get right edge
- `bottom()`: Get bottom edge
- `contains_point(point)`: Check if point is inside
- `intersects(other)`: Check if intersects with other rect
- `is_visible(viewport)`: Check if visible in viewport
- `inset(padding)`: Create inset rect
- `inset_by(left, top, right, bottom)`: Create inset with different padding

## Alignment

```rust
pub enum Alignment {
    Start,   // Left (horizontal) or Top (vertical)
    Center,  // Center
    End,     // Right (horizontal) or Bottom (vertical)
}
```

## Direction

```rust
pub enum Direction {
    Horizontal,  // Left to right
    Vertical,    // Top to bottom
}
```

## Layout Helpers

The `layout` module provides layout helper functions:

- `center_x(parent, child_width)`: Center horizontally
- `center_y(parent, child_height)`: Center vertically
- `center(parent, child_size)`: Center both
- `align_right(parent, child_width, padding)`: Align right
- `align_bottom(parent, child_height, padding)`: Align bottom
- `stack_vertical(parent, heights, spacing, padding)`: Stack vertically
- `stack_horizontal(parent, widths, spacing, padding)`: Stack horizontally

## Text Layout

The `text` module provides text layout helpers:

- `TextMetrics`: Text measurement data
- `top_y(rect, metrics)`: Calculate Y position for text
- `baseline_y(rect, metrics)`: Calculate baseline Y
- `left_aligned(rect, font_size, padding)`: Left-aligned position
- `center_aligned(rect, text_width, font_size)`: Center-aligned position
- `right_aligned(rect, text_width, font_size, padding)`: Right-aligned position

## Container System

The `container` module provides section-based layouts:

- `Section`: Section with title and items
- `SectionStack`: Vertical stack of sections

## Cursor Helpers

The `cursor` module provides cursor positioning:

- `rect_at_position(input_rect, cursor_x, font_size)`: Calculate cursor rect

## Related Documentation

- [Layout System](../architecture/layout.md)
- [Component System](../architecture/components.md)

