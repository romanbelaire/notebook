# UI Components

The `ui/components/` module provides container components.

## Root

The `Root` component is the top-level container:

```rust
pub struct Root {
    pub children: Vec<Box<dyn Renderable>>,
    pub rect: Rect,
}
```

### Methods

- `new(viewport_size)`: Create new root
- `add_child(child)`: Add child component

## VStack

Vertical stack layout:

```rust
pub struct VStack {
    pub children: Vec<Box<dyn Renderable>>,
    pub spacing: f32,
    pub alignment: Alignment,
}
```

Stacks children vertically with spacing.

## HStack

Horizontal stack layout (if implemented):

```rust
pub struct HStack {
    pub children: Vec<Box<dyn Renderable>>,
    pub spacing: f32,
    pub alignment: Alignment,
}
```

Stacks children horizontally with spacing.

## SectionList

Reusable section list built on `ScrollView`: scroll, hover highlight, selection border, and collapsible row actions (expand handle → edit/delete buttons). Use one `SectionList` per sidebar section (Conversations, Documents, Insights).

```rust
pub struct SectionList {
    pub scroll_view: ScrollView,
    pub item_height: f32,
    /// Which row has expanded actions (handle clicked). None = all collapsed.
    pub expanded_index: Option<usize>,
    // expand_animations (internal)
}
```

### Construction

- `new(position, size, item_height)`: Create a section list with the given rect and fixed row height.

### Methods

- **Update (call every frame)**
  - `update(dt, item_count)`: Update scroll, highlight, selection, and expand animations.
- **Expand / collapse**
  - `get_expand_animation(index)`: Expand value for a row (0.0 = collapsed, 1.0 = expanded).
- **Hit testing & layout**
  - `get_item_at(pos, item_count)`: Index of item at position (absolute coords), or `None`.
  - `item_y_for_index(index)`: World-space Y for highlight/selection bar for that item.
  - `contains(pos)`: Whether the list contains the point.
- **Scroll**
  - `scroll(delta)`: Scroll by delta.
- **Highlight**
  - `set_highlight_target(y)` / `clear_highlight()`: Hover highlight.
- **Selection**
  - `set_selection_border_target(y)` / `clear_selection_border()`: Selection border.
- **Layout**
  - `set_content_height(height)`: Total content height for scroll extent.
  - `set_position_size(position, size)`: Update position and size (delegates to `scroll_view`).

## Related Documentation

- [Component System](../architecture/components.md)
- [Layout System](../architecture/layout.md)

