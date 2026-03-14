# Object-Oriented UI Core System

## Overview
A standardized, modular UI system that provides consistent positioning, layout, and text rendering across the entire application.

## Problem Solved
Previously, UI code had:
- Inconsistent text positioning calculations scattered across components
- Hardcoded offsets and magic numbers for alignment
- No standard way to handle rectangles and bounds
- Manual baseline calculations that were often incorrect
- Cursor positioning that didn't match text properly

## Solution: `native-ui/src/ui/core.rs`

### 1. **Rect - Core Geometric Primitive**

```rust
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}
```

**Key Methods:**
- `position()`, `size()`, `center()` - Get derived values
- `right()`, `bottom()` - Get edge coordinates
- `contains_point(point)` - Hit testing
- `intersects(other)` - Collision detection
- `inset(padding)` - Create smaller rect with uniform padding
- `inset_by(left, top, right, bottom)` - Create smaller rect with different padding per side

**Usage:**
```rust
let input_rect = Rect::from_pos_size(position, size);
if input_rect.contains_point(mouse_pos) {
    // Handle click
}
```

### 2. **Layout Helpers**

Module: `core::layout`

**Functions:**
- `center_x(parent, child_width)` - Horizontal centering
- `center_y(parent, child_height)` - Vertical centering
- `center(parent, child_size)` - Both axes centering
- `align_right(parent, child_width, padding)` - Right alignment
- `align_bottom(parent, child_height, padding)` - Bottom alignment
- `stack_vertical(parent, heights, spacing, padding)` - Vertical layout
- `stack_horizontal(parent, widths, spacing, padding)` - Horizontal layout

**Example:**
```rust
use crate::ui::core::layout;

let parent = Rect::new(0.0, 0.0, 800.0, 600.0);
let button_width = 100.0;
let button_height = 40.0;

// Center button in parent
let button_pos = layout::center(&parent, Vec2::new(button_width, button_height));
```

### 3. **Text Positioning with Parley Integration**

Module: `core::text`

#### TextMetrics Structure
```rust
pub struct TextMetrics {
    pub width: f32,
    pub height: f32,
    pub baseline_offset: f32,  // Distance from top to baseline
}
```

**Two Ways to Create Metrics:**

1. **Approximation** (fast, for most UI):
```rust
let metrics = TextMetrics::approximate(font_size, char_count);
```

2. **Accurate** (using Parley):
```rust
let (width, height, baseline) = renderer.measure_text_accurate("Hello", 14.0);
let metrics = TextMetrics::from_parley(width, height, baseline);
```

#### Text Positioning Functions

**Quick Positioning** (uses approximation):
```rust
// Left-aligned text
let pos = text::left_aligned(&rect, font_size, padding_left);

// Center-aligned text
let pos = text::center_aligned(&rect, text_width, font_size);

// Right-aligned text
let pos = text::right_aligned(&rect, text_width, font_size, padding_right);
```

**Accurate Positioning** (uses Parley metrics):
```rust
let metrics = TextMetrics::from_parley(width, height, baseline);
let pos = text::center_aligned_accurate(&rect, &metrics);
```

**How It Works:**
1. Calculate vertical center of text bounding box in container
2. Add baseline offset to get actual baseline position
3. This ensures text is perfectly vertically centered

### 4. **Cursor Positioning**

Module: `core::cursor`

```rust
// Create cursor rect that aligns with text
let cursor_rect = cursor::rect_at_position(&input_rect, cursor_x, font_size);

// With accurate metrics
let cursor_rect = cursor::rect_at_position_with_metrics(&input_rect, cursor_x, &metrics);
```

Cursor automatically:
- Matches text height
- Aligns with text's bounding box
- Positions correctly relative to baseline

## Parley Integration

### What is Parley?
Parley is a high-quality text layout library that provides accurate font metrics including:
- Actual text width (not approximation)
- Line height
- Baseline position
- Proper glyph positioning

### How We Use It

**Already Available:**
- `renderer.compute_glyph_positions()` - Get x-position of each character
- `renderer.measure_text()` - Quick approximation
- `renderer.measure_text_accurate()` - Parley-based accurate measurement (NEW)

**When to Use Accurate Metrics:**
- Critical text alignment (headers, buttons)
- When text must be pixel-perfect
- Complex text layouts

**When Approximation is Fine:**
- Regular UI text
- Non-critical alignments
- Performance-sensitive rendering

## Migration Guide

### Before (Old Way):
```rust
// Hardcoded positioning
let text_pos = Vec2::new(
    button_pos.x + 10.0,
    button_pos.y + button_size.y / 2.0
);

// Manual cursor calculation
let cursor_y = input_y + 4.0;
let cursor_height = input_height - 8.0;
```

### After (New Way):
```rust
// Use Rect and layout helpers
let button_rect = Rect::new(button_pos.x, button_pos.y, button_width, button_height);
let text_pos = text::center_aligned(&button_rect, text_width, font_size);

// Use cursor helper
let cursor_rect = cursor::rect_at_position(&input_rect, cursor_x, font_size);
```

## Benefits

### 1. **Consistency**
- All text uses same baseline calculation
- All rectangles have same interface
- All layout uses same helpers

### 2. **Accuracy**
- Text positioning accounts for actual font metrics
- Cursor matches text exactly
- No magic numbers or approximations (when using accurate mode)

### 3. **Maintainability**
- One place to fix text positioning bugs
- Clear, documented functions
- Easy to add new layout patterns

### 4. **Type Safety**
- `Rect` encapsulates position + size
- `TextMetrics` encapsulates font measurements
- Compile-time guarantees

## Current Status

### ✅ Completed:
- Created `core.rs` with Rect, layout, text, and cursor modules
- Integrated Parley's accurate text metrics
- Updated chat window to use new system
- All functions tested and compiling

### ⏳ In Progress:
- Updating header to use core.rs
- Updating sidebar to use core.rs

### 📋 Next Steps:
1. Migrate all components to use `Rect` instead of separate `position` + `size`
2. Replace manual text positioning with `text::` helpers
3. Consider creating higher-level components (Button, Input) that use core.rs internally
4. Add more layout patterns as needed (grid, flex-like layouts)

## Usage Examples

### Button with Centered Text
```rust
use crate::ui::core::{Rect, text};
use crate::ui::style;

let button_rect = Rect::new(x, y, 100.0, 36.0);

// Background quad
let bg = Quad {
    position: button_rect.position(),
    size: button_rect.size(),
    color: style::button::PRIMARY,
    corner_radius: style::corner_radius::MEDIUM,
};

// Centered text
let text_width = renderer.measure_text("Click Me", 14.0).x;
let text_pos = text::center_aligned(&button_rect, text_width, 14.0);
renderer.queue_text("Click Me", text_pos, style::text::PRIMARY, 14.0);
```

### Input Field with Cursor
```rust
use crate::ui::core::{Rect, text, cursor};

let input_rect = Rect::from_pos_size(input.position, input.size);

// Text position (left-aligned with padding)
let text_pos = text::left_aligned(&input_rect, 14.0, 12.0);
renderer.queue_text(&input.text, text_pos, color, 14.0);

// Cursor at character position
let glyph_positions = renderer.compute_glyph_positions(&input.text, 14.0, text_pos.x);
let cursor_x = glyph_positions[input.cursor_position];
let cursor_rect = cursor::rect_at_position(&input_rect, cursor_x, 14.0);

// Render cursor
let cursor_quad = Quad {
    position: cursor_rect.position(),
    size: cursor_rect.size(),
    color: style::text::PRIMARY,
    corner_radius: 0.0,
};
```

### Stacked Layout
```rust
use crate::ui::core::layout;

let parent = Rect::new(0.0, 0.0, 300.0, 600.0);
let item_heights = vec![40.0, 40.0, 40.0, 40.0];
let spacing = 8.0;
let padding = 12.0;

let item_rects = layout::stack_vertical(&parent, &item_heights, spacing, padding);

for (i, rect) in item_rects.iter().enumerate() {
    // Render item background
    // Render item content at rect
}
```

## Performance Considerations

### Approximation vs. Accurate Metrics

**Approximation** (`baseline_y_approx`):
- Very fast (no Parley layout)
- Good enough for most UI (±1-2 pixels)
- Use for: Regular text, lists, non-critical alignment

**Accurate** (`baseline_y` with `from_parley`):
- Slower (creates Parley layout)
- Pixel-perfect positioning
- Use for: Buttons, headers, critical UI elements

**Recommendation:**
- Use approximation by default
- Only measure accurately for important elements
- Cache accurate measurements when possible

## API Reference

### Rect
```rust
Rect::new(x, y, width, height) -> Rect
Rect::from_pos_size(position: Vec2, size: Vec2) -> Rect
rect.position() -> Vec2
rect.size() -> Vec2
rect.center() -> Vec2
rect.right() -> f32
rect.bottom() -> f32
rect.contains_point(point: Vec2) -> bool
rect.intersects(other: &Rect) -> bool
rect.inset(padding: f32) -> Rect
rect.inset_by(left, top, right, bottom: f32) -> Rect
```

### layout
```rust
layout::center_x(parent: &Rect, child_width: f32) -> f32
layout::center_y(parent: &Rect, child_height: f32) -> f32
layout::center(parent: &Rect, child_size: Vec2) -> Vec2
layout::align_right(parent: &Rect, child_width: f32, padding: f32) -> f32
layout::align_bottom(parent: &Rect, child_height: f32, padding: f32) -> f32
layout::stack_vertical(parent: &Rect, heights: &[f32], spacing: f32, padding: f32) -> Vec<Rect>
layout::stack_horizontal(parent: &Rect, widths: &[f32], spacing: f32, padding: f32) -> Vec<Rect>
```

### text
```rust
TextMetrics::approximate(font_size: f32, char_count: usize) -> TextMetrics
TextMetrics::from_parley(width: f32, height: f32, baseline: f32) -> TextMetrics

text::baseline_y(rect: &Rect, metrics: &TextMetrics) -> f32
text::baseline_y_approx(rect: &Rect, font_size: f32) -> f32
text::left_aligned(rect: &Rect, font_size: f32, padding_left: f32) -> Vec2
text::center_aligned(rect: &Rect, text_width: f32, font_size: f32) -> Vec2
text::right_aligned(rect: &Rect, text_width: f32, font_size: f32, padding_right: f32) -> Vec2
text::left_aligned_accurate(rect: &Rect, padding_left: f32, metrics: &TextMetrics) -> Vec2
text::center_aligned_accurate(rect: &Rect, metrics: &TextMetrics) -> Vec2
text::right_aligned_accurate(rect: &Rect, padding_right: f32, metrics: &TextMetrics) -> Vec2
```

### cursor
```rust
cursor::rect_at_position(input_rect: &Rect, cursor_x: f32, font_size: f32) -> Rect
cursor::rect_at_position_with_metrics(input_rect: &Rect, cursor_x: f32, metrics: &TextMetrics) -> Rect
```

### renderer (NEW)
```rust
renderer.measure_text_accurate(text: &str, size: f32) -> (width: f32, height: f32, baseline: f32)
```

## Summary

The new `core.rs` system provides:
- **Standardized geometric primitives** (Rect)
- **Layout helpers** for common patterns
- **Text positioning** that accounts for baselines
- **Parley integration** for accurate metrics
- **Type-safe, modular, object-oriented** approach

This eliminates alignment bugs and makes UI code much more maintainable!

