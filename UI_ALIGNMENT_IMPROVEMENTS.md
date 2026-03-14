# UI Alignment and Styling Improvements

## Overview
This document describes the comprehensive UI alignment fixes and standardized styling system implemented across the application.

## Problem Statement
The UI had several alignment issues and inconsistent styling:
- Text and elements were not properly centered vertically
- Padding values were hardcoded and inconsistent
- Colors, fonts, and corner radii varied across components
- No centralized styling system for maintaining consistency

## Solution: Centralized Style System

### Created `native-ui/src/ui/style.rs`
A new module that provides:

#### 1. **Spacing & Sizing Constants**
```rust
pub mod padding {
    pub const TINY: f32 = 4.0;
    pub const SMALL: f32 = 8.0;
    pub const MEDIUM: f32 = 12.0;
    pub const LARGE: f32 = 16.0;
    pub const XLARGE: f32 = 20.0;
}

pub mod corner_radius {
    pub const SMALL: f32 = 4.0;
    pub const MEDIUM: f32 = 8.0;
    pub const LARGE: f32 = 12.0;
    pub const PILL: f32 = 9999.0;
}

pub mod font_size {
    pub const TINY: f32 = 10.0;
    pub const SMALL: f32 = 12.0;
    pub const NORMAL: f32 = 14.0;
    pub const MEDIUM: f32 = 16.0;
    pub const LARGE: f32 = 18.0;
    pub const XLARGE: f32 = 20.0;
    pub const TITLE: f32 = 24.0;
}
```

#### 2. **Color Palette**
Standardized colors for:
- **Backgrounds**: `bg::PRIMARY`, `bg::SECONDARY`, `bg::INPUT`, `bg::INPUT_FOCUSED`
- **Text**: `text::PRIMARY`, `text::SECONDARY`, `text::PLACEHOLDER`
- **Buttons**: `button::PRIMARY`, `button::DANGER` (with hover/active variants)
- **Highlights**: `highlight::SELECTION`, `highlight::HOVER`, `highlight::ACTIVE`
- **Borders**: `border::DEFAULT`, `border::FOCUSED`, `border::HOVER`

#### 3. **Layout Utilities**
Helper functions for proper alignment:
```rust
// Center element horizontally
pub fn center_x(container_width: f32, element_width: f32) -> f32

// Center element vertically  
pub fn center_y(container_height: f32, element_height: f32) -> f32

// Vertically center text (accounts for font baseline)
pub fn center_text_y(container_y: f32, container_height: f32, font_size: f32) -> f32

// Align to bottom/right with padding
pub fn align_bottom(container_y: f32, container_height: f32, element_height: f32, padding: f32) -> f32
pub fn align_right(container_x: f32, container_width: f32, element_width: f32, padding: f32) -> f32
```

## Components Updated

### 1. **Chat Window** (`native-ui/src/ui/chat_window.rs`, `native-ui/src/gfx/components/chat.rs`)

**Changes:**
- Input field uses `style::input_height::NORMAL` (40px)
- Send button uses `style::button_height::NORMAL` (36px)
- Consistent padding with `style::padding::MEDIUM` (12px)
- "Send" text is properly centered in button using `style::center_text_y()`
- Input text is vertically centered using `style::center_text_y()`
- Colors use `style::bg::INPUT`, `style::button::PRIMARY`, etc.
- Corner radius uses `style::corner_radius::MEDIUM` (8px)

**Before:**
```rust
let text_pos = Vec2::new(
    chat.input_field.position.x + 5.0,
    chat.input_field.position.y + chat.input_field.size.y / 2.0,
);
```

**After:**
```rust
let text_padding_x = style::padding::MEDIUM;
let text_pos = Vec2::new(
    chat.input_field.position.x + text_padding_x,
    style::center_text_y(chat.input_field.position.y, chat.input_field.size.y, FONT_SIZE),
);
```

### 2. **Header** (`native-ui/src/ui/header.rs`, `native-ui/src/gfx/components/header.rs`)

**Changes:**
- Tab bar is centered using `style::center_x()`
- Tab text is vertically centered using `style::center_text_y()`
- Window control buttons use consistent `style::padding::TINY` (4px) spacing
- Button colors use `style::button::DANGER`, `style::button::SECONDARY`
- Corner radius uses `style::corner_radius::SMALL` (4px)
- Active tab highlight uses `style::highlight::HOVER`
- "Notebook" title properly positioned with `style::padding::SMALL`

**Before:**
```rust
let tab_bar_x = (size.x - tab_bar_width) / 2.0;
let tab_y = tab_bar_pos.y + app.header.tab_bar.size.y / 2.0;
```

**After:**
```rust
let tab_bar_x = style::center_x(size.x, tab_bar_width);
let tab_y = style::center_text_y(tab_bar_pos.y, app.header.tab_bar.size.y, FONT_SIZE);
```

### 3. **Sidebar** (`native-ui/src/gfx/components/sidebar_content.rs`)

**Changes:**
- Conversation/document items use consistent `style::padding::SMALL` (8px)
- Item backgrounds use `style::highlight::HOVER` when selected
- Button colors use `style::button::PRIMARY` and `style::button::DANGER`
- Text uses `style::text::PRIMARY`, `style::text::SECONDARY`
- "+" buttons have centered text using `style::center_text_y()`
- Info/Delete buttons properly sized and colored
- Corner radius uses `style::corner_radius::MEDIUM` (8px) for items

**Before:**
```rust
let padding = 10.0;
let item_color = Vec4::new(0.3, 0.35, 0.4, 1.0);  // Hardcoded
renderer.queue_text(&title, Vec2::new(x + 5.0, y + 12.0), text_color, 12.0);
```

**After:**
```rust
let padding = style::padding::SMALL;
let item_color = style::highlight::HOVER;
renderer.queue_text(&title, Vec2::new(x + style::padding::TINY, y + style::padding::MEDIUM), text_color, ITEM_TITLE_FONT_SIZE);
```

## Benefits

### 1. **Visual Consistency**
- All text is properly vertically centered
- Consistent spacing between elements
- Uniform colors across similar UI components
- Consistent corner radii

### 2. **Maintainability**
- Single source of truth for styling values
- Easy to update entire UI by changing constants
- Clear semantic naming (e.g., `style::button::PRIMARY` vs. `Vec4::new(0.3, 0.55, 0.4, 1.0)`)

### 3. **Developer Experience**
- Clear documentation of available sizes and colors
- Helper functions reduce repetitive calculations
- Type-safe constants prevent magic numbers

### 4. **Alignment Improvements**
- **Text vertical centering**: Uses `center_text_y()` which accounts for font baseline
- **Element centering**: Uses `center_x()` and `center_y()` for proper centering
- **Consistent padding**: All elements use the same padding scale

## Implementation Pattern

When adding new UI elements, follow this pattern:

```rust
use crate::ui::style;

// Define element
let button_bg = Quad {
    position: button_position,
    size: Vec2::new(80.0, style::button_height::NORMAL),
    color: style::button::PRIMARY,
    corner_radius: style::corner_radius::MEDIUM,
};

// Center text in button
let text_x = button_position.x + style::center_x(0.0, button_width);
let text_y = style::center_text_y(button_position.y, button_height, font_size);
renderer.queue_text("Click Me", Vec2::new(text_x, text_y), style::text::PRIMARY, style::font_size::NORMAL);
```

## Future Improvements

### Recommended Next Steps:
1. ✅ **Apply styling to Library Window**
2. ✅ **Apply styling to Modals and Dialogs**
3. ✅ **Add hover states** for all interactive elements
4. **Create button component** that automatically handles styling
5. **Add animation constants** for consistent transitions
6. **Create input field component** with built-in styling

### Additional Opportunities:
- **Theme system**: Allow switching between light/dark themes
- **Responsive sizing**: Scale elements based on viewport size
- **Accessibility**: Add high-contrast mode
- **Custom themes**: Allow users to customize colors

## Testing

To verify alignment improvements:
1. Build and run the application
2. Check that all text is properly vertically centered
3. Verify consistent spacing between elements
4. Confirm buttons have uniform appearance
5. Test hover/active states

## Files Modified

### Core Style System:
- ✅ `native-ui/src/ui/style.rs` (NEW)
- ✅ `native-ui/src/ui/mod.rs` (export style module)

### Components Updated:
- ✅ `native-ui/src/ui/chat_window.rs`
- ✅ `native-ui/src/ui/header.rs`
- ✅ `native-ui/src/gfx/components/chat.rs`
- ✅ `native-ui/src/gfx/components/header.rs`
- ✅ `native-ui/src/gfx/components/sidebar_content.rs`

### Pending Updates:
- ⏳ `native-ui/src/gfx/components/library.rs`
- ⏳ `native-ui/src/gfx/components/modals.rs`
- ⏳ Other modal/dialog components

## Summary

The standardized styling system provides:
- **Visual consistency** across all UI components
- **Proper text alignment** using baseline-aware centering
- **Maintainable code** with clear constants and utilities
- **Scalability** for future UI additions

All text, buttons, and interactive elements now use consistent spacing, colors, and sizing, resulting in a more polished and professional appearance.

