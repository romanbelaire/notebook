/// Centralized UI styling constants and utilities
/// This module provides consistent spacing, sizing, colors, and other style properties across the application.
use glam::Vec4;

// ===== SPACING & SIZING =====

/// Standard padding for UI elements
pub mod padding {
    pub const TINY: f32 = 4.0;
    pub const SMALL: f32 = 8.0;
    pub const MEDIUM: f32 = 12.0;
    pub const LARGE: f32 = 16.0;
    pub const XLARGE: f32 = 20.0;
}

/// Standard corner radius values
pub mod corner_radius {
    pub const SMALL: f32 = 4.0;
    pub const MEDIUM: f32 = 8.0;
    pub const LARGE: f32 = 12.0;
    pub const PILL: f32 = 9999.0;
}

/// Standard font sizes
pub mod font_size {
    pub const TINY: f32 = 10.0;
    pub const SMALL: f32 = 12.0;
    pub const NORMAL: f32 = 14.0;
    pub const MEDIUM: f32 = 16.0;
    pub const LARGE: f32 = 18.0;
    pub const XLARGE: f32 = 20.0;
    pub const TITLE: f32 = 24.0;
}

/// Standard button heights
pub mod button_height {
    pub const SMALL: f32 = 28.0;
    pub const NORMAL: f32 = 36.0;
    pub const LARGE: f32 = 44.0;
}

/// Standard input field heights
pub mod input_height {
    pub const SMALL: f32 = 32.0;
    pub const NORMAL: f32 = 40.0;
    pub const LARGE: f32 = 48.0;
}

// ===== COLORS =====

/// Background colors
pub mod bg {
    use super::Vec4;
    
    // Ultra-dark space palette: 0B0C0D base with subtle layering
    pub const PRIMARY: Vec4 = Vec4::new(0.043, 0.047, 0.051, 1.0);       // #0b0c0d
    pub const SECONDARY: Vec4 = Vec4::new(0.066, 0.071, 0.078, 1.0);     // #111316
    pub const TERTIARY: Vec4 = Vec4::new(0.082, 0.089, 0.098, 1.0);      // #15181a
    pub const INPUT: Vec4 = Vec4::new(0.082, 0.089, 0.098, 1.0);         // same as tertiary
    pub const INPUT_FOCUSED: Vec4 = Vec4::new(0.098, 0.107, 0.118, 1.0); // #191c1e
    
    // Chat message bubbles
    pub const USER_MESSAGE: Vec4 = Vec4::new(0.16, 0.18, 0.20, 1.0);       // slightly lifted dark for user
    pub const ASSISTANT_MESSAGE: Vec4 = Vec4::new(0.13, 0.14, 0.16, 1.0);  // near-base dark for assistant
    pub const MUTED_MESSAGE: Vec4 = Vec4::new(0.10, 0.11, 0.12, 0.7);      // darker, semi-transparent
    /// Transparent backplate for shard (message pair) nodes in constellation view
    pub const SHARD_BACKPLATE: Vec4 = Vec4::new(0.10, 0.11, 0.12, 0.8);
}

/// Text colors
pub mod text {
    use super::Vec4;
    
    // Slightly beige text to reduce eye strain on very dark backgrounds
    pub const PRIMARY: Vec4 = Vec4::new(0.909, 0.867, 0.769, 1.0);   // #e8dcc4
    pub const SECONDARY: Vec4 = Vec4::new(0.780, 0.733, 0.635, 1.0); // #c7bbA2-ish
    pub const TERTIARY: Vec4 = Vec4::new(0.600, 0.563, 0.482, 1.0);  // deeper beige
    pub const PLACEHOLDER: Vec4 = Vec4::new(0.600, 0.563, 0.482, 0.7);
    pub const ACCENT: Vec4 = Vec4::new(0.941, 0.890, 0.753, 1.0);    // slightly brighter beige accent
}

/// Button colors
pub mod button {
    use super::Vec4;
    
    // Transparent-ish pill buttons with white outlines; fill darkens slightly on hover/active
    pub const PRIMARY: Vec4 = Vec4::new(0.043, 0.047, 0.051, 1.0);
    pub const PRIMARY_HOVER: Vec4 = Vec4::new(0.066, 0.071, 0.078, 1.0);
    pub const PRIMARY_ACTIVE: Vec4 = Vec4::new(0.035, 0.039, 0.043, 1.0);
    
    pub const SECONDARY: Vec4 = Vec4::new(0.066, 0.071, 0.078, 1.0);
    pub const SECONDARY_HOVER: Vec4 = Vec4::new(0.082, 0.089, 0.098, 1.0);
    pub const SECONDARY_ACTIVE: Vec4 = Vec4::new(0.051, 0.055, 0.059, 1.0);
    
    pub const DANGER: Vec4 = Vec4::new(0.7, 0.2, 0.2, 1.0);
    pub const DANGER_HOVER: Vec4 = Vec4::new(0.8, 0.25, 0.25, 1.0);
    pub const DANGER_ACTIVE: Vec4 = Vec4::new(0.6, 0.15, 0.15, 1.0);
}

/// Border colors
pub mod border {
    use super::Vec4;
    
    // Thin white outlines in different opacities
    pub const DEFAULT: Vec4 = Vec4::new(1.0, 1.0, 1.0, 0.18);
    pub const FOCUSED: Vec4 = Vec4::new(1.0, 1.0, 1.0, 0.36);
    pub const HOVER: Vec4 = Vec4::new(1.0, 1.0, 1.0, 0.26);
}

/// Selection and highlight colors
pub mod highlight {
    use super::Vec4;
    
    pub const SELECTION: Vec4 = Vec4::new(1.0, 1.0, 1.0, 0.16);
    pub const HOVER: Vec4 = Vec4::new(0.082, 0.089, 0.098, 1.0);
    pub const ACTIVE: Vec4 = Vec4::new(0.098, 0.107, 0.118, 1.0);
}

// ===== LAYOUT UTILITIES =====

/// Calculate centered position
pub fn center_x(container_width: f32, element_width: f32) -> f32 {
    (container_width - element_width) / 2.0
}

/// Calculate centered position
pub fn center_y(container_height: f32, element_height: f32) -> f32 {
    (container_height - element_height) / 2.0
}

/// Vertically center text within a container
/// Text rendering uses baseline positioning, so we need to account for ascent
pub fn center_text_y(container_y: f32, container_height: f32, font_size: f32) -> f32 {
    // Center of container minus half the font size to get proper baseline position
    container_y + (container_height - font_size) / 2.0 + font_size * 0.75
}

/// Align element to bottom of container with padding
pub fn align_bottom(container_y: f32, container_height: f32, element_height: f32, padding: f32) -> f32 {
    container_y + container_height - element_height - padding
}

/// Align element to right of container with padding
pub fn align_right(container_x: f32, container_width: f32, element_width: f32, padding: f32) -> f32 {
    container_x + container_width - element_width - padding
}

