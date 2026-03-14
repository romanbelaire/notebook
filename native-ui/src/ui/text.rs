use glam::{Vec2, Vec4};
use crate::ui::core::{Rect, text};
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;
use crate::ui::components::Renderable;
use crate::ui::style;

/// A text label component that can be used in the layout system
/// 
/// Text components must be children of container components (VStack, HStack, etc.)
/// and cannot be orphaned floating text. Text can only be created through container
/// component methods to ensure it's always part of the component hierarchy.
pub struct Text {
    text: String,
    rect: Rect,
    font_size: f32,
    color: Vec4,
    alignment: TextAlignment,
    scissor_rect: Option<Rect>,  // Optional scissor rect for clipping
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
}

/// Private token that only container components can create
/// This ensures Text can only be instantiated through container methods
mod private {
    /// Token that proves the caller is a container component
    /// Only VStack and HStack can create this token
    pub struct TextCreationToken(());
    
    impl TextCreationToken {
        /// Create a new token - only callable from container components
        pub fn new() -> Self {
            Self(())
        }
    }
}

/// Public token type for creating Text components
/// Only container components (VStack, HStack) can create this
pub struct TextCreationToken(private::TextCreationToken);

impl TextCreationToken {
    /// Create a new token - only callable from container components
    /// This is pub(crate) so only VStack/HStack in the same crate can call it
    pub(crate) fn new() -> Self {
        Self(private::TextCreationToken::new())
    }
}

impl Text {
    /// Create a new Text component from a container (VStack, HStack).
    /// 
    /// This method requires a `TextCreationToken` which can only be obtained
    /// from container components. This ensures Text components are always part
    /// of a container hierarchy when created through the normal API.
    /// 
    /// External code should use container methods like `VStack::add_text()` instead.
    pub(crate) fn new_internal(text: impl Into<String>, _token: TextCreationToken) -> Self {
        Self {
            text: text.into(),
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            font_size: style::font_size::NORMAL,
            color: style::text::PRIMARY,
            alignment: TextAlignment::Left,
            scissor_rect: None,
        }
    }
    
    /// Create a new Text component for direct rendering (legacy support).
    /// 
    /// # Warning
    /// This method should only be used in render functions for standalone text
    /// that cannot be part of a VStack/HStack (e.g., button labels, window titles).
    /// All such Text components must have proper parent tracking via `renderer.push_parent()`.
    /// 
    /// Prefer using `VStack::add_text()` or `HStack::add_text()` when possible.
    /// 
    /// # Deprecated
    /// This method is deprecated. Use VStack/HStack::add_text() or create Text through
    /// a container component. This method is kept for backward compatibility during migration.
    #[deprecated(note = "Use VStack/HStack::add_text() or create Text through container components")]
    pub(crate) fn new_for_render(text: impl Into<String>) -> Self {
        // Bypasses token requirement for render functions
        // TODO: Remove once all render functions use component tree
        Self {
            text: text.into(),
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            font_size: style::font_size::NORMAL,
            color: style::text::PRIMARY,
            alignment: TextAlignment::Left,
            scissor_rect: None,
        }
    }
    
    /// Get the text content
    pub fn text(&self) -> &str {
        &self.text
    }
    
    /// Set the text content
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }
    
    /// Set font size (builder method)
    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.font_size = font_size;
        self
    }
    
    /// Set color (builder method)
    pub fn with_color(mut self, color: Vec4) -> Self {
        self.color = color;
        self
    }
    
    /// Set alignment (builder method)
    pub fn with_alignment(mut self, alignment: TextAlignment) -> Self {
        self.alignment = alignment;
        self
    }
    
    /// Set scissor rect (builder method)
    pub fn with_scissor(mut self, scissor_rect: Option<Rect>) -> Self {
        self.scissor_rect = scissor_rect;
        self
    }
    
    /// Get font size
    pub fn font_size(&self) -> f32 {
        self.font_size
    }
    
    /// Get color
    pub fn color(&self) -> Vec4 {
        self.color
    }
    
    /// Get alignment
    pub fn alignment(&self) -> TextAlignment {
        self.alignment
    }
    
    /// Compute wrapped text size with word wrapping
    /// `measure_fn` should be a function that measures text width: `fn(&str, f32) -> Vec2`
    pub fn wrapped_size_with_measure(&self, max_width: f32, mut measure_fn: impl FnMut(&str, f32) -> Vec2) -> Vec2 {
        let line_height = self.font_size * 1.2;
        let words: Vec<&str> = self.text.split_whitespace().collect();
        let mut current_line = String::new();
        let mut lines = Vec::new();
        let mut max_line_width: f32 = 0.0;
        
        for word in words {
            let test_line = if current_line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", current_line, word)
            };
            
            let test_size = measure_fn(&test_line, self.font_size);
            
            if test_size.x > max_width && !current_line.is_empty() {
                // Current line is full, save it and start new one
                let line_size = measure_fn(&current_line, self.font_size);
                max_line_width = max_line_width.max(line_size.x);
                lines.push(current_line);
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }
        
        // Add the last line
        if !current_line.is_empty() {
            let line_size = measure_fn(&current_line, self.font_size);
            max_line_width = max_line_width.max(line_size.x);
            lines.push(current_line);
        }
        
        let height = if lines.is_empty() {
            line_height
        } else {
            lines.len() as f32 * line_height
        };
        
        Vec2::new(max_line_width, height)
    }
}

use crate::app::App;

impl Renderable for Text {
    fn wrapped_size(&self, max_width: Option<f32>, measure_fn: Option<&mut dyn FnMut(&str, f32) -> Vec2>) -> Vec2 {
        if let (Some(max_w), Some(measure)) = (max_width, measure_fn) {
            // Use word wrapping - call the helper method
            // Create a wrapper closure that calls the mutable closure
            let measure_wrapper = |text: &str, font_size: f32| -> Vec2 {
                measure(text, font_size)
            };
            self.wrapped_size_with_measure(max_w, measure_wrapper)
        } else {
            // No wrapping, use min_size
            self.min_size()
        }
    }
    
    fn render(&self, renderer: &mut Renderer, _app: &App, _vertices: &mut Vec<Vertex>, _dirty_rect: Option<Rect>) {
        // Text components should be validated by their parent container
        // We use a content-based ID instead of pointer-based to avoid duplicates
        // when the same Text content is rendered in different contexts
        let text_hash = format!("{:x}", self.text.chars().take(50).map(|c| c as u32).sum::<u32>());
        let component_id = format!("text_{}_{}_{}", text_hash, self.rect.x as i32, self.rect.y as i32);
        
        // Check if this component should be skipped (orphaned or duplicate)
        if renderer.should_skip_component(&component_id) {
            return; // Skip rendering
        }
        
        // Validate and check if rendering should proceed
        // Parent is determined from the current parent stack
        if !renderer.validate_component(&component_id, None, "Text") {
            return; // Skip rendering (orphaned or duplicate)
        }
        
        let text_pos = match self.alignment {
            TextAlignment::Left => {
                text::left_aligned(&self.rect, self.font_size, 0.0)
            }
            TextAlignment::Center => {
                let text_width = renderer.measure_text(&self.text, self.font_size).x;
                text::center_aligned(&self.rect, text_width, self.font_size)
            }
            TextAlignment::Right => {
                let text_width = renderer.measure_text(&self.text, self.font_size).x;
                text::right_aligned(&self.rect, text_width, self.font_size, 0.0)
            }
        };
        
        // Use scissor-aware text rendering if scissor rect is provided
        if let Some(ref scissor) = self.scissor_rect {
            renderer.queue_text_with_ui_scissor(&self.text, text_pos, self.color, self.font_size, Some(scissor));
        } else {
            renderer.queue_text(&self.text, text_pos, self.color, self.font_size);
        }
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        self.rect = available_rect;
    }
    
    fn min_size(&self) -> Vec2 {
        // Estimate text size - actual measurement happens during render
        // Use a standard line height for text components (25.0) to match layout expectations
        // Width is estimated based on character count
        let char_count = self.text.chars().count();
        let estimated_width = char_count as f32 * self.font_size * 0.6; // Approximate char width
        // Use 25.0 as standard text line height to match previous layout system
        Vec2::new(estimated_width, 25.0)
    }
}

