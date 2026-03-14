/// Core UI primitives and layout system
/// This module provides the foundation for all UI components with standardized properties
use glam::Vec2;

/// Rectangle representing position and size
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    pub fn from_pos_size(position: Vec2, size: Vec2) -> Self {
        Self {
            x: position.x,
            y: position.y,
            width: size.x,
            height: size.y,
        }
    }

    pub fn position(&self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    pub fn size(&self) -> Vec2 {
        Vec2::new(self.width, self.height)
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.x 
            && point.x <= self.right() 
            && point.y >= self.y 
            && point.y <= self.bottom()
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.right() 
            && self.right() > other.x 
            && self.y < other.bottom() 
            && self.bottom() > other.y
    }

    /// Check if this rect is visible within the viewport (for frustum culling)
    /// Returns true if the rect intersects with the viewport, false if completely off-screen
    pub fn is_visible(&self, viewport: &Rect) -> bool {
        self.intersects(viewport)
    }

    /// Create a rect inset by padding on all sides
    pub fn inset(&self, padding: f32) -> Rect {
        Rect {
            x: self.x + padding,
            y: self.y + padding,
            width: (self.width - padding * 2.0).max(0.0),
            height: (self.height - padding * 2.0).max(0.0),
        }
    }

    /// Create a rect inset by different padding on each side
    pub fn inset_by(&self, left: f32, top: f32, right: f32, bottom: f32) -> Rect {
        Rect {
            x: self.x + left,
            y: self.y + top,
            width: (self.width - left - right).max(0.0),
            height: (self.height - top - bottom).max(0.0),
        }
    }
}

/// Alignment options for positioning elements
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alignment {
    Start,
    Center,
    End,
}

/// Layout direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Horizontal,
    Vertical,
}

/// Base properties that all UI elements should have
pub trait UIElement {
    /// Get the bounding rectangle of this element
    fn bounds(&self) -> Rect;

    /// Check if a point is inside this element
    fn contains(&self, point: Vec2) -> bool {
        self.bounds().contains_point(point)
    }

    /// Get the position of this element
    fn position(&self) -> Vec2 {
        self.bounds().position()
    }

    /// Get the size of this element
    fn size(&self) -> Vec2 {
        self.bounds().size()
    }

    /// Update layout (called when parent changes)
    fn update_layout(&mut self, parent_bounds: Rect);
}

/// Helper functions for calculating positions
pub mod layout {
    use super::*;

    /// Center a child element within a parent horizontally
    pub fn center_x(parent: &Rect, child_width: f32) -> f32 {
        parent.x + (parent.width - child_width) / 2.0
    }

    /// Center a child element within a parent vertically
    pub fn center_y(parent: &Rect, child_height: f32) -> f32 {
        parent.y + (parent.height - child_height) / 2.0
    }

    /// Center a child element within a parent
    pub fn center(parent: &Rect, child_size: Vec2) -> Vec2 {
        Vec2::new(
            center_x(parent, child_size.x),
            center_y(parent, child_size.y),
        )
    }

    /// Align element to the right of parent with padding
    pub fn align_right(parent: &Rect, child_width: f32, padding: f32) -> f32 {
        parent.right() - child_width - padding
    }

    /// Align element to the bottom of parent with padding
    pub fn align_bottom(parent: &Rect, child_height: f32, padding: f32) -> f32 {
        parent.bottom() - child_height - padding
    }

    /// Create a rect positioned at the top of parent with padding
    pub fn align_top(parent: &Rect, child_width: f32, child_height: f32, padding: f32) -> Rect {
        Rect::new(
            parent.x + padding,
            parent.y + padding,
            child_width,
            child_height,
        )
    }

    /// Stack elements vertically with spacing
    pub fn stack_vertical(parent: &Rect, child_heights: &[f32], spacing: f32, padding: f32) -> Vec<Rect> {
        let mut rects = Vec::new();
        let mut current_y = parent.y + padding;

        for &height in child_heights {
            rects.push(Rect::new(
                parent.x + padding,
                current_y,
                parent.width - padding * 2.0,
                height,
            ));
            current_y += height + spacing;
        }

        rects
    }

    /// Stack elements horizontally with spacing
    pub fn stack_horizontal(parent: &Rect, child_widths: &[f32], spacing: f32, padding: f32) -> Vec<Rect> {
        let mut rects = Vec::new();
        let mut current_x = parent.x + padding;

        for &width in child_widths {
            rects.push(Rect::new(
                current_x,
                parent.y + padding,
                width,
                parent.height - padding * 2.0,
            ));
            current_x += width + spacing;
        }

        rects
    }
}

/// Text rendering helpers that account for baseline
pub mod text {
    use super::*;

    /// Text metrics for accurate positioning
    #[derive(Debug, Clone, Copy)]
    pub struct TextMetrics {
        pub width: f32,
        pub height: f32,
        /// Distance from top of bounding box to baseline
        pub baseline_offset: f32,
    }

    impl TextMetrics {
        /// Create metrics using approximation (when accurate measurement not available)
        pub fn approximate(font_size: f32, char_count: usize) -> Self {
            Self {
                // Treat glyphs as ~10% wider for layout to ease reading
                width: char_count as f32 * font_size * 0.66,
                height: font_size * 1.2,
                baseline_offset: font_size * 0.75,
            }
        }

        /// Create metrics from Parley measurement
        /// Use renderer.measure_text_accurate() to get (width, height, baseline)
        pub fn from_parley(width: f32, height: f32, baseline: f32) -> Self {
            Self {
                width,
                height,
                baseline_offset: baseline,
            }
        }
    }

    /// Calculate Y position for text rendering to vertically center text in a rect
    /// Returns the Y coordinate for the TOP of the text line (not the baseline!)
    /// Vello/Parley will internally add the baseline offset when rendering
    pub fn top_y(rect: &Rect, metrics: &TextMetrics) -> f32 {
        // Center the text's bounding box in the container
        rect.y + (rect.height - metrics.height) / 2.0
    }

    /// Calculate Y position using font size approximation (for quick positioning)
    pub fn top_y_approx(rect: &Rect, font_size: f32) -> f32 {
        top_y(rect, &TextMetrics::approximate(font_size, 0))
    }
    
    /// Calculate actual baseline Y position (for aligning other elements like cursors)
    pub fn baseline_y(rect: &Rect, metrics: &TextMetrics) -> f32 {
        top_y(rect, metrics) + metrics.baseline_offset
    }

    /// Calculate position for left-aligned text vertically centered in rect
    /// Returns position for TOP of text line (Vello/Parley adds baseline internally)
    pub fn left_aligned(rect: &Rect, font_size: f32, padding_left: f32) -> Vec2 {
        Vec2::new(
            rect.x + padding_left,
            top_y_approx(rect, font_size),
        )
    }

    /// Calculate position for left-aligned text with accurate metrics
    pub fn left_aligned_accurate(rect: &Rect, padding_left: f32, metrics: &TextMetrics) -> Vec2 {
        Vec2::new(
            rect.x + padding_left,
            top_y(rect, metrics),
        )
    }

    /// Calculate position for center-aligned text vertically centered in rect
    pub fn center_aligned(rect: &Rect, text_width: f32, font_size: f32) -> Vec2 {
        Vec2::new(
            rect.x + (rect.width - text_width) / 2.0,
            top_y_approx(rect, font_size),
        )
    }

    /// Calculate position for center-aligned text with accurate metrics
    pub fn center_aligned_accurate(rect: &Rect, metrics: &TextMetrics) -> Vec2 {
        Vec2::new(
            rect.x + (rect.width - metrics.width) / 2.0,
            top_y(rect, metrics),
        )
    }

    /// Calculate position for right-aligned text vertically centered in rect
    pub fn right_aligned(rect: &Rect, text_width: f32, font_size: f32, padding_right: f32) -> Vec2 {
        Vec2::new(
            rect.right() - text_width - padding_right,
            top_y_approx(rect, font_size),
        )
    }

    /// Calculate position for right-aligned text with accurate metrics
    pub fn right_aligned_accurate(rect: &Rect, padding_right: f32, metrics: &TextMetrics) -> Vec2 {
        Vec2::new(
            rect.right() - metrics.width - padding_right,
            top_y(rect, metrics),
        )
    }
}

/// Container and section system for organizing UI elements
pub mod container {
    use super::*;

    /// A section within a container (e.g., "Conversations", "Documents")
    #[derive(Debug, Clone)]
    pub struct Section {
        /// Section title (e.g., "Conversations")
        pub title: String,
        /// Height of the title area
        pub title_height: f32,
        /// Items in this section
        pub item_count: usize,
        /// Height of each item
        pub item_height: f32,
        /// Whether the section content is scrollable
        pub scrollable: bool,
        /// Current scroll offset (only used if scrollable)
        pub scroll_offset: f32,
        /// Maximum height for scrollable content (None = use available space)
        pub max_content_height: Option<f32>,
    }

    impl Section {
        pub fn new(title: String, item_height: f32) -> Self {
            Self {
                title,
                title_height: 30.0,
                item_count: 0,
                item_height,
                scrollable: true,
                scroll_offset: 0.0,
                max_content_height: None,
            }
        }

        /// Calculate the total height of this section (title + content)
        pub fn total_height(&self) -> f32 {
            let content_height = self.item_count as f32 * self.item_height;
            let visible_content_height = if let Some(max_h) = self.max_content_height {
                content_height.min(max_h)
            } else {
                content_height
            };
            self.title_height + visible_content_height
        }

        /// Get the rect for the title area
        pub fn title_rect(&self, container_rect: &Rect, y_offset: f32) -> Rect {
            Rect::new(
                container_rect.x,
                container_rect.y + y_offset,
                container_rect.width,
                self.title_height,
            )
        }

        /// Get the rect for the scrollable content area
        pub fn content_rect(&self, container_rect: &Rect, y_offset: f32) -> Rect {
            let content_height = self.item_count as f32 * self.item_height;
            let visible_height = if let Some(max_h) = self.max_content_height {
                content_height.min(max_h)
            } else {
                content_height
            };
            
            Rect::new(
                container_rect.x,
                container_rect.y + y_offset + self.title_height,
                container_rect.width,
                visible_height,
            )
        }

        /// Get rect for a specific item index (accounting for scroll)
        /// Returns the rect even if partially offscreen - scissor rect will handle clipping
        pub fn item_rect(&self, container_rect: &Rect, y_offset: f32, item_index: usize, padding: f32) -> Option<Rect> {
            if item_index >= self.item_count {
                return None;
            }

            let content_rect = self.content_rect(container_rect, y_offset);
            let item_y = content_rect.y + (item_index as f32 * self.item_height) - self.scroll_offset;

            // Cull items that are completely outside the visible area
            // The GPU scissor will handle pixel-perfect clipping for items near the edges
            if item_y + self.item_height < content_rect.y || item_y > content_rect.bottom() {
                return None;
            }

            Some(Rect::new(
                content_rect.x + padding,
                item_y,
                content_rect.width - padding * 2.0,
                self.item_height,
            ))
        }
    }

    /// A vertical stack of sections with proper spacing
    #[derive(Debug, Clone)]
    pub struct SectionStack {
        pub sections: Vec<Section>,
        pub spacing: f32,
    }

    impl SectionStack {
        pub fn new(spacing: f32) -> Self {
            Self {
                sections: Vec::new(),
                spacing,
            }
        }

        pub fn add_section(&mut self, section: Section) {
            self.sections.push(section);
        }

        /// Calculate layout for all sections within a container
        /// Returns vec of (section_index, y_offset) for rendering
        pub fn layout(&self, container_rect: &Rect) -> Vec<(usize, f32)> {
            let mut result = Vec::new();
            let mut current_y = 0.0;

            for (i, section) in self.sections.iter().enumerate() {
                result.push((i, current_y));
                current_y += section.total_height() + self.spacing;
            }

            result
        }

        /// Get total height of all sections
        pub fn total_height(&self) -> f32 {
            let mut height = 0.0;
            for section in &self.sections {
                height += section.total_height() + self.spacing;
            }
            height.max(0.0)
        }

        /// Find which section and item (if any) contains a point
        pub fn hit_test(&self, container_rect: &Rect, point: Vec2, padding: f32) -> Option<SectionHit> {
            let layout = self.layout(container_rect);

            for (section_idx, y_offset) in layout {
                let section = &self.sections[section_idx];
                
                // Check title hit
                let title_rect = section.title_rect(container_rect, y_offset);
                if title_rect.contains_point(point) {
                    return Some(SectionHit::Title(section_idx));
                }

                // Check content hit
                let content_rect = section.content_rect(container_rect, y_offset);
                if content_rect.contains_point(point) {
                    // Find which item
                    for item_idx in 0..section.item_count {
                        if let Some(item_rect) = section.item_rect(container_rect, y_offset, item_idx, padding) {
                            if item_rect.contains_point(point) {
                                return Some(SectionHit::Item(section_idx, item_idx));
                            }
                        }
                    }
                    return Some(SectionHit::Content(section_idx));
                }
            }

            None
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum SectionHit {
        /// Hit the title of a section
        Title(usize),
        /// Hit an item within a section
        Item(usize, usize),
        /// Hit the content area but not a specific item
        Content(usize),
    }
}

/// Cursor rendering helpers
pub mod cursor {
    use super::*;

    /// Calculate cursor rect for a text input at a given character position
    /// Uses text metrics for accurate alignment with text
    /// Cursor aligns with the text's baseline and height
    pub fn rect_at_position_with_metrics(
        input_rect: &Rect,
        cursor_x: f32,
        metrics: &text::TextMetrics,
    ) -> Rect {
        // Cursor height should be smaller than the full line height
        // Use the font size (not line height) for a more natural cursor
        let cursor_height = metrics.height * 0.85;  // Slightly shorter than full height
        
        // Position cursor to align with text, starting just above the baseline
        let text_baseline = text::baseline_y(input_rect, metrics);
        let cursor_y = text_baseline - metrics.baseline_offset * 0.85;
        
        Rect::new(
            cursor_x,
            cursor_y,
            1.0,  // Half width for thinner cursor
            cursor_height,
        )
    }

    /// Calculate cursor rect using font size approximation
    pub fn rect_at_position(
        input_rect: &Rect,
        cursor_x: f32,
        font_size: f32,
    ) -> Rect {
        let metrics = text::TextMetrics::approximate(font_size, 0);
        rect_at_position_with_metrics(input_rect, cursor_x, &metrics)
    }
}

/// Standard text input rendering
pub mod text_input_render {
    use super::*;
    use crate::gfx::types::{Vertex, Quad};
    use crate::gfx::renderer::Renderer;
    use crate::ui::TextInput;
    use crate::ui::style;
    use crate::app::App;
    use crate::ui::components::Renderable;

    /// Render a standard text input field with proper cursor and selection.
    /// When wrap_text is true, text is word-wrapped to the input width (e.g. for shard modal).
    pub fn render_text_input(
        renderer: &mut Renderer,
        input: &TextInput,
        app: &App,
        vertices: &mut Vec<Vertex>,
        font_size: Option<f32>,
        padding: Option<f32>,
        corner_radius: Option<f32>,
        wrap_text: bool,
    ) {
        const DEFAULT_FONT_SIZE: f32 = style::font_size::NORMAL;
        const DEFAULT_PADDING: f32 = style::padding::SMALL;
        const DEFAULT_CORNER_RADIUS: f32 = style::corner_radius::SMALL;

        let font_size = font_size.unwrap_or(DEFAULT_FONT_SIZE);
        let text_padding = padding.unwrap_or(DEFAULT_PADDING);
        let corner_radius = corner_radius.unwrap_or(DEFAULT_CORNER_RADIUS);

        // Create rect for input field
        let input_rect = Rect::from_pos_size(input.position, input.size);

        // Input field background
        let input_bg = Quad {
            position: input_rect.position(),
            size: input_rect.size(),
            color: if input.focused {
                style::bg::INPUT_FOCUSED
            } else {
                style::bg::INPUT
            },
            corner_radius,
            bubble_effect: false,
                slider_effect: false,
        };
        vertices.extend_from_slice(&input_bg.to_vertices());

        // Render input text or placeholder
        let text_buf = input.text.clone();
        let text_is_empty = text_buf.is_empty();
        let text_to_show = if text_is_empty {
            input.placeholder.clone()
        } else {
            text_buf.clone()
        };
        let text_color = if text_is_empty {
            style::text::PLACEHOLDER
        } else {
            style::text::PRIMARY
        };

        let text_rect = Rect::new(
            input_rect.x + text_padding,
            input_rect.y,
            input_rect.width - text_padding * 2.0,
            input_rect.height,
        );

        if wrap_text {
            let content_id = format!("text_input_content_{:p}", input);
            renderer.push_parent(content_id.clone());
            renderer.validate_component(&content_id, None, "TextInputContent");
            renderer.push_scissor(&text_rect);
            renderer.queue_plain_text_wrapped(&text_to_show, text_rect.position(), text_color, font_size, text_rect.width);
            renderer.pop_scissor();
            renderer.pop_parent();
        } else {
            let mut text_component = crate::ui::text::Text::new_for_render(&text_to_show)
                .with_font_size(font_size)
                .with_color(text_color)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            text_component.update_layout(text_rect, None, None);
            let content_id = format!("text_input_content_{:p}", input);
            renderer.push_parent(content_id.clone());
            renderer.validate_component(&content_id, None, "TextInputContent");
            text_component.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
        
        // Compute glyph positions for cursor/selection (still needed for cursor positioning)
        let text_pos = text::left_aligned(&input_rect, font_size, text_padding);
        let positions = if !input.glyph_positions.is_empty() && 
                         input.text == text_to_show {
            input.glyph_positions.clone()
        } else {
            renderer.compute_glyph_positions(&text_to_show, font_size, text_pos.x)
        };

        // Render selection highlight if we have an active selection
        if let (Some(sel_start), Some(sel_end)) = (input.selection_start, input.selection_end) {
            let sel_start_pos = if sel_start < positions.len() {
                positions[sel_start]
            } else {
                positions.last().copied().unwrap_or(text_pos.x)
            };
            let sel_end_pos = if sel_end < positions.len() {
                positions[sel_end]
            } else {
                positions.last().copied().unwrap_or(text_pos.x)
            };
            
            let sel_bg = Quad {
                position: Vec2::new(sel_start_pos, input_rect.y),
                size: Vec2::new(sel_end_pos - sel_start_pos, input_rect.height),
                color: style::highlight::SELECTION,
                corner_radius: 0.0,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&sel_bg.to_vertices());
        }

        // Render cursor with blinking and smooth interpolation
        if input.focused && app.cursor_visible {
            // Use smoothly interpolated cursor position
            let cursor_pos_float = app.cursor_position_animation.value;
            let cursor_pos_floor = cursor_pos_float.floor() as usize;
            let cursor_pos_ceil = cursor_pos_float.ceil() as usize;
            let t = cursor_pos_float - cursor_pos_float.floor();
            
            let cursor_x = if cursor_pos_ceil < positions.len() && cursor_pos_floor < positions.len() {
                // Interpolate between two glyph positions for smooth movement
                let x1 = positions[cursor_pos_floor];
                let x2 = positions[cursor_pos_ceil];
                x1 + (x2 - x1) * t
            } else if cursor_pos_floor < positions.len() {
                positions[cursor_pos_floor]
            } else {
                // If at end or beyond, use last position or starting position
                positions.last().copied().unwrap_or(text_pos.x)
            };
            
            // Use cursor helper to get proper rect
            let cursor_rect = cursor::rect_at_position(&input_rect, cursor_x, font_size);
            
            let cursor_quad = Quad {
                position: cursor_rect.position(),
                size: cursor_rect.size(),
                color: style::text::PRIMARY,
                corner_radius: 0.0,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&cursor_quad.to_vertices());
        }
    }
}

