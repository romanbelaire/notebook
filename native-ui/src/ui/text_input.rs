use glam::Vec2;
use crate::ui::text_editor::TextEditor;
use crate::ui::components::Renderable;
use crate::ui::core::text_input_render;
use crate::ui::core::Rect;
use crate::ui::shadow::ShadowSpec;
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;

#[derive(Clone)]
pub struct TextInput {
    pub position: Vec2,
    pub size: Vec2,
    pub text: String,
    pub cursor_position: usize,
    /// Cached glyph x-positions for the current text; computed during render and reused for hit-testing.
    pub glyph_positions: Vec<f32>,
    pub selection_start: Option<usize>,  // Start of selection (can be after end if selecting backward)
    pub selection_end: Option<usize>,    // End of selection
    pub focused: bool,
    pub placeholder: String,
    /// Ephemeral display-only suggestion after the cursor (cleared on blur or real text change).
    pub ghost_text: String,
    pub cursor_blink_time: f32,
    pub is_selecting: bool,  // Whether user is currently dragging to select
    pub selection_anchor: Option<usize>,  // Anchor point for selection
    /// Cursor visibility state (updated from App during render)
    pub cursor_visible: bool,
    /// Cursor position animation value for smooth interpolation (updated from App)
    pub cursor_animation_value: f32,
    /// Internal rect for layout system
    pub rect: Rect,
    pub shadow: Option<ShadowSpec>,
}

impl TextInput {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        Self {
            position,
            size,
            text: String::new(),
            cursor_position: 0,
            glyph_positions: Vec::new(),
            selection_start: None,
            selection_end: None,
            focused: false,
            placeholder: "Type here...".to_string(),
            ghost_text: String::new(),
            cursor_blink_time: 0.0,
            is_selecting: false,
            selection_anchor: None,
            cursor_visible: true,
            cursor_animation_value: 0.0,
            rect: Rect::from_pos_size(position, size),
            shadow: None,
        }
    }

    /// Attach a drop shadow.
    pub fn with_shadow(mut self, spec: ShadowSpec) -> Self {
        self.shadow = Some(spec);
        self
    }

    pub fn set_placeholder(&mut self, placeholder: String) {
        self.placeholder = placeholder;
    }

    pub fn set_ghost_text(&mut self, s: String) {
        self.ghost_text = s;
    }

    pub fn clear_ghost_text(&mut self) {
        self.ghost_text.clear();
    }

    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.position.x
            && p.x <= self.position.x + self.size.x
            && p.y >= self.position.y
            && p.y <= self.position.y + self.size.y
    }

    pub fn on_focus(&mut self) {
        self.focused = true;
    }

    pub fn on_blur(&mut self) {
        self.focused = false;
        self.clear_ghost_text();
        self.selection_start = None;
        self.selection_end = None;
        self.is_selecting = false;
        self.selection_anchor = None;
        self.cursor_blink_time = 0.0;  // Reset cursor blink when unfocused
    }

    pub fn on_text_input(&mut self, text: &str) {
        if self.has_selection() {
            self.delete_selection();
        }
        
        // Ensure cursor position is valid before insertion
        self.ensure_cursor_valid();
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position;
        
        let mut chars: Vec<char> = self.text.chars().collect();
        let text_chars: Vec<char> = text.chars().collect();
        
        chars.splice(cursor_pos..cursor_pos, text_chars.iter().cloned());
        self.text = chars.into_iter().collect();
        // Cursor moves forward by the length of inserted text
        self.cursor_position = cursor_pos + text_chars.len();
        // Ensure cursor position is still valid after text change
        self.ensure_cursor_valid();
        
        // Clear selection after inserting text
        self.selection_start = None;
        self.selection_end = None;
        self.clear_ghost_text();
    }

    pub fn on_char_received(&mut self, ch: char) {
        self.on_text_input(&ch.to_string());
    }

    pub fn on_backspace(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }

        // Ensure cursor position is valid before deletion
        self.ensure_cursor_valid();
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position;
        
        if cursor_pos > 0 {
            let mut chars: Vec<char> = self.text.chars().collect();
            chars.remove(cursor_pos - 1);
            self.text = chars.into_iter().collect();
            // Cursor moves backward by 1
            self.cursor_position = cursor_pos - 1;
            // Ensure cursor position is still valid after text change
            self.ensure_cursor_valid();
            self.clear_ghost_text();
        }
    }
    
    pub fn on_backspace_word(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }
        
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position.min(text_len);
        
        if cursor_pos == 0 {
            return;
        }
        
        let chars: Vec<char> = self.text.chars().collect();
        let mut new_pos = cursor_pos;
        
        // Skip whitespace
        while new_pos > 0 && chars[new_pos - 1].is_whitespace() {
            new_pos -= 1;
        }
        
        // Skip word characters
        while new_pos > 0 && !chars[new_pos - 1].is_whitespace() {
            new_pos -= 1;
        }
        
        // Delete from new_pos to cursor_pos
        if new_pos < cursor_pos {
            let mut chars: Vec<char> = self.text.chars().collect();
            chars.drain(new_pos..cursor_pos);
            self.text = chars.into_iter().collect();
            self.cursor_position = new_pos;
            self.ensure_cursor_valid();
            self.clear_ghost_text();
        }
    }

    pub fn on_delete(&mut self) {
        if self.has_selection() {
            self.delete_selection();
            return;
        }

        // Standard delete behavior: delete character at cursor, cursor stays in place
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position.min(text_len);
        
        if cursor_pos < text_len {
            // Delete the character at the cursor position
            let mut chars: Vec<char> = self.text.chars().collect();
            chars.remove(cursor_pos);
            self.text = chars.into_iter().collect();
            // Cursor position stays the same (standard forward delete behavior)
            // The character that was at cursor_pos is now gone, but cursor stays at cursor_pos
            self.cursor_position = cursor_pos;
            // Ensure cursor is still valid
            self.ensure_cursor_valid();
            self.clear_ghost_text();
        }
    }

    pub fn move_cursor_left(&mut self, extend_selection: bool) {
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position.min(text_len);
        
        if cursor_pos > 0 {
            if extend_selection {
                if self.selection_anchor.is_none() {
                    self.selection_anchor = Some(cursor_pos);
                }
                self.cursor_position = cursor_pos - 1;
                self.update_selection_from_anchor();
            } else {
                self.cursor_position = cursor_pos - 1;
                self.clear_selection();
            }
        } else if !extend_selection {
            self.clear_selection();
        }
    }

    pub fn move_cursor_right(&mut self, extend_selection: bool) {
        self.ensure_cursor_valid();
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position;
        
        if cursor_pos < text_len {
            if extend_selection {
                if self.selection_anchor.is_none() {
                    self.selection_anchor = Some(cursor_pos);
                }
                self.cursor_position = cursor_pos + 1;
                self.update_selection_from_anchor();
            } else {
                self.cursor_position = cursor_pos + 1;
                self.clear_selection();
            }
        } else if !extend_selection {
            self.clear_selection();
        }
        self.ensure_cursor_valid();
    }

    pub fn move_cursor_to_start(&mut self, extend_selection: bool) {
        if extend_selection {
            if self.selection_anchor.is_none() {
                let text_len = self.text.chars().count();
                self.selection_anchor = Some(self.cursor_position.min(text_len));
            }
            self.cursor_position = 0;
            self.update_selection_from_anchor();
        } else {
            self.cursor_position = 0;
            self.clear_selection();
        }
    }

    pub fn move_cursor_to_end(&mut self, extend_selection: bool) {
        let text_len = self.text.chars().count();
        if extend_selection {
            if self.selection_anchor.is_none() {
                self.selection_anchor = Some(self.cursor_position.min(text_len));
            }
            self.cursor_position = text_len;
            self.update_selection_from_anchor();
        } else {
            self.cursor_position = text_len;
            self.clear_selection();
        }
    }

    pub fn move_cursor_word_left(&mut self, extend_selection: bool) {
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position.min(text_len);
        
        if cursor_pos == 0 {
            if !extend_selection {
                self.clear_selection();
            }
            return;
        }
        
        let chars: Vec<char> = self.text.chars().collect();
        let mut new_pos = cursor_pos;
        
        // Skip whitespace
        while new_pos > 0 && chars[new_pos - 1].is_whitespace() {
            new_pos -= 1;
        }
        
        // Skip word characters
        while new_pos > 0 && !chars[new_pos - 1].is_whitespace() {
            new_pos -= 1;
        }
        
        if extend_selection {
            if self.selection_anchor.is_none() {
                self.selection_anchor = Some(cursor_pos);
            }
            self.cursor_position = new_pos;
            self.update_selection_from_anchor();
        } else {
            self.cursor_position = new_pos;
            self.clear_selection();
        }
    }

    pub fn move_cursor_word_right(&mut self, extend_selection: bool) {
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position.min(text_len);
        
        if cursor_pos >= text_len {
            if !extend_selection {
                self.clear_selection();
            }
            return;
        }
        
        let chars: Vec<char> = self.text.chars().collect();
        let mut new_pos = cursor_pos;
        
        // Skip word characters
        while new_pos < text_len && !chars[new_pos].is_whitespace() {
            new_pos += 1;
        }
        
        // Skip whitespace
        while new_pos < text_len && chars[new_pos].is_whitespace() {
            new_pos += 1;
        }
        
        if extend_selection {
            if self.selection_anchor.is_none() {
                self.selection_anchor = Some(cursor_pos);
            }
            self.cursor_position = new_pos;
            self.update_selection_from_anchor();
        } else {
            self.cursor_position = new_pos;
            self.clear_selection();
        }
    }

    pub fn select_all(&mut self) {
        let text_len = self.text.chars().count();
        self.selection_start = Some(0);
        self.selection_end = Some(text_len);
        self.cursor_position = text_len;
        self.selection_anchor = Some(0);
    }

    pub fn has_selection(&self) -> bool {
        if let (Some(start), Some(end)) = (self.selection_start, self.selection_end) {
            start != end
        } else {
            false
        }
    }

    pub fn get_selection_range(&self) -> Option<(usize, usize)> {
        if let (Some(start), Some(end)) = (self.selection_start, self.selection_end) {
            if start != end {
                Some((start.min(end), start.max(end)))
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Ensures cursor_position is always valid (between 0 and text length, inclusive)
    /// This should be called after any text modification to prevent cursor from going out of bounds
    pub fn ensure_cursor_valid(&mut self) {
        let text_len = self.text.chars().count();
        // Clamp cursor to valid range: [0, text_len]
        if self.cursor_position > text_len {
            self.cursor_position = text_len;
        }
        // cursor_position is usize, so it can't be < 0
    }

    pub fn clear_selection(&mut self) {
        self.selection_start = None;
        self.selection_end = None;
        self.selection_anchor = None;
    }

    fn update_selection_from_anchor(&mut self) {
        if let Some(anchor) = self.selection_anchor {
            let cursor_pos = self.cursor_position;
            self.selection_start = Some(anchor.min(cursor_pos));
            self.selection_end = Some(anchor.max(cursor_pos));
        }
    }

    pub fn delete_selection(&mut self) {
        if let Some((start, end)) = self.get_selection_range() {
            let mut chars: Vec<char> = self.text.chars().collect();
            let start_idx = start.min(chars.len());
            let end_idx = end.min(chars.len());
            chars.drain(start_idx..end_idx);
            self.text = chars.into_iter().collect();
            // Cursor moves to start of selection after deletion
            self.cursor_position = start_idx;
            // Ensure cursor position is still valid after text change
            self.ensure_cursor_valid();
            self.clear_selection();
            self.clear_ghost_text();
        }
    }

    /// Get cursor position from precomputed glyph positions (more efficient)
    pub fn get_cursor_position_from_positions(&self, point: Vec2, positions: &[f32]) -> usize {
        let rel_x = (point.x - self.position.x - 5.0).max(0.0);
        
        // Binary search for cursor position using precomputed positions
        let mut low = 0;
        let mut high = positions.len().saturating_sub(1);
        
        while low < high {
            let mid = (low + high) / 2;
            let pos_x = positions[mid];
            
            if pos_x < rel_x {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        // Find the closest position
        if low > 0 && (rel_x - positions[low - 1]).abs() < (rel_x - positions[low]).abs() {
            low - 1
        } else {
            low.min(positions.len().saturating_sub(1))
        }
    }

    pub fn on_mouse_down(&mut self, pos: Vec2, click_count: u32) {
        if self.contains(pos) {
            let new_pos = self.get_cursor_position_from_positions(pos, &self.glyph_positions);
            self.cursor_position = new_pos.min(self.text.chars().count());
            self.selection_anchor = Some(self.cursor_position);
            self.is_selecting = true;
            self.clear_selection();
            
            // Handle double/triple click
            match click_count {
                2 => {
                    // Double click: select word
                    self.select_word_at_cursor();
                }
                3 => {
                    // Triple click: select line
                    self.select_line();
                }
                _ => {
                    // Single click: just position cursor
                }
            }
        }
    }
    
    pub fn select_word_at_cursor(&mut self) {
        let text_len = self.text.chars().count();
        let cursor_pos = self.cursor_position.min(text_len);
        
        if text_len == 0 {
            return;
        }
        
        let chars: Vec<char> = self.text.chars().collect();
        
        // Find word boundaries
        let mut start = cursor_pos;
        let mut end = cursor_pos;
        
        // If cursor is on whitespace, select the whitespace
        if cursor_pos < text_len && chars[cursor_pos].is_whitespace() {
            // Select backward to find start of whitespace
            while start > 0 && chars[start - 1].is_whitespace() {
                start -= 1;
            }
            // Select forward to find end of whitespace
            while end < text_len && chars[end].is_whitespace() {
                end += 1;
            }
        } else {
            // Select word characters backward
            while start > 0 && !chars[start - 1].is_whitespace() {
                start -= 1;
            }
            // Select word characters forward
            while end < text_len && !chars[end].is_whitespace() {
                end += 1;
            }
        }
        
        self.selection_start = Some(start);
        self.selection_end = Some(end);
        self.cursor_position = end;
        self.selection_anchor = Some(start);
    }
    
    pub fn select_line(&mut self) {
        // Select entire line (all text for single-line input)
        let text_len = self.text.chars().count();
        self.selection_start = Some(0);
        self.selection_end = Some(text_len);
        self.cursor_position = text_len;
        self.selection_anchor = Some(0);
    }

    pub fn on_mouse_move(&mut self, pos: Vec2) {
        if self.is_selecting && self.contains(pos) {
            let new_pos = self.get_cursor_position_from_positions(pos, &self.glyph_positions);
            self.cursor_position = new_pos.min(self.text.chars().count());
            if let Some(anchor) = self.selection_anchor {
                self.selection_start = Some(anchor.min(self.cursor_position));
                self.selection_end = Some(anchor.max(self.cursor_position));
            }
        }
    }

    pub fn on_mouse_up(&mut self) {
        self.is_selecting = false;
        // Keep selection_anchor for Shift+Arrow key navigation
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor_position = 0;
        self.clear_selection();
        self.clear_ghost_text();
    }

    pub fn update(&mut self, dt: f32) {
        if self.focused {
            self.cursor_blink_time += dt;
            if self.cursor_blink_time > 1.0 {
                self.cursor_blink_time -= 1.0;
            }
        } else {
            self.cursor_blink_time = 0.0;
        }
    }

    pub fn is_cursor_visible(&self) -> bool {
        if !self.focused {
            return false;
        }
        self.cursor_blink_time < 0.5
    }

    pub fn set_glyph_positions(&mut self, positions: Vec<f32>) {
        self.glyph_positions = positions;
    }
    
    pub fn get_selected_text(&self) -> String {
        if let Some((start, end)) = self.get_selection_range() {
            let chars: Vec<char> = self.text.chars().collect();
            chars[start..end].iter().collect()
        } else {
            String::new()
        }
    }
    
    pub fn paste(&mut self, text: &str) {
        if self.has_selection() {
            self.delete_selection();
        }
        self.on_text_input(text);
    }
}

impl TextEditor for TextInput {
    fn on_char_received(&mut self, ch: char) {
        self.on_char_received(ch);
    }
    
    fn on_backspace(&mut self) {
        self.on_backspace();
    }
    
    fn on_backspace_word(&mut self) {
        self.on_backspace_word();
    }
    
    fn on_delete(&mut self) {
        self.on_delete();
    }
    
    fn on_arrow_left(&mut self, shift: bool, ctrl: bool) {
        if ctrl {
            self.move_cursor_word_left(shift);
        } else {
            self.move_cursor_left(shift);
        }
    }
    
    fn on_arrow_right(&mut self, shift: bool, ctrl: bool) {
        if ctrl {
            self.move_cursor_word_right(shift);
        } else {
            self.move_cursor_right(shift);
        }
    }
    
    fn on_arrow_up(&mut self, _shift: bool) {
        // Single-line input, move to start
        self.move_cursor_to_start(false);
    }
    
    fn on_arrow_down(&mut self, _shift: bool) {
        // Single-line input, move to end
        self.move_cursor_to_end(false);
    }
    
    fn on_home(&mut self, shift: bool) {
        self.move_cursor_to_start(shift);
    }
    
    fn on_end(&mut self, shift: bool) {
        self.move_cursor_to_end(shift);
    }
    
    fn get_cursor_position(&self) -> usize {
        self.cursor_position
    }
    
    fn is_focused(&self) -> bool {
        self.focused
    }
    
    fn on_mouse_down(&mut self, pos: Vec2, click_count: u32) {
        if self.contains(pos) {
            let new_pos = self.get_cursor_position_from_positions(pos, &self.glyph_positions);
            self.cursor_position = new_pos.min(self.text.chars().count());
            self.selection_anchor = Some(self.cursor_position);
            self.is_selecting = true;
            self.clear_selection();
            
            // Handle double/triple click
            match click_count {
                2 => {
                    self.select_word_at_cursor();
                }
                3 => {
                    self.select_line();
                }
                _ => {}
            }
        }
    }
    
    fn on_mouse_move(&mut self, pos: Vec2) {
        if self.is_selecting && self.contains(pos) {
            let new_pos = self.get_cursor_position_from_positions(pos, &self.glyph_positions);
            self.cursor_position = new_pos.min(self.text.chars().count());
            if let Some(anchor) = self.selection_anchor {
                self.selection_start = Some(anchor.min(self.cursor_position));
                self.selection_end = Some(anchor.max(self.cursor_position));
            }
        }
    }
    
    fn on_mouse_up(&mut self) {
        self.on_mouse_up();
    }
    
    fn contains(&self, pos: Vec2) -> bool {
        self.contains(pos)
    }
    
    fn focus(&mut self) {
        self.on_focus();
    }
    
    fn blur(&mut self) {
        self.on_blur();
    }
}

use crate::app::App;

impl Renderable for TextInput {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        let component_id = format!("textinput_{:p}", self);
        renderer.validate_component(&component_id, None, "TextInput");

        use crate::ui::style;

        const DEFAULT_FONT_SIZE: f32 = style::font_size::NORMAL;
        const DEFAULT_PADDING: f32 = style::padding::SMALL;
        const DEFAULT_CORNER_RADIUS: f32 = style::corner_radius::SMALL;

        if let Some(spec) = &self.shadow {
            renderer.queue_shadow(&self.rect, DEFAULT_CORNER_RADIUS, spec);
        }

        text_input_render::render_text_input_inner(
            renderer,
            self,
            app,
            vertices,
            Some(DEFAULT_FONT_SIZE),
            Some(DEFAULT_PADDING),
            Some(DEFAULT_CORNER_RADIUS),
            false,
            dirty_rect,
        );
    }
    
    fn bounds(&self) -> Rect {
        self.rect
    }
    
    fn update_layout(&mut self, available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
        self.rect = available_rect;
        self.position = available_rect.position();
        self.size = available_rect.size();
    }
    
    fn min_size(&self) -> Vec2 {
        // TextInput has a minimum height based on font size
        Vec2::new(100.0, 40.0) // Minimum width 100, height 40
    }

    fn corner_radius(&self) -> f32 {
        crate::ui::style::corner_radius::SMALL
    }
}

