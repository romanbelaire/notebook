use glam::Vec2;

/// Unified trait for all text editing components
/// Each editor handles its own rendering, while the centralized router
/// in App handles input routing and cursor animation updates.
pub trait TextEditor {
    /// Handle character input
    fn on_char_received(&mut self, ch: char);
    
    /// Handle backspace (delete backward one character)
    fn on_backspace(&mut self);
    
    /// Handle Ctrl+Backspace (delete backward one word)
    fn on_backspace_word(&mut self);
    
    /// Handle Delete (delete forward one character)
    fn on_delete(&mut self);
    
    /// Handle arrow key navigation
    fn on_arrow_left(&mut self, shift: bool, ctrl: bool);
    fn on_arrow_right(&mut self, shift: bool, ctrl: bool);
    fn on_arrow_up(&mut self, shift: bool);
    fn on_arrow_down(&mut self, shift: bool);
    
    /// Handle Home/End keys
    fn on_home(&mut self, shift: bool);
    fn on_end(&mut self, shift: bool);
    
    /// Get current cursor position (for animation)
    fn get_cursor_position(&self) -> usize;
    
    /// Check if this editor is currently focused
    fn is_focused(&self) -> bool;
    
    /// Handle mouse down event (with click count for double/triple click)
    fn on_mouse_down(&mut self, pos: Vec2, click_count: u32);
    
    /// Handle mouse move event (for text selection)
    fn on_mouse_move(&mut self, pos: Vec2);
    
    /// Handle mouse up event
    fn on_mouse_up(&mut self);
    
    /// Check if point is within this editor's bounds
    fn contains(&self, pos: Vec2) -> bool;
    
    /// Focus this editor
    fn focus(&mut self);
    
    /// Blur this editor
    fn blur(&mut self);
}

