use glam::Vec2;
use crate::utils::animation::{SpringAnimation, AnimationPreset};

pub struct ScrollView {
    pub position: Vec2,
    pub size: Vec2,
    pub content_height: f32,
    pub scroll_offset: f32,
    pub target_scroll_offset: f32,
    pub scroll_offset_animation: SpringAnimation,
    pub scrollbar_width: f32,
    pub is_scrolling: bool,
    pub scroll_velocity: f32,
    // Unified highlight bar animation
    pub highlight_bar_y: f32,
    pub highlight_bar_target_y: f32,
    pub highlight_bar_animation: SpringAnimation,
    pub highlight_bar_height: f32,
    pub highlight_bar_visible: bool,
    // Selection border animation (for active selection) - uses opacity fade instead of position
    pub selection_border_opacity: f32,
    pub selection_border_animation: SpringAnimation,
    pub selection_border_y: f32,  // Static position, only opacity animates
    pub selection_border_height: f32,
    pub selection_border_visible: bool,
}

impl ScrollView {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        let scroll_animation = SpringAnimation::with_preset(0.0, AnimationPreset::Snappy);
        
        // Highlight bar animation - use Snappy instead of Bouncy to avoid continuous oscillation
        // Snappy has higher damping (20.0 vs 12.0) which prevents flickering
        let highlight_animation = SpringAnimation::with_preset(0.0, AnimationPreset::Snappy);
        
        // Selection border animation - uses opacity fade (0.0 = invisible, 1.0 = visible)
        let selection_border_animation = SpringAnimation::with_preset(0.0, AnimationPreset::Snappy);
        
        Self {
            position,
            size,
            content_height: 0.0,
            scroll_offset: 0.0,
            target_scroll_offset: 0.0,
            scroll_offset_animation: scroll_animation,
            scrollbar_width: 10.0,
            is_scrolling: false,
            scroll_velocity: 0.0,
            highlight_bar_y: 0.0,
            highlight_bar_target_y: 0.0,
            highlight_bar_animation: highlight_animation,
            highlight_bar_height: 40.0,  // Default item height
            highlight_bar_visible: false,
            selection_border_opacity: 0.0,
            selection_border_animation: selection_border_animation,
            selection_border_y: 0.0,
            selection_border_height: 40.0,  // Default item height
            selection_border_visible: false,
        }
    }
    
    pub fn set_highlight_target(&mut self, y: f32) {
        // Only update target if it actually changed (avoid unnecessary animation resets)
        if (self.highlight_bar_target_y - y).abs() > 0.01 {
            self.highlight_bar_target_y = y;
            self.highlight_bar_animation.target = y;
            self.highlight_bar_visible = true;
        }
    }
    
    pub fn clear_highlight(&mut self) {
        self.highlight_bar_visible = false;
    }
    
    pub fn set_selection_border_target(&mut self, y: f32) {
        // Update position immediately (no animation)
        self.selection_border_y = y;
        // Fade in by setting opacity target to 1.0
        self.selection_border_animation.target = 1.0;
        self.selection_border_visible = true;
    }
    
    pub fn clear_selection_border(&mut self) {
        // Fade out by setting opacity target to 0.0
        self.selection_border_animation.target = 0.0;
        // Hide when fully faded out
        if self.selection_border_opacity < 0.01 {
            self.selection_border_visible = false;
        }
    }

    /// Set row/item height for highlight bar and selection border (e.g. 40.0 or 35.0).
    pub fn set_item_height(&mut self, height: f32) {
        self.highlight_bar_height = height;
        self.selection_border_height = height;
    }

    pub fn set_content_height(&mut self, height: f32) {
        self.content_height = height;
        self.clamp_target_scroll();
        self.scroll_offset_animation.target = self.target_scroll_offset;
    }

    pub fn scroll(&mut self, delta: f32) {
        self.target_scroll_offset += delta;
        self.clamp_target_scroll();
        self.scroll_offset_animation.target = self.target_scroll_offset;
    }

    pub fn scroll_to(&mut self, offset: f32) {
        self.target_scroll_offset = offset;
        self.clamp_target_scroll();
        self.scroll_offset_animation.target = self.target_scroll_offset;
    }

    pub fn scroll_to_bottom(&mut self) {
        self.target_scroll_offset = (self.content_height - self.size.y).max(0.0);
        self.scroll_offset_animation.target = self.target_scroll_offset;
    }
    
    fn clamp_target_scroll(&mut self) {
        let max_scroll = (self.content_height - self.size.y).max(0.0);
        self.target_scroll_offset = self.target_scroll_offset.max(0.0).min(max_scroll);
    }


    pub fn visible_rect(&self) -> (Vec2, Vec2) {
        (
            Vec2::new(self.position.x, self.position.y - self.scroll_offset),
            Vec2::new(self.size.x, self.size.y),
        )
    }

    pub fn needs_scrollbar(&self) -> bool {
        self.content_height > self.size.y
    }

    pub fn scrollbar_thumb_height(&self) -> f32 {
        if !self.needs_scrollbar() {
            return 0.0;
        }
        (self.size.y / self.content_height) * self.size.y
    }

    pub fn scrollbar_thumb_position(&self) -> f32 {
        if !self.needs_scrollbar() {
            return 0.0;
        }
        let ratio = self.scroll_offset / (self.content_height - self.size.y);
        ratio * (self.size.y - self.scrollbar_thumb_height())
    }

    pub fn hit_test(&self, pos: Vec2) -> ScrollHit {
        let rel = pos - self.position;
        if rel.x < 0.0 || rel.x > self.size.x || rel.y < 0.0 || rel.y > self.size.y {
            return ScrollHit::Outside;
        }

        if self.needs_scrollbar() && rel.x > self.size.x - self.scrollbar_width {
            return ScrollHit::Scrollbar;
        }

        ScrollHit::Content
    }

    pub fn update(&mut self, dt: f32) {
        // Update smooth scroll animation
        self.scroll_offset_animation.update(dt);
        self.scroll_offset = self.scroll_offset_animation.value;
        
        // Update highlight bar animation
        if self.highlight_bar_visible {
            // Only update if animation hasn't settled (avoid unnecessary updates)
            if !self.highlight_bar_animation.is_at_target() {
                self.highlight_bar_animation.update(dt);
            }
            self.highlight_bar_y = self.highlight_bar_animation.value;
        }
        
        // Update selection border animation (opacity fade)
        if self.selection_border_visible || self.selection_border_animation.value > 0.01 {
            // Continue animating even if not visible (for fade out)
            if !self.selection_border_animation.is_at_target() {
                self.selection_border_animation.update(dt);
            }
            self.selection_border_opacity = self.selection_border_animation.value;
            // Hide when fully faded out
            if self.selection_border_opacity < 0.01 {
                self.selection_border_visible = false;
            }
        }
        
        // Legacy velocity-based scrolling (for momentum)
        if !self.is_scrolling {
            self.scroll_velocity *= 0.9;  // Friction
            if self.scroll_velocity.abs() > 0.1 {
                self.target_scroll_offset += self.scroll_velocity * dt;
                self.clamp_target_scroll();
                self.scroll_offset_animation.target = self.target_scroll_offset;
            }
        }
    }

    /// True if scroll or highlight/selection animations are still moving (needs continuous redraw).
    pub fn has_active_animation(&self) -> bool {
        if !self.scroll_offset_animation.is_at_target() {
            return true;
        }
        if self.highlight_bar_visible && !self.highlight_bar_animation.is_at_target() {
            return true;
        }
        if (self.selection_border_visible || self.selection_border_animation.value > 0.01)
            && !self.selection_border_animation.is_at_target()
        {
            return true;
        }
        if self.scroll_velocity.abs() > 0.1 {
            return true;
        }
        false
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollHit {
    Content,
    Scrollbar,
    Outside,
}

