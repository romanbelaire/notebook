//! Reusable section list: scroll, hover highlight, selection border, and
//! collapsible row actions (expand handle → edit/delete buttons).
//! Use one SectionList per sidebar section (Conversations, Documents, Insights).

use glam::Vec2;
use std::collections::HashMap;
use crate::ui::ScrollView;
use crate::utils::animation::{SpringAnimation, AnimationPreset};

pub struct SectionList {
    pub scroll_view: ScrollView,
    pub item_height: f32,
    /// Which row has expanded actions (handle clicked). None = all collapsed.
    pub expanded_index: Option<usize>,
    /// Per-row expand animation (0.0 = collapsed, 1.0 = expanded).
    expand_animations: HashMap<usize, SpringAnimation>,
}

impl SectionList {
    pub fn new(position: Vec2, size: Vec2, item_height: f32) -> Self {
        let mut scroll_view = ScrollView::new(position, size);
        scroll_view.set_item_height(item_height);
        Self {
            scroll_view,
            item_height,
            expanded_index: None,
            expand_animations: HashMap::new(),
        }
    }

    /// Update scroll, highlight, selection, and expand animations. Call every frame.
    pub fn update(&mut self, dt: f32, item_count: usize) {
        self.scroll_view.update(dt);
        for index in 0..item_count {
            self.ensure_expand_animation(index);
        }
        let expanded = self.expanded_index;
        self.expand_animations.retain(|&index, animation| {
            if index >= item_count {
                return false;
            }
            let target = if Some(index) == expanded { 1.0 } else { 0.0 };
            animation.set_target(target);
            !animation.is_at_target() || Some(index) == expanded
        });
        for animation in self.expand_animations.values_mut() {
            animation.update(dt);
        }
    }

    /// True if scroll or expand animations are still moving (needs continuous redraw).
    pub fn has_active_animation(&self) -> bool {
        if self.scroll_view.has_active_animation() {
            return true;
        }
        self.expand_animations
            .values()
            .any(|a| !a.is_at_target())
    }

    /// Expand animation value for a row (0.0 = collapsed, 1.0 = expanded).
    pub fn get_expand_animation(&self, index: usize) -> f32 {
        if let Some(animation) = self.expand_animations.get(&index) {
            animation.value
        } else {
            if self.expanded_index == Some(index) {
                1.0
            } else {
                0.0
            }
        }
    }

    fn ensure_expand_animation(&mut self, index: usize) {
        let target = if self.expanded_index == Some(index) { 1.0 } else { 0.0 };
        let animation = self.expand_animations
            .entry(index)
            .or_insert_with(|| {
                let mut anim = SpringAnimation::with_preset(0.0, AnimationPreset::Snappy);
                anim.set_target(target);
                anim
            });
        if animation.target != target {
            animation.set_target(target);
        }
    }

    /// Index of item at position (absolute coords), or None.
    pub fn get_item_at(&self, pos: Vec2, item_count: usize) -> Option<usize> {
        if !self.scroll_view.contains(pos) {
            return None;
        }
        let rel = pos - self.scroll_view.position;
        let index = ((rel.y + self.scroll_view.scroll_offset) / self.item_height) as usize;
        if index < item_count {
            Some(index)
        } else {
            None
        }
    }

    /// World-space Y for highlight/selection bar for a given item index.
    pub fn item_y_for_index(&self, index: usize) -> f32 {
        self.scroll_view.position.y + (index as f32 * self.item_height) - self.scroll_view.scroll_offset
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        self.scroll_view.contains(pos)
    }

    pub fn scroll(&mut self, delta: f32) {
        self.scroll_view.scroll(delta);
    }

    pub fn set_highlight_target(&mut self, y: f32) {
        self.scroll_view.set_highlight_target(y);
    }

    pub fn clear_highlight(&mut self) {
        self.scroll_view.clear_highlight();
    }

    pub fn set_selection_border_target(&mut self, y: f32) {
        self.scroll_view.set_selection_border_target(y);
    }

    pub fn clear_selection_border(&mut self) {
        self.scroll_view.clear_selection_border();
    }

    pub fn set_content_height(&mut self, height: f32) {
        self.scroll_view.set_content_height(height);
    }

    /// Position and size (delegate to scroll_view for layout).
    pub fn set_position_size(&mut self, position: Vec2, size: Vec2) {
        self.scroll_view.position = position;
        self.scroll_view.size = size;
    }
}
