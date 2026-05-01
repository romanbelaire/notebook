use glam::Vec2;
use crate::utils::animation::{SpringAnimation, AnimationPreset};
use serde::{Serialize, Deserialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum Tab {
    Chat,
    Notepad,
    Library,
    Data,
    Settings,
}

impl Tab {
    pub fn all() -> Vec<Tab> {
        vec![Tab::Chat, Tab::Notepad, Tab::Library, Tab::Data, Tab::Settings]
    }

    pub fn label(&self) -> &'static str {
        match self {
            Tab::Chat => "Chat",
            Tab::Notepad => "Notepad",
            Tab::Library => "Library",
            Tab::Data => "Data",
            Tab::Settings => "Settings",
        }
    }
}

pub struct TabBar {
    pub tabs: Vec<Tab>,
    pub active_index: usize,
    pub position: Vec2,
    pub size: Vec2,
    pub slider_animation: SpringAnimation,
    /// Trailing edge lags behind so the pill stretches (leading moves first, following catches up).
    pub slider_trailing_animation: SpringAnimation,
    pub hovered_index: Option<usize>,
}

impl TabBar {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        let tabs = Tab::all();
        let active_index = 0;
        
        // Leading and trailing edges: smooth spring (no SDF liquid stretch)
        let mut slider_animation =
            SpringAnimation::with_preset(active_index as f32, AnimationPreset::Mechanical);
        slider_animation.target = active_index as f32;
        slider_animation.value = active_index as f32;

        let mut slider_trailing_animation =
            SpringAnimation::with_preset(active_index as f32, AnimationPreset::Mechanical);
        slider_trailing_animation.target = active_index as f32;
        slider_trailing_animation.value = active_index as f32;

        Self {
            tabs,
            active_index,
            position,
            size,
            slider_animation,
            slider_trailing_animation,
            hovered_index: None,
        }
    }

    pub fn set_active(&mut self, index: usize) {
        if index < self.tabs.len() {
            self.active_index = index;
            self.slider_animation.target = index as f32;
            self.slider_trailing_animation.target = index as f32;
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.slider_animation.update(dt);
        self.slider_trailing_animation.update(dt);
    }

    pub fn hit_test(&self, pos: Vec2) -> Option<usize> {
        let rel = pos - self.position;
        if rel.x < 0.0 || rel.x > self.size.x || rel.y < 0.0 || rel.y > self.size.y {
            return None;
        }

        let tab_width = self.size.x / self.tabs.len() as f32;
        let index = (rel.x / tab_width).floor() as usize;
        if index < self.tabs.len() {
            Some(index)
        } else {
            None
        }
    }

    pub fn on_mouse_move(&mut self, pos: Vec2) {
        self.hovered_index = self.hit_test(pos);
    }

    pub fn on_mouse_click(&mut self, pos: Vec2) -> Option<usize> {
        if let Some(index) = self.hit_test(pos) {
            self.set_active(index);
            Some(index)
        } else {
            None
        }
    }

    pub fn slider_position(&self) -> f32 {
        let tab_width = self.size.x / self.tabs.len() as f32;
        let leading = self.slider_animation.value;
        let following = self.slider_trailing_animation.value;
        leading.min(following) * tab_width
    }

    pub fn slider_width(&self) -> f32 {
        let tab_width = self.size.x / self.tabs.len() as f32;
        let leading = self.slider_animation.value;
        let following = self.slider_trailing_animation.value;
        (leading.max(following) - leading.min(following)) * tab_width + tab_width
    }
}

