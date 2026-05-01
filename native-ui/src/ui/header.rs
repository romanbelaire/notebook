use glam::Vec2;
use crate::ui::tab_bar::TabBar;

pub struct HeaderWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub title: String,
    pub tab_bar: TabBar,
    pub show_window_controls: bool,
    pub minimize_button: crate::ui::Button,
    pub maximize_button: crate::ui::Button,
    pub close_button: crate::ui::Button,
    pub is_maximized: bool,
}

impl HeaderWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        use crate::ui::style;
        
        let tab_bar_width = style::hero::TAB_BAR_WIDTH;
        let tab_bar_height = style::hero::TAB_BAR_HEIGHT;
        let tab_bar_x = style::center_x(size.x, tab_bar_width);
        let tab_bar_y = size.y - style::stroke::INSTRUMENT_RULE_PX - tab_bar_height;

        let tab_bar = TabBar::new(
            Vec2::new(tab_bar_x, tab_bar_y),
            Vec2::new(tab_bar_width, tab_bar_height),
        );

        let button_size = Vec2::new(30.0, 30.0);
        let button_spacing = style::padding::TINY;
        let controls_x = size.x - button_size.x * 3.0 - button_spacing * 2.0 - style::padding::SMALL;
        
        Self {
            position,
            size,
            title: "Constellar".to_string(),
            tab_bar,
            show_window_controls: true,
            minimize_button: crate::ui::Button::new(
                Vec2::new(controls_x, style::padding::SMALL),
                button_size,
                "−",
            ),
            maximize_button: crate::ui::Button::new(
                Vec2::new(controls_x + button_size.x + button_spacing, style::padding::SMALL),
                button_size,
                "□",
            ),
            close_button: crate::ui::Button::new(
                Vec2::new(controls_x + (button_size.x + button_spacing) * 2.0, style::padding::SMALL),
                button_size,
                "×",
            ),
            is_maximized: false,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.tab_bar.update(dt);
    }

    pub fn hit_test(&self, pos: Vec2) -> Option<HeaderHit> {
        let rel = pos - self.position;
        if rel.x < 0.0 || rel.x > self.size.x || rel.y < 0.0 || rel.y > self.size.y {
            return None;
        }

        if let Some(tab_index) = self.tab_bar.hit_test(rel) {
            return Some(HeaderHit::Tab(tab_index));
        }

        Some(HeaderHit::Background)
    }

    pub fn on_mouse_move(&mut self, pos: Vec2) {
        let rel = pos - self.position;
        self.tab_bar.on_mouse_move(rel);
    }

    pub fn on_mouse_click(&mut self, pos: Vec2) -> Option<HeaderClick> {
        let rel = pos - self.position;
        if let Some(tab_index) = self.tab_bar.on_mouse_click(rel) {
            return Some(HeaderClick::Tab(tab_index));
        }
        
        // Check window controls
        if self.minimize_button.contains(rel) {
            return Some(HeaderClick::Minimize);
        }
        if self.maximize_button.contains(rel) {
            return Some(HeaderClick::Maximize);
        }
        if self.close_button.contains(rel) {
            return Some(HeaderClick::Close);
        }
        
        None
    }
    
    pub fn update_layout(&mut self, viewport_size: Vec2) {
        use crate::ui::style;
        
        self.size = Vec2::new(viewport_size.x, self.size.y);
        
        // Update tab bar position (keep it centered)
        let tab_bar_width = style::hero::TAB_BAR_WIDTH;
        let tab_bar_x = style::center_x(viewport_size.x, tab_bar_width);
        self.tab_bar.position = Vec2::new(
            tab_bar_x,
            self.size.y - style::stroke::INSTRUMENT_RULE_PX - self.tab_bar.size.y,
        );
        
        // Update window control buttons
        let button_size = Vec2::new(30.0, 30.0);
        let button_spacing = style::padding::TINY;
        let controls_x = viewport_size.x - button_size.x * 3.0 - button_spacing * 2.0 - style::padding::SMALL;
        
        self.minimize_button.position = Vec2::new(controls_x, style::padding::SMALL);
        self.maximize_button.position = Vec2::new(controls_x + button_size.x + button_spacing, style::padding::SMALL);
        self.close_button.position = Vec2::new(controls_x + (button_size.x + button_spacing) * 2.0, style::padding::SMALL);
    }
}

#[derive(Debug)]
pub enum HeaderHit {
    Tab(usize),
    Background,
}

#[derive(Debug)]
pub enum HeaderClick {
    Tab(usize),
    Minimize,
    Maximize,
    Close,
}

