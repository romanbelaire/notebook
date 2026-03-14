use glam::Vec2;
use crate::ui::{Button, ButtonState};
use crate::ui::core::Rect;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolbarButton {
    Bold,
    Italic,
    Underline,
    Strikethrough,
    Code,
    Link,
}

pub struct Toolbar {
    pub position: Vec2,
    pub size: Vec2,
    pub bold_button: Button,
    pub italic_button: Button,
    pub underline_button: Button,
    pub strikethrough_button: Button,
    pub code_button: Button,
    pub link_button: Button,
    pub button_size: Vec2,
    pub button_spacing: f32,
}

impl Toolbar {
    pub fn new(position: Vec2, width: f32) -> Self {
        const BUTTON_HEIGHT: f32 = 32.0;
        const BUTTON_WIDTH: f32 = 32.0;
        const BUTTON_SPACING: f32 = 4.0;
        
        let button_size = Vec2::new(BUTTON_WIDTH, BUTTON_HEIGHT);
        let toolbar_height = BUTTON_HEIGHT;
        
        Self {
            position,
            size: Vec2::new(width, toolbar_height),
            bold_button: Button::new(Vec2::ZERO, button_size, "B"),
            italic_button: Button::new(Vec2::ZERO, button_size, "I"),
            underline_button: Button::new(Vec2::ZERO, button_size, "U"),
            strikethrough_button: Button::new(Vec2::ZERO, button_size, "S"),
            code_button: Button::new(Vec2::ZERO, button_size, "</>"),
            link_button: Button::new(Vec2::ZERO, button_size, "🔗"),
            button_size,
            button_spacing: BUTTON_SPACING,
        }
    }

    pub fn update_layout(&mut self) {
        use crate::ui::core::layout;
        
        let toolbar_rect = Rect::new(
            self.position.x,
            self.position.y,
            self.size.x,
            self.size.y,
        );
        
        // Calculate button widths for horizontal stack
        let button_widths = [
            self.button_size.x,
            self.button_size.x,
            self.button_size.x,
            self.button_size.x,
            self.button_size.x,
            self.button_size.x,
        ];
        
        // Use stack_horizontal to position buttons
        let button_rects = layout::stack_horizontal(
            &toolbar_rect,
            &button_widths,
            self.button_spacing,
            0.0,
        );
        
        if let Some(rect) = button_rects.get(0) {
            self.bold_button.position = rect.position();
        }
        if let Some(rect) = button_rects.get(1) {
            self.italic_button.position = rect.position();
        }
        if let Some(rect) = button_rects.get(2) {
            self.underline_button.position = rect.position();
        }
        if let Some(rect) = button_rects.get(3) {
            self.strikethrough_button.position = rect.position();
        }
        if let Some(rect) = button_rects.get(4) {
            self.code_button.position = rect.position();
        }
        if let Some(rect) = button_rects.get(5) {
            self.link_button.position = rect.position();
        }
    }

    pub fn hit_test(&self, pos: Vec2) -> Option<ToolbarButton> {
        if self.bold_button.contains(pos) {
            Some(ToolbarButton::Bold)
        } else if self.italic_button.contains(pos) {
            Some(ToolbarButton::Italic)
        } else if self.underline_button.contains(pos) {
            Some(ToolbarButton::Underline)
        } else if self.strikethrough_button.contains(pos) {
            Some(ToolbarButton::Strikethrough)
        } else if self.code_button.contains(pos) {
            Some(ToolbarButton::Code)
        } else if self.link_button.contains(pos) {
            Some(ToolbarButton::Link)
        } else {
            None
        }
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }

    pub fn on_hover(&mut self, pos: Vec2) {
        self.bold_button.on_hover(pos);
        self.italic_button.on_hover(pos);
        self.underline_button.on_hover(pos);
        self.strikethrough_button.on_hover(pos);
        self.code_button.on_hover(pos);
        self.link_button.on_hover(pos);
    }

    pub fn on_mouse_down(&mut self, pos: Vec2) {
        if self.bold_button.contains(pos) {
            self.bold_button.on_press();
        } else if self.italic_button.contains(pos) {
            self.italic_button.on_press();
        } else if self.underline_button.contains(pos) {
            self.underline_button.on_press();
        } else if self.strikethrough_button.contains(pos) {
            self.strikethrough_button.on_press();
        } else if self.code_button.contains(pos) {
            self.code_button.on_press();
        } else if self.link_button.contains(pos) {
            self.link_button.on_press();
        }
    }

    pub fn on_mouse_up(&mut self, pos: Vec2) {
        self.bold_button.on_release();
        self.italic_button.on_release();
        self.underline_button.on_release();
        self.strikethrough_button.on_release();
        self.code_button.on_release();
        self.link_button.on_release();
    }

    pub fn on_cancel(&mut self) {
        self.bold_button.on_cancel();
        self.italic_button.on_cancel();
        self.underline_button.on_cancel();
        self.strikethrough_button.on_cancel();
        self.code_button.on_cancel();
        self.link_button.on_cancel();
    }
}

