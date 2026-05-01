use glam::Vec2;
use crate::ui::Button;

pub struct IngestImportFailuresModal {
    pub is_open: bool,
    pub lines: Vec<String>,
    pub position: Vec2,
    pub size: Vec2,
    pub close_button: Button,
}

impl IngestImportFailuresModal {
    pub fn new() -> Self {
        let modal_width = 560.0;
        let modal_height = 420.0;
        Self {
            is_open: false,
            lines: Vec::new(),
            position: Vec2::ZERO,
            size: Vec2::new(modal_width, modal_height),
            close_button: Button::new(Vec2::ZERO, Vec2::new(100.0, 36.0), "Close"),
        }
    }

    pub fn open(&mut self, lines: Vec<String>) {
        self.lines = lines;
        self.is_open = true;
    }

    pub fn close(&mut self) {
        self.is_open = false;
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        if !self.is_open {
            return false;
        }
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }

    pub fn update_layout(&mut self, viewport: Vec2) {
        let modal_width = 560.0;
        let modal_height = 420.0;
        let center_x = viewport.x / 2.0;
        let center_y = viewport.y / 2.0;
        self.position = Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0);
        self.size = Vec2::new(modal_width, modal_height);
        self.close_button.position = Vec2::new(
            self.position.x + modal_width / 2.0 - 50.0,
            self.position.y + modal_height - 52.0,
        );
        self.close_button.size = Vec2::new(100.0, 36.0);
    }
}
