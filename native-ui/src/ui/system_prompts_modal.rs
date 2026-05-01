use glam::Vec2;
use uuid::Uuid;

use crate::state::SystemPromptEntry;
use crate::ui::{Button, TextInput};

/// Manage named system prompts (persisted in settings).
pub struct SystemPromptsModal {
    pub is_open: bool,
    pub position: Vec2,
    pub size: Vec2,
    pub prompts: Vec<SystemPromptEntry>,
    pub name_input: TextInput,
    pub content_input: TextInput,
    pub close_button: Button,
    pub save_button: Button,
    pub delete_buttons: Vec<Button>,
}

impl SystemPromptsModal {
    pub fn new() -> Self {
        let modal_width = 720.0;
        let modal_height = 560.0;
        let center_x = 960.0;
        let center_y = 540.0;
        Self {
            is_open: false,
            position: Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0),
            size: Vec2::new(modal_width, modal_height),
            prompts: Vec::new(),
            name_input: TextInput::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 110.0),
                Vec2::new(modal_width - 40.0, 36.0),
            ),
            content_input: TextInput::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 180.0),
                Vec2::new(modal_width - 40.0, 280.0),
            ),
            close_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 50.0, center_y - modal_height / 2.0 + 20.0),
                Vec2::new(30.0, 30.0),
                "×",
            ),
            save_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 140.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(120.0, 36.0),
                "Add prompt",
            ),
            delete_buttons: Vec::new(),
        }
    }

    pub fn open(&mut self, prompts: Vec<SystemPromptEntry>) {
        self.prompts = prompts;
        self.name_input.text.clear();
        self.content_input.text.clear();
        self.is_open = true;
        self.rebuild_delete_buttons();
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.delete_buttons.clear();
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        if !self.is_open {
            return false;
        }
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }

    pub fn rebuild_delete_buttons(&mut self) {
        self.delete_buttons.clear();
        let row_h = 32.0;
        let list_x = self.position.x + 20.0;
        let list_y = self.position.y + 70.0;
        let list_w = self.size.x - 40.0;
        for i in 0..self.prompts.len() {
            let y = list_y + i as f32 * row_h;
            let btn = Button::new(Vec2::new(list_x + list_w - 90.0, y), Vec2::new(70.0, 26.0), "Delete");
            self.delete_buttons.push(btn);
        }
    }

    pub fn add_from_inputs(&mut self) {
        let name = self.name_input.text.trim().to_string();
        let content = self.content_input.text.trim().to_string();
        if name.is_empty() || content.is_empty() {
            return;
        }
        self.prompts.push(SystemPromptEntry {
            id: Uuid::new_v4().to_string(),
            name,
            content,
        });
        self.name_input.text.clear();
        self.content_input.text.clear();
        self.rebuild_delete_buttons();
    }

    pub fn remove_at(&mut self, index: usize) {
        if index < self.prompts.len() {
            self.prompts.remove(index);
            self.rebuild_delete_buttons();
        }
    }

    pub fn update_layout(&mut self, viewport_size: Vec2) {
        let modal_width = 720.0;
        let modal_height = 560.0;
        let center_x = viewport_size.x / 2.0;
        let center_y = viewport_size.y / 2.0;
        self.position = Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0);
        self.size = Vec2::new(modal_width, modal_height);

        self.name_input.position = Vec2::new(self.position.x + 20.0, self.position.y + 110.0);
        self.name_input.size = Vec2::new(modal_width - 40.0, 36.0);

        self.content_input.position = Vec2::new(self.position.x + 20.0, self.position.y + 180.0);
        self.content_input.size = Vec2::new(modal_width - 40.0, 280.0);

        self.close_button.position = Vec2::new(self.position.x + modal_width - 50.0, self.position.y + 20.0);
        self.save_button.position = Vec2::new(self.position.x + modal_width - 140.0, self.position.y + modal_height - 50.0);

        self.rebuild_delete_buttons();
    }
}
