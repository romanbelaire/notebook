use glam::Vec2;
use crate::api::models::Insight;
use crate::ui::TextInput;
use crate::ui::Button;

pub struct InsightModal {
    pub is_open: bool,
    pub insight: Option<Insight>,
    pub position: Vec2,
    pub size: Vec2,
    pub title_input: TextInput,
    pub text_input: TextInput,
    pub is_editing_title: bool,
    pub is_editing_text: bool,
    pub draft_title: String,
    pub draft_text: String,
    pub close_button: Button,
    pub save_button: Button,
    pub delete_button: Button,
    pub pdf_peek: Option<PdfPeek>,
}

pub struct PdfPeek {
    pub source: String,
    pub page: Option<u32>,
    pub position: Vec2,
}

impl InsightModal {
    pub fn new() -> Self {
        let modal_width = 800.0;
        let modal_height = 600.0;
        let center_x = 960.0; // Will be centered in renderer
        let center_y = 540.0;
        
        Self {
            is_open: false,
            insight: None,
            position: Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0),
            size: Vec2::new(modal_width, modal_height),
            title_input: TextInput::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 60.0),
                Vec2::new(modal_width - 40.0, 30.0),
            ),
            text_input: TextInput::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 120.0),
                Vec2::new(modal_width - 40.0, 300.0),
            ),
            is_editing_title: false,
            is_editing_text: false,
            draft_title: String::new(),
            draft_text: String::new(),
            close_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 50.0, center_y - modal_height / 2.0 + 20.0),
                Vec2::new(30.0, 30.0),
                "×",
            ),
            save_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 150.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(80.0, 30.0),
                "Save",
            ),
            delete_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 250.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(100.0, 30.0),
                "Delete",
            ),
            pdf_peek: None,
        }
    }

    pub fn open(&mut self, insight: Insight) {
        self.insight = Some(insight.clone());
        self.draft_title = insight.title.clone();
        self.draft_text = insight.text.clone();
        self.is_open = true;
        self.is_editing_title = false;
        self.is_editing_text = false;
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.insight = None;
        self.pdf_peek = None;
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        if !self.is_open {
            return false;
        }
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }

    pub fn update_layout(&mut self, viewport_size: Vec2) {
        let modal_width = 800.0;
        let modal_height = 600.0;
        let center_x = viewport_size.x / 2.0;
        let center_y = viewport_size.y / 2.0;
        
        self.position = Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0);
        self.size = Vec2::new(modal_width, modal_height);
        
        self.title_input.position = Vec2::new(self.position.x + 20.0, self.position.y + 60.0);
        self.title_input.size = Vec2::new(modal_width - 40.0, 30.0);
        
        self.text_input.position = Vec2::new(self.position.x + 20.0, self.position.y + 120.0);
        self.text_input.size = Vec2::new(modal_width - 40.0, 300.0);
        
        self.close_button.position = Vec2::new(self.position.x + modal_width - 50.0, self.position.y + 20.0);
        self.save_button.position = Vec2::new(self.position.x + modal_width - 150.0, self.position.y + modal_height - 50.0);
        self.delete_button.position = Vec2::new(self.position.x + modal_width - 250.0, self.position.y + modal_height - 50.0);
    }
}

