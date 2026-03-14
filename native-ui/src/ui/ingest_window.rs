use glam::Vec2;
use crate::ui::{TextInput, Button};

pub struct IngestWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub pdf_dir_input: TextInput,
    pub browse_button: Button,
    pub ingest_button: Button,
    pub status_text: String,
    pub is_ingesting: bool,
    pub progress: f32,
}

impl IngestWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        let padding = 20.0;
        let input_height = 40.0;
        let button_height = 40.0;
        let browse_button_width = 80.0;
        let ingest_button_width = 100.0;
        
        let pdf_dir_input = TextInput::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(size.x - padding * 3.0 - browse_button_width - ingest_button_width, input_height),
        );

        let browse_button = Button::new(
            Vec2::new(
                position.x + padding * 2.0 + pdf_dir_input.size.x,
                position.y + padding
            ),
            Vec2::new(browse_button_width, button_height),
            "Browse",
        );

        let ingest_button = Button::new(
            Vec2::new(
                position.x + size.x - padding - ingest_button_width,
                position.y + padding
            ),
            Vec2::new(ingest_button_width, button_height),
            "Ingest",
        );

        Self {
            position,
            size,
            pdf_dir_input,
            browse_button,
            ingest_button,
            status_text: String::new(),
            is_ingesting: false,
            progress: 0.0,
        }
    }

    pub fn update_layout(&mut self) {
        use crate::ui::core::{layout, Rect};
        let padding = 20.0;
        let input_height = 40.0;
        let button_height = 40.0;
        let browse_button_width = 80.0;
        let ingest_button_width = 100.0;

        // Use horizontal stack to position input and buttons
        let input_container = Rect::new(
            self.position.x + padding,
            self.position.y + padding,
            self.size.x - padding * 2.0,
            button_height,
        );
        
        let input_width = self.size.x - padding * 3.0 - browse_button_width - ingest_button_width;
        let button_rects = layout::stack_horizontal(
            &input_container,
            &[input_width, browse_button_width, ingest_button_width],
            padding,
            0.0,
        );

        self.pdf_dir_input.position = button_rects[0].position();
        self.pdf_dir_input.size = button_rects[0].size();

        self.browse_button.position = button_rects[1].position();
        self.browse_button.size = Vec2::new(browse_button_width, button_height);

        self.ingest_button.position = button_rects[2].position();
        self.ingest_button.size = Vec2::new(ingest_button_width, button_height);
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }

    pub fn is_browse_button_clicked(&self, pos: Vec2) -> bool {
        pos.x >= self.browse_button.position.x
            && pos.x <= self.browse_button.position.x + self.browse_button.size.x
            && pos.y >= self.browse_button.position.y
            && pos.y <= self.browse_button.position.y + self.browse_button.size.y
    }
}

