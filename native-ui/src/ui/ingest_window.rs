use glam::Vec2;
use crate::ui::{TextInput, Button};
use crate::ui::core::Rect;

/// Must match [`crate::gfx::components::data::render_data`] layout.
pub const INGEST_PADDING: f32 = 20.0;
pub const INGEST_SECTION_SPACING: f32 = 30.0;
pub const INGEST_TITLE_HEIGHT: f32 = 40.0;
pub const INGEST_LABEL_HEIGHT: f32 = 25.0;
pub const INGEST_VSTACK_SPACING: f32 = 10.0;
pub const INGEST_INPUT_HEIGHT: f32 = 40.0;
pub const INGEST_SUBMIT_BUTTON_WIDTH: f32 = 88.0;
pub const INGEST_INPUT_ROW_GAP: f32 = 10.0;
pub const INGEST_BUTTON_ROW_GAP: f32 = 10.0;
pub const INGEST_BIB_BUTTON_WIDTH: f32 = 120.0;
pub const INGEST_BROWSE_BUTTON_WIDTH: f32 = 80.0;
pub const INGEST_INGEST_BUTTON_WIDTH: f32 = 100.0;

pub struct IngestWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub pdf_dir_input: TextInput,
    pub submit_button: Button,
    pub bib_upload_button: Button,
    pub browse_button: Button,
    pub ingest_button: Button,
    pub view_failures_button: Button,
    pub status_text: String,
    /// Last completed import summary (ArXiv or directory ingest with stats when available).
    pub import_summary_line: String,
    pub show_view_failures_button: bool,
    pub failure_lines: Vec<String>,
    pub is_ingesting: bool,
    pub progress: f32,
}

impl IngestWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        let padding = INGEST_PADDING;
        let input_height = INGEST_INPUT_HEIGHT;

        let mut pdf_dir_input = TextInput::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(100.0, input_height),
        );
        pdf_dir_input.set_placeholder("Paste arXiv IDs/URLs (multiple supported)".to_string());

        let submit_button = Button::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(INGEST_SUBMIT_BUTTON_WIDTH, input_height),
            "Submit",
        );

        let bib_upload_button = Button::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(INGEST_BIB_BUTTON_WIDTH, input_height),
            "Upload .bib",
        );

        let browse_button = Button::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(INGEST_BROWSE_BUTTON_WIDTH, input_height),
            "Browse",
        );

        let ingest_button = Button::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(INGEST_INGEST_BUTTON_WIDTH, input_height),
            "Ingest",
        );

        let view_failures_button = Button::new(
            Vec2::new(position.x + padding, position.y + 200.0),
            Vec2::new(160.0, 36.0),
            "View failures",
        );

        Self {
            position,
            size,
            pdf_dir_input,
            submit_button,
            bib_upload_button,
            browse_button,
            ingest_button,
            view_failures_button,
            status_text: String::new(),
            import_summary_line: String::new(),
            show_view_failures_button: false,
            failure_lines: Vec::new(),
            is_ingesting: false,
            progress: 0.0,
        }
    }

    pub fn update_layout(&mut self) {
        let container_x = self.position.x + INGEST_PADDING;
        let container_y = self.position.y + INGEST_PADDING;
        let container_w = self.size.x - INGEST_PADDING * 2.0;

        let input_section_y = container_y + INGEST_TITLE_HEIGHT + INGEST_SECTION_SPACING;
        let input_row_y = input_section_y + INGEST_LABEL_HEIGHT + INGEST_VSTACK_SPACING;

        let input_w = container_w - INGEST_SUBMIT_BUTTON_WIDTH - INGEST_INPUT_ROW_GAP;

        let input_rect = Rect::new(container_x, input_row_y, input_w, INGEST_INPUT_HEIGHT);
        self.pdf_dir_input.position = input_rect.position();
        self.pdf_dir_input.size = input_rect.size();
        self.pdf_dir_input.rect = input_rect;

        self.submit_button.position = Vec2::new(
            container_x + input_w + INGEST_INPUT_ROW_GAP,
            input_row_y,
        );
        self.submit_button.size = Vec2::new(INGEST_SUBMIT_BUTTON_WIDTH, INGEST_INPUT_HEIGHT);

        let input_stack_height =
            INGEST_LABEL_HEIGHT + INGEST_VSTACK_SPACING + INGEST_INPUT_HEIGHT;
        let button_y = input_section_y + input_stack_height + INGEST_BUTTON_ROW_GAP;

        self.bib_upload_button.position = Vec2::new(container_x, button_y);
        self.bib_upload_button.size = Vec2::new(INGEST_BIB_BUTTON_WIDTH, INGEST_INPUT_HEIGHT);

        self.browse_button.position = Vec2::new(
            container_x + INGEST_BIB_BUTTON_WIDTH + 10.0,
            button_y,
        );
        self.browse_button.size = Vec2::new(INGEST_BROWSE_BUTTON_WIDTH, INGEST_INPUT_HEIGHT);

        self.ingest_button.position = Vec2::new(
            container_x + INGEST_BIB_BUTTON_WIDTH + 10.0 + INGEST_BROWSE_BUTTON_WIDTH + 20.0,
            button_y,
        );
        self.ingest_button.size = Vec2::new(INGEST_INGEST_BUTTON_WIDTH, INGEST_INPUT_HEIGHT);

        const SECTION_SPACING: f32 = 30.0;
        let status_section_y = button_y + INGEST_INPUT_HEIGHT + SECTION_SPACING;
        let status_block_height = if !self.import_summary_line.is_empty() {
            100.0
        } else if !self.status_text.is_empty() {
            60.0
        } else {
            0.0
        };
        let view_failures_y = status_section_y + status_block_height + 10.0;
        self.view_failures_button.position = Vec2::new(self.position.x + INGEST_PADDING, view_failures_y);
        self.view_failures_button.size = Vec2::new(160.0, 36.0);
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

    pub fn is_bib_upload_button_clicked(&self, pos: Vec2) -> bool {
        pos.x >= self.bib_upload_button.position.x
            && pos.x <= self.bib_upload_button.position.x + self.bib_upload_button.size.x
            && pos.y >= self.bib_upload_button.position.y
            && pos.y <= self.bib_upload_button.position.y + self.bib_upload_button.size.y
    }

    pub fn is_view_failures_button_clicked(&self, pos: Vec2) -> bool {
        if !self.show_view_failures_button {
            return false;
        }
        pos.x >= self.view_failures_button.position.x
            && pos.x <= self.view_failures_button.position.x + self.view_failures_button.size.x
            && pos.y >= self.view_failures_button.position.y
            && pos.y <= self.view_failures_button.position.y + self.view_failures_button.size.y
    }
}
