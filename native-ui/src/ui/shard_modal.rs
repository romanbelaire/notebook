use glam::Vec2;
use crate::state::graph::GraphShard;
use crate::ui::core::{Rect, layout};
use crate::ui::{TextInput, Button};

/// Modal for editing a shard's messages (user + assistant) in a stacked layout.
/// Supports markdown in the editable text areas.
pub struct ShardModal {
    pub is_open: bool,
    pub shard_id: Option<String>,
    pub shard: Option<GraphShard>,
    pub position: Vec2,
    pub size: Vec2,
    pub user_input: TextInput,
    pub assistant_input: TextInput,
    pub close_button: Button,
    pub save_button: Button,
    /// Remove this shard from the graph (only relevant when opened from constellation).
    pub remove_from_graph_button: Button,
}

impl ShardModal {
    pub fn new() -> Self {
        let modal_width = 700.0;
        let modal_height = 500.0;
        let center_x = 960.0;
        let center_y = 540.0;
        let padding = 20.0;
        let msg_height = 150.0;

        Self {
            is_open: false,
            shard_id: None,
            shard: None,
            position: Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0),
            size: Vec2::new(modal_width, modal_height),
            user_input: TextInput::new(
                Vec2::new(center_x - modal_width / 2.0 + padding, center_y - modal_height / 2.0 + 60.0),
                Vec2::new(modal_width - padding * 2.0, msg_height),
            ),
            assistant_input: TextInput::new(
                Vec2::new(center_x - modal_width / 2.0 + padding, center_y - modal_height / 2.0 + 60.0 + msg_height + 16.0),
                Vec2::new(modal_width - padding * 2.0, msg_height),
            ),
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
            remove_from_graph_button: Button::new(
                Vec2::new(center_x - modal_width / 2.0 + padding, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(140.0, 30.0),
                "Remove from graph",
            ),
        }
    }

    pub fn open(&mut self, shard_id: String, shard: GraphShard) {
        self.shard_id = Some(shard_id);
        self.shard = Some(shard.clone());
        self.user_input.text = shard.user_content.clone().unwrap_or_default();
        self.assistant_input.text = shard.assistant_content.clone().unwrap_or_default();
        self.is_open = true;
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.shard_id = None;
        self.shard = None;
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        if !self.is_open {
            return false;
        }
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }

    pub fn update_layout(&mut self, viewport_size: Vec2) {
        const PADDING: f32 = 20.0;
        const HEADER_HEIGHT: f32 = 55.0; // padding + title row + gap to content
        const LABEL_HEIGHT: f32 = 18.0;
        const BUTTON_ROW_HEIGHT: f32 = 36.0;
        const SECTION_SPACING: f32 = 20.0;

        let modal_width = 700.0;
        let modal_height = 500.0;
        let center_x = viewport_size.x / 2.0;
        let center_y = viewport_size.y / 2.0;

        self.position = Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0);
        self.size = Vec2::new(modal_width, modal_height);

        let content_top = self.position.y + HEADER_HEIGHT;
        let content_height = modal_height - HEADER_HEIGHT - PADDING;
        let container = Rect::new(
            self.position.x + PADDING,
            content_top,
            modal_width - PADDING * 2.0,
            content_height,
        );
        let section_heights = [
            LABEL_HEIGHT,
            self.user_input.size.y,
            LABEL_HEIGHT,
            self.assistant_input.size.y,
            BUTTON_ROW_HEIGHT,
        ];
        let rects = layout::stack_vertical(&container, &section_heights, SECTION_SPACING, 0.0);

        self.user_input.position = rects[1].position();
        self.user_input.size = rects[1].size();

        self.assistant_input.position = rects[3].position();
        self.assistant_input.size = rects[3].size();

        self.close_button.position = Vec2::new(self.position.x + modal_width - 50.0, self.position.y + 20.0);

        let button_row = rects[4];
        self.save_button.position = Vec2::new(button_row.right() - self.save_button.size.x - 10.0, button_row.y);
        self.remove_from_graph_button.position = Vec2::new(button_row.x, button_row.y);
    }
}
