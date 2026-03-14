use glam::Vec2;
use crate::ui::{Button, TextInput, ScrollView};
use crate::api::models::Insight;

pub struct ChatInfoDialog {
    pub is_open: bool,
    pub conversation_id: Option<String>,
    pub position: Vec2,
    pub size: Vec2,
    pub title_input: TextInput,
    pub is_editing_title: bool,
    pub draft_title: String,
    pub close_button: Button,
    pub delete_button: Button,
    pub citations_list: ScrollView,
    pub insights_list: ScrollView,
    pub citation_mode: CitationMode,
    pub mode_toggle_button: Button,
    pub selected_insight_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CitationMode {
    All,
    Unique,
}

impl ChatInfoDialog {
    pub fn new() -> Self {
        let modal_width = 700.0;
        let modal_height = 600.0;
        let center_x = 960.0;
        let center_y = 540.0;
        
        Self {
            is_open: false,
            conversation_id: None,
            position: Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0),
            size: Vec2::new(modal_width, modal_height),
            title_input: TextInput::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 60.0),
                Vec2::new(modal_width - 100.0, 30.0),
            ),
            is_editing_title: false,
            draft_title: String::new(),
            close_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 50.0, center_y - modal_height / 2.0 + 20.0),
                Vec2::new(30.0, 30.0),
                "×",
            ),
            delete_button: Button::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(100.0, 30.0),
                "Delete",
            ),
            citations_list: ScrollView::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 150.0),
                Vec2::new(modal_width - 40.0, 200.0),
            ),
            insights_list: ScrollView::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 370.0),
                Vec2::new(modal_width - 40.0, 180.0),
            ),
            citation_mode: CitationMode::All,
            mode_toggle_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 120.0, center_y - modal_height / 2.0 + 120.0),
                Vec2::new(100.0, 25.0),
                "Show Unique",
            ),
            selected_insight_id: None,
        }
    }

    pub fn open(&mut self, conversation_id: String, title: String) {
        self.conversation_id = Some(conversation_id);
        self.draft_title = title;
        self.is_open = true;
        self.is_editing_title = false;
        self.selected_insight_id = None;
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.conversation_id = None;
        self.is_editing_title = false;
        self.selected_insight_id = None;
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        if !self.is_open {
            return false;
        }
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }

    pub fn update_layout(&mut self, viewport_size: Vec2) {
        let modal_width = 700.0;
        let modal_height = 600.0;
        let center_x = viewport_size.x / 2.0;
        let center_y = viewport_size.y / 2.0;
        
        self.position = Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0);
        self.size = Vec2::new(modal_width, modal_height);
        
        self.title_input.position = Vec2::new(self.position.x + 20.0, self.position.y + 60.0);
        self.title_input.size = Vec2::new(modal_width - 100.0, 30.0);
        
        self.close_button.position = Vec2::new(self.position.x + modal_width - 50.0, self.position.y + 20.0);
        self.delete_button.position = Vec2::new(self.position.x + 20.0, self.position.y + modal_height - 50.0);
        
        self.citations_list.position = Vec2::new(self.position.x + 20.0, self.position.y + 150.0);
        self.citations_list.size = Vec2::new(modal_width - 40.0, 200.0);
        
        self.insights_list.position = Vec2::new(self.position.x + 20.0, self.position.y + 370.0);
        self.insights_list.size = Vec2::new(modal_width - 40.0, 180.0);
        
        self.mode_toggle_button.position = Vec2::new(self.position.x + modal_width - 120.0, self.position.y + 120.0);
    }

    pub fn toggle_citation_mode(&mut self) {
        self.citation_mode = match self.citation_mode {
            CitationMode::All => CitationMode::Unique,
            CitationMode::Unique => CitationMode::All,
        };
    }

    pub fn get_citation_at(&self, pos: Vec2, citations: &[serde_json::Value]) -> Option<usize> {
        if !self.citations_list.contains(pos - self.position) {
            return None;
        }
        let rel_pos = pos - self.citations_list.position;
        let item_height = 25.0;
        let scroll_offset = self.citations_list.scroll_offset;
        let index = ((rel_pos.y + scroll_offset) / item_height) as usize;
        if index < citations.len() {
            Some(index)
        } else {
            None
        }
    }

    pub fn get_insight_at(&self, pos: Vec2, insights: &[Insight]) -> Option<usize> {
        if !self.insights_list.contains(pos - self.position) {
            return None;
        }
        let rel_pos = pos - self.insights_list.position;
        let item_height = 30.0;
        let scroll_offset = self.insights_list.scroll_offset;
        let index = ((rel_pos.y + scroll_offset) / item_height) as usize;
        if index < insights.len() {
            Some(index)
        } else {
            None
        }
    }
}

