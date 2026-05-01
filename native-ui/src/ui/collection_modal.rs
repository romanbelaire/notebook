use glam::Vec2;
use std::collections::HashSet;

use crate::ui::{Button, ScrollView, TextInput};

use super::library_window::Paper;

pub struct CollectionModal {
    pub is_open: bool,
    pub collection_id: Option<i32>,
    pub collection_name: String,
    pub papers: Vec<Paper>,
    pub filtered_papers: Vec<Paper>,
    pub selected_papers: HashSet<i32>,
    pub position: Vec2,
    pub size: Vec2,
    pub search_input: TextInput,
    pub papers_list: ScrollView,
    pub close_button: Button,
    pub delete_button: Button,
    pub remove_from_collection_button: Button,
    pub delete_confirm: bool,
}

impl CollectionModal {
    pub fn new() -> Self {
        let size = Vec2::new(860.0, 620.0);
        let position = Vec2::new(960.0 - size.x / 2.0, 540.0 - size.y / 2.0);
        let search_input = TextInput::new(
            Vec2::new(position.x + 20.0, position.y + 60.0),
            Vec2::new(size.x - 40.0, 36.0),
        );
        let papers_list = ScrollView::new(
            Vec2::new(position.x + 20.0, position.y + 110.0),
            Vec2::new(size.x - 40.0, size.y - 170.0),
        );
        let close_button = Button::new(
            Vec2::new(position.x + size.x - 50.0, position.y + 15.0),
            Vec2::new(30.0, 30.0),
            "×",
        );
        let delete_button = Button::new(
            Vec2::new(position.x + 20.0, position.y + size.y - 48.0),
            Vec2::new(130.0, 32.0),
            "Delete",
        );
        let remove_from_collection_button = Button::new(
            Vec2::new(position.x + 165.0, position.y + size.y - 48.0),
            Vec2::new(220.0, 32.0),
            "Remove from collection",
        );
        Self {
            is_open: false,
            collection_id: None,
            collection_name: String::new(),
            papers: Vec::new(),
            filtered_papers: Vec::new(),
            selected_papers: HashSet::new(),
            position,
            size,
            search_input,
            papers_list,
            close_button,
            delete_button,
            remove_from_collection_button,
            delete_confirm: false,
        }
    }

    pub fn open(&mut self, id: i32, name: String, papers: Vec<Paper>) {
        self.is_open = true;
        self.collection_id = Some(id);
        self.collection_name = name;
        self.papers = papers;
        self.search_input.text.clear();
        self.selected_papers.clear();
        self.delete_confirm = false;
        self.update_filtered_papers();
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.collection_id = None;
        self.collection_name.clear();
        self.papers.clear();
        self.filtered_papers.clear();
        self.selected_papers.clear();
    }

    pub fn update_layout(&mut self, viewport_size: Vec2) {
        self.position = Vec2::new(viewport_size.x * 0.5 - self.size.x * 0.5, viewport_size.y * 0.5 - self.size.y * 0.5);
        self.close_button.position = Vec2::new(self.position.x + self.size.x - 50.0, self.position.y + 15.0);
        self.search_input.position = Vec2::new(self.position.x + 20.0, self.position.y + 60.0);
        self.search_input.size = Vec2::new(self.size.x - 40.0, 36.0);
        self.papers_list.position = Vec2::new(self.position.x + 20.0, self.position.y + 110.0);
        self.papers_list.size = Vec2::new(self.size.x - 40.0, self.size.y - 170.0);
        self.delete_button.position = Vec2::new(self.position.x + 20.0, self.position.y + self.size.y - 48.0);
        self.remove_from_collection_button.position = Vec2::new(self.position.x + 165.0, self.position.y + self.size.y - 48.0);
        self.papers_list.set_content_height((self.filtered_papers.len() as f32 * 48.0) + 10.0);
    }

    pub fn update_filtered_papers(&mut self) {
        let query = self.search_input.text.to_ascii_lowercase();
        if query.is_empty() {
            self.filtered_papers = self.papers.clone();
        } else {
            self.filtered_papers = self.papers.iter().filter(|paper| {
                paper.filename.to_ascii_lowercase().contains(&query)
                    || paper.title.clone().unwrap_or_default().to_ascii_lowercase().contains(&query)
                    || paper.authors.clone().unwrap_or_default().to_ascii_lowercase().contains(&query)
            }).cloned().collect();
        }
        self.papers_list.set_content_height((self.filtered_papers.len() as f32 * 48.0) + 10.0);
    }

    pub fn get_paper_at(&self, pos: Vec2) -> Option<usize> {
        if !self.papers_list.contains(pos) {
            return None;
        }
        let rel = pos - self.papers_list.position;
        let index = ((rel.y + self.papers_list.scroll_offset) / 48.0) as usize;
        if index < self.filtered_papers.len() {
            Some(index)
        } else {
            None
        }
    }
}
