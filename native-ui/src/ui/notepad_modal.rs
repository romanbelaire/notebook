use glam::Vec2;
use crate::ui::{Button, ScrollView};
use crate::ui::library_window::Paper;

pub struct NotepadModal {
    pub is_open: bool,
    pub position: Vec2,
    pub size: Vec2,
    pub close_button: Button,
    pub papers_list: ScrollView,
    pub import_button: Button,
    pub delete_button: Button,
    pub selected_paper_index: Option<usize>,
    pub papers: Vec<Paper>,
    pub filtered_papers: Vec<Paper>,
}

impl NotepadModal {
    pub fn new() -> Self {
        let modal_width = 600.0;
        let modal_height = 500.0;
        let center_x = 960.0;
        let center_y = 540.0;
        
        Self {
            is_open: false,
            position: Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0),
            size: Vec2::new(modal_width, modal_height),
            close_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 50.0, center_y - modal_height / 2.0 + 20.0),
                Vec2::new(30.0, 30.0),
                "×",
            ),
            papers_list: ScrollView::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 80.0),
                Vec2::new(modal_width - 40.0, modal_height - 180.0),
            ),
            import_button: Button::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(100.0, 30.0),
                "Import",
            ),
            delete_button: Button::new(
                Vec2::new(center_x - modal_width / 2.0 + 140.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(100.0, 30.0),
                "Delete",
            ),
            selected_paper_index: None,
            papers: Vec::new(),
            filtered_papers: Vec::new(),
        }
    }

    pub fn open(&mut self) {
        self.is_open = true;
        self.selected_paper_index = None;
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.selected_paper_index = None;
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        if !self.is_open {
            return false;
        }
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }

    pub fn update_layout(&mut self, viewport_size: Vec2) {
        let modal_width = 600.0;
        let modal_height = 500.0;
        let center_x = viewport_size.x / 2.0;
        let center_y = viewport_size.y / 2.0;
        
        self.position = Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0);
        self.size = Vec2::new(modal_width, modal_height);
        
        self.close_button.position = Vec2::new(self.position.x + modal_width - 50.0, self.position.y + 20.0);
        self.papers_list.position = Vec2::new(self.position.x + 20.0, self.position.y + 80.0);
        self.papers_list.size = Vec2::new(modal_width - 40.0, modal_height - 180.0);
        
        self.import_button.position = Vec2::new(self.position.x + 20.0, self.position.y + modal_height - 50.0);
        self.delete_button.position = Vec2::new(self.position.x + 140.0, self.position.y + modal_height - 50.0);
    }

    pub fn set_papers(&mut self, papers: Vec<Paper>) {
        self.papers = papers.clone();
        // Filter to only markdown/txt/docx files
        self.filtered_papers = papers.into_iter()
            .filter(|p| {
                let filename = p.filename.to_lowercase();
                filename.ends_with(".md") || 
                filename.ends_with(".markdown") || 
                filename.ends_with(".txt") || 
                filename.ends_with(".docx")
            })
            .collect();
        
        // Update scroll view content height
        let content_height = 10.0 + (self.filtered_papers.len() as f32 * 40.0) + 10.0;
        self.papers_list.set_content_height(content_height);
    }

    pub fn get_paper_at(&self, pos: Vec2) -> Option<usize> {
        if !self.papers_list.contains(pos - self.position) {
            return None;
        }
        let rel_pos = pos - self.papers_list.position;
        let item_height = 40.0;
        let scroll_offset = self.papers_list.scroll_offset;
        let index = ((rel_pos.y + scroll_offset) / item_height) as usize;
        if index < self.filtered_papers.len() {
            Some(index)
        } else {
            None
        }
    }
}

