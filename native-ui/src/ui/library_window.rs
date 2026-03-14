use glam::Vec2;
use crate::ui::{ScrollView, TextInput, Button};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Paper {
    pub id: i32,
    pub filename: String,
    pub title: Option<String>,
    pub authors: Option<String>,
    pub year: Option<i32>,
}

#[derive(Clone, Debug)]
pub struct LibraryCollection {
    pub id: i32,
    pub name: String,
    pub paper_count: i32,
    pub papers: Vec<Paper>,
}

impl Default for LibraryCollection {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            paper_count: 0,
            papers: Vec::new(),
        }
    }
}

pub struct LibraryWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub papers_list: ScrollView,
    pub collections_list: ScrollView,
    pub search_input: TextInput,
    pub new_collection_button: Button,
    pub new_collection_input: TextInput,
    pub is_creating_collection: bool,
    pub selected_collection_id: Option<i32>,
    pub papers: Vec<Paper>,
    pub collections: Vec<LibraryCollection>,
    pub filtered_papers: Vec<Paper>,
    pub search_query: String,
    pub selected_papers: HashSet<i32>,  // Selected paper IDs
    pub delete_confirm: bool,  // Whether delete confirmation is active
    pub delete_button: Button,  // Delete button
}

impl LibraryWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        let left_panel_width = 300.0;
        let right_panel_width = size.x - left_panel_width;
        let padding = 10.0;
        let search_height = 40.0;

        let papers_list = ScrollView::new(
            Vec2::new(position.x + left_panel_width + padding, position.y + search_height + padding),
            Vec2::new(right_panel_width - padding * 2.0, size.y - search_height - padding * 2.0),
        );

        let collections_list = ScrollView::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(left_panel_width - padding * 2.0, size.y - padding * 2.0),
        );

        let search_input = TextInput::new(
            Vec2::new(position.x + left_panel_width + padding, position.y + padding),
            Vec2::new(right_panel_width - padding * 2.0, search_height - padding * 2.0),
        );

        let button_height = 35.0;
        let new_collection_button = Button::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(left_panel_width - padding * 2.0, button_height),
            "+ New Collection",
        );

        let new_collection_input = TextInput::new(
            Vec2::new(position.x + padding, position.y + padding + button_height + 5.0),
            Vec2::new(left_panel_width - padding * 2.0, 30.0),
        );

        let delete_button = Button::new(
            Vec2::new(position.x + left_panel_width + padding, position.y + padding + search_height + 5.0),
            Vec2::new(100.0, 30.0),
            "Delete",
        );

        Self {
            position,
            size,
            papers_list,
            collections_list,
            search_input,
            new_collection_button,
            new_collection_input,
            is_creating_collection: false,
            selected_collection_id: None,
            papers: Vec::new(),
            collections: Vec::new(),
            filtered_papers: Vec::new(),
            search_query: String::new(),
            selected_papers: HashSet::new(),
            delete_confirm: false,
            delete_button,
        }
    }

    pub fn update_layout(&mut self) {
        use crate::ui::style;
        use crate::ui::core::{Rect, layout};
        
        let left_panel_width = 300.0;
        let padding = style::padding::SMALL;
        let search_height = 40.0;
        let button_spacing = 5.0; // Small spacing between search and delete button

        // Create window rect
        let window_rect = Rect::new(
            self.position.x,
            self.position.y,
            self.size.x,
            self.size.y,
        );
        let content_rect = window_rect.inset(padding);
        
        // Split into left and right panels using stack_horizontal
        let panel_widths = [left_panel_width, content_rect.width - left_panel_width];
        let panel_rects = layout::stack_horizontal(&content_rect, &panel_widths, 0.0, 0.0);
        
        // Left panel: collections list
        if let Some(left_panel) = panel_rects.get(0) {
            self.collections_list.position = left_panel.position();
            self.collections_list.size = left_panel.size();
        }
        
        // Right panel: search input + papers list + delete button
        if let Some(right_panel) = panel_rects.get(1) {
            // Use stack_vertical for search + papers list
            let papers_list_height = right_panel.height - search_height - button_spacing;
            let right_panel_heights = [search_height, papers_list_height];
            let right_panel_rects = layout::stack_vertical(right_panel, &right_panel_heights, button_spacing, 0.0);
            
            // Search input (top)
            if let Some(search_rect) = right_panel_rects.get(0) {
                let search_content_rect = search_rect.inset(padding);
                self.search_input.position = search_content_rect.position();
                self.search_input.size = search_content_rect.size();
            }
            
            // Papers list (below search)
            if let Some(papers_rect) = right_panel_rects.get(1) {
                let papers_content_rect = papers_rect.inset(padding);
                self.papers_list.position = papers_content_rect.position();
                self.papers_list.size = papers_content_rect.size();
            }
            
            // Delete button (below search with small spacing)
            if let Some(search_rect) = right_panel_rects.get(0) {
                let delete_button_y = search_rect.bottom() + button_spacing;
                self.delete_button.position = Vec2::new(
                    right_panel.x + padding,
                    delete_button_y,
                );
            }
        }

        // Update scroll view content heights
        let collections_height = 40.0 + (self.collections.len() as f32 * 35.0);
        self.collections_list.set_content_height(collections_height);
        
        let papers_height = 10.0 + (self.filtered_papers.len() as f32 * 50.0) + 10.0;
        self.papers_list.set_content_height(papers_height);
    }

    pub fn set_papers(&mut self, papers: Vec<Paper>) {
        self.papers = papers;
        self.update_filtered_papers();
    }

    pub fn set_collections(&mut self, collections: Vec<LibraryCollection>) {
        self.collections = collections;
    }

    pub fn update_search(&mut self) {
        self.search_query = self.search_input.text.clone();
        self.update_filtered_papers();
    }

    fn update_filtered_papers(&mut self) {
        if self.search_query.is_empty() {
            if let Some(collection_id) = self.selected_collection_id {
                // Show papers from selected collection
                if let Some(collection) = self.collections.iter().find(|c| c.id == collection_id) {
                    self.filtered_papers = collection.papers.clone();
                } else {
                    self.filtered_papers = Vec::new();
                }
            } else {
                // Show all papers
                self.filtered_papers = self.papers.clone();
            }
        } else {
            // Filter by search query
            let query_lower = self.search_query.to_lowercase();
            let base_papers: Vec<Paper> = if let Some(collection_id) = self.selected_collection_id {
                if let Some(collection) = self.collections.iter().find(|c| c.id == collection_id) {
                    collection.papers.clone()
                } else {
                    Vec::new()
                }
            } else {
                self.papers.clone()
            };

            self.filtered_papers = base_papers.iter()
                .filter(|paper| {
                    paper.filename.to_lowercase().contains(&query_lower) ||
                    paper.title.as_ref().map(|t| t.to_lowercase().contains(&query_lower)).unwrap_or(false) ||
                    paper.authors.as_ref().map(|a| a.to_lowercase().contains(&query_lower)).unwrap_or(false)
                })
                .cloned()
                .collect();
        }
    }

    pub fn select_collection(&mut self, collection_id: Option<i32>) {
        self.selected_collection_id = collection_id;
        self.update_filtered_papers();
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }

    pub fn is_new_collection_button_clicked(&self, pos: Vec2) -> bool {
        pos.x >= self.new_collection_button.position.x
            && pos.x <= self.new_collection_button.position.x + self.new_collection_button.size.x
            && pos.y >= self.new_collection_button.position.y
            && pos.y <= self.new_collection_button.position.y + self.new_collection_button.size.y
    }

    pub fn get_paper_at(&self, pos: Vec2) -> Option<usize> {
        let rel_pos = pos - self.papers_list.position;
        if rel_pos.x >= 0.0 && rel_pos.x <= self.papers_list.size.x &&
           rel_pos.y >= 0.0 && rel_pos.y <= self.papers_list.size.y {
            let scroll_adjusted_y = rel_pos.y + self.papers_list.scroll_offset;
            let item_index = ((scroll_adjusted_y - 10.0) / 50.0) as usize;
            if item_index < self.filtered_papers.len() {
                return Some(item_index);
            }
        }
        None
    }

    pub fn get_collection_at(&self, pos: Vec2) -> Option<usize> {
        let rel_pos = pos - self.collections_list.position;
        if rel_pos.x >= 0.0 && rel_pos.x <= self.collections_list.size.x &&
           rel_pos.y >= 0.0 && rel_pos.y <= self.collections_list.size.y {
            let scroll_adjusted_y = rel_pos.y + self.collections_list.scroll_offset;
            // Check "All papers" option first (at y = 10-40)
            if scroll_adjusted_y >= 10.0 && scroll_adjusted_y < 40.0 {
                return Some(usize::MAX);
            }
            // Then check collections (starting at y = 40)
            if scroll_adjusted_y >= 40.0 {
                let item_index = ((scroll_adjusted_y - 40.0) / 35.0) as usize;
                if item_index < self.collections.len() {
                    return Some(item_index);
                }
            }
        }
        None
    }

    pub fn toggle_paper_selection(&mut self, paper_id: i32) {
        if self.selected_papers.contains(&paper_id) {
            self.selected_papers.remove(&paper_id);
        } else {
            self.selected_papers.insert(paper_id);
        }
    }

    pub fn is_paper_selected(&self, paper_id: i32) -> bool {
        self.selected_papers.contains(&paper_id)
    }

    pub fn get_selected_paper_ids(&self) -> Vec<i32> {
        self.selected_papers.iter().cloned().collect()
    }

    pub fn clear_selection(&mut self) {
        self.selected_papers.clear();
    }

    pub fn get_checkbox_position(&self, paper_idx: usize) -> Vec2 {
        let rel_y = 10.0 + (paper_idx as f32 * 50.0);
        Vec2::new(
            self.papers_list.position.x + 10.0,
            self.papers_list.position.y + rel_y - self.papers_list.scroll_offset,
        )
    }
}

