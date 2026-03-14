use glam::Vec2;
use crate::ui::{TextInput, Button, ScrollView};

#[derive(Clone, Debug)]
pub enum SearchResult {
    Conversation {
        id: String,
        title: String,
        preview: String,
    },
    Insight {
        id: String,
        title: String,
        preview: String,
    },
    Paper {
        id: i32,
        filename: String,
        title: Option<String>,
        preview: String,
    },
}

pub struct GlobalSearchModal {
    pub is_open: bool,
    pub position: Vec2,
    pub size: Vec2,
    pub search_input: TextInput,
    pub close_button: Button,
    pub results_list: ScrollView,
    pub results: Vec<SearchResult>,
    pub query: String,
}

impl GlobalSearchModal {
    pub fn new() -> Self {
        let modal_width = 800.0;
        let modal_height = 600.0;
        let center_x = 960.0;
        let center_y = 540.0;
        
        Self {
            is_open: false,
            position: Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0),
            size: Vec2::new(modal_width, modal_height),
            search_input: TextInput::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 20.0),
                Vec2::new(modal_width - 80.0, 40.0),
            ),
            close_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 50.0, center_y - modal_height / 2.0 + 20.0),
                Vec2::new(30.0, 30.0),
                "×",
            ),
            results_list: ScrollView::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y - modal_height / 2.0 + 80.0),
                Vec2::new(modal_width - 40.0, modal_height - 120.0),
            ),
            results: Vec::new(),
            query: String::new(),
        }
    }

    pub fn open(&mut self) {
        self.is_open = true;
        self.search_input.on_focus();
        self.query.clear();
        self.results.clear();
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.search_input.on_blur();
        self.query.clear();
        self.results.clear();
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
        
        self.search_input.position = Vec2::new(self.position.x + 20.0, self.position.y + 20.0);
        self.search_input.size = Vec2::new(modal_width - 80.0, 40.0);
        
        self.close_button.position = Vec2::new(self.position.x + modal_width - 50.0, self.position.y + 20.0);
        
        self.results_list.position = Vec2::new(self.position.x + 20.0, self.position.y + 80.0);
        self.results_list.size = Vec2::new(modal_width - 40.0, modal_height - 120.0);
        
        // Update scroll view content height based on results
        let result_height = self.results.len() as f32 * 60.0;
        self.results_list.set_content_height(result_height.max(100.0));
    }

    pub fn search(&mut self, query: &str, conversations: &[crate::state::chat::Conversation], insights: &[crate::api::models::Insight], papers: &[crate::api::models::ApiPaper]) {
        self.query = query.to_lowercase();
        self.results.clear();
        
        if self.query.is_empty() {
            return;
        }
        
        // Search conversations
        for conv in conversations {
            if conv.title.to_lowercase().contains(&self.query) {
                let preview = if conv.shards.is_empty() {
                    "No messages".to_string()
                } else {
                    conv.shards[0].text.chars().take(100).collect()
                };
                self.results.push(SearchResult::Conversation {
                    id: conv.id.clone(),
                    title: conv.title.clone(),
                    preview,
                });
            }
            
            // Search in shards
            for shard in &conv.shards {
                if shard.text.to_lowercase().contains(&self.query) {
                    let preview = shard.text.chars().take(100).collect();
                    self.results.push(SearchResult::Conversation {
                        id: conv.id.clone(),
                        title: format!("{} (message)", conv.title),
                        preview,
                    });
                    break; // Only add once per conversation
                }
            }
        }
        
        // Search insights
        for insight in insights {
            if insight.title.to_lowercase().contains(&self.query) || insight.text.to_lowercase().contains(&self.query) {
                let preview = insight.text.chars().take(100).collect();
                self.results.push(SearchResult::Insight {
                    id: insight.id.clone(),
                    title: insight.title.clone(),
                    preview,
                });
            }
        }
        
        // Search papers
        for paper in papers {
            let searchable_text = format!(
                "{} {} {}",
                paper.filename.to_lowercase(),
                paper.title.as_ref().map(|t| t.to_lowercase()).unwrap_or_default(),
                paper.authors.as_ref().map(|a| a.to_lowercase()).unwrap_or_default(),
            );
            
            if searchable_text.contains(&self.query) {
                let preview = paper.title.as_ref().unwrap_or(&paper.filename).clone();
                self.results.push(SearchResult::Paper {
                    id: paper.id,
                    filename: paper.filename.clone(),
                    title: paper.title.clone(),
                    preview,
                });
            }
        }
    }

    pub fn get_result_at(&self, pos: Vec2) -> Option<usize> {
        let rel_pos = pos - self.results_list.position;
        if rel_pos.x >= 0.0 && rel_pos.x <= self.results_list.size.x &&
           rel_pos.y >= 0.0 && rel_pos.y <= self.results_list.size.y {
            let scroll_adjusted_y = rel_pos.y + self.results_list.scroll_offset;
            let item_index = (scroll_adjusted_y / 60.0) as usize;
            if item_index < self.results.len() {
                return Some(item_index);
            }
        }
        None
    }
}

