use serde::{Serialize, Deserialize};
use crate::stylus::block::Block;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub blocks: Vec<Block>,
    pub metadata: DocumentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub theme: String,
    pub plugins: Vec<String>,
}

impl Document {
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            blocks: vec![Block::new(crate::stylus::block::BlockType::Paragraph)],
            metadata: DocumentMetadata {
                title: "Untitled".to_string(),
                created_at: now,
                updated_at: now,
                theme: "default".to_string(),
                plugins: Vec::new(),
            },
        }
    }

    pub fn from_blocks(blocks: Vec<Block>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            blocks: if blocks.is_empty() {
                vec![Block::new(crate::stylus::block::BlockType::Paragraph)]
            } else {
                blocks
            },
            metadata: DocumentMetadata {
                title: "Untitled".to_string(),
                created_at: now,
                updated_at: now,
                theme: "default".to_string(),
                plugins: Vec::new(),
            },
        }
    }

    pub fn update_metadata(&mut self) {
        self.metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    pub fn set_title(&mut self, title: String) {
        self.metadata.title = title;
        self.update_metadata();
    }

    pub fn get_block(&self, id: &str) -> Option<&Block> {
        self.blocks.iter().find(|b| b.id == id)
    }

    pub fn get_block_mut(&mut self, id: &str) -> Option<&mut Block> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    pub fn get_block_index(&self, id: &str) -> Option<usize> {
        self.blocks.iter().position(|b| b.id == id)
    }

    pub fn insert_block(&mut self, block: Block, after_id: Option<&str>) -> usize {
        if let Some(after_id) = after_id {
            if let Some(index) = self.get_block_index(after_id) {
                self.blocks.insert(index + 1, block);
                index + 1
            } else {
                self.blocks.push(block);
                self.blocks.len() - 1
            }
        } else {
            self.blocks.push(block);
            self.blocks.len() - 1
        }
    }

    pub fn remove_block(&mut self, id: &str) -> Option<Block> {
        if let Some(index) = self.get_block_index(id) {
            Some(self.blocks.remove(index))
        } else {
            None
        }
    }

    pub fn move_block(&mut self, id: &str, new_index: usize) -> bool {
        if let Some(current_index) = self.get_block_index(id) {
            if new_index < self.blocks.len() {
                let block = self.blocks.remove(current_index);
                let adjusted_index = if new_index > current_index {
                    new_index - 1
                } else {
                    new_index
                };
                self.blocks.insert(adjusted_index, block);
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

