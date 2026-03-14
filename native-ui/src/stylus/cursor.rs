use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cursor {
    pub block_id: String,
    pub position: usize,
}

impl Cursor {
    pub fn new(block_id: String, position: usize) -> Self {
        Self { block_id, position }
    }

    pub fn at_start(block_id: String) -> Self {
        Self {
            block_id,
            position: 0,
        }
    }

    pub fn at_end(block_id: String, text_len: usize) -> Self {
        Self {
            block_id,
            position: text_len,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Selection {
    pub start: Cursor,
    pub end: Cursor,
}

impl Selection {
    pub fn new(start: Cursor, end: Cursor) -> Self {
        Self { start, end }
    }

    pub fn is_collapsed(&self) -> bool {
        self.start.block_id == self.end.block_id && self.start.position == self.end.position
    }

    pub fn normalize(&mut self) {
        // Ensure start comes before end
        if self.start.block_id > self.end.block_id {
            std::mem::swap(&mut self.start, &mut self.end);
        } else if self.start.block_id == self.end.block_id && self.start.position > self.end.position {
            std::mem::swap(&mut self.start, &mut self.end);
        }
    }

    pub fn contains_block(&self, block_id: &str) -> bool {
        if self.start.block_id == self.end.block_id {
            block_id == self.start.block_id.as_str()
        } else {
            block_id >= self.start.block_id.as_str() && block_id <= self.end.block_id.as_str()
        }
    }
}
