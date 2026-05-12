//! Application-managed paste board ("Slate") — scrollable list of stashed snippets.

#[derive(Clone, Debug)]
pub struct SlateEntry {
    pub id: String,
    pub preview: String,
}

pub struct SlateState {
    pub visible: bool,
    pub entries: Vec<SlateEntry>,
}

impl SlateState {
    pub fn new() -> Self {
        Self {
            visible: true,
            entries: Vec::new(),
        }
    }

    pub fn push_preview(&mut self, preview: String) -> String {
        let id = format!("slate_{}", uuid::Uuid::new_v4().simple());
        self.entries.push(SlateEntry {
            id: id.clone(),
            preview,
        });
        id
    }
}

impl Default for SlateState {
    fn default() -> Self {
        Self::new()
    }
}
