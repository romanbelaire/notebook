use crate::ui::tab_bar::Tab;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIState {
    pub active_tab: Tab,
    pub sidebar_open: bool,
    pub focus_new_collection: bool,
}

impl UIState {
    pub fn new() -> Self {
        Self {
            active_tab: Tab::Chat,
            sidebar_open: true,
            focus_new_collection: false,
        }
    }

    pub fn set_active_tab(&mut self, tab: Tab) {
        self.active_tab = tab;
    }

    pub fn toggle_sidebar(&mut self) {
        self.sidebar_open = !self.sidebar_open;
    }

    pub fn set_sidebar_open(&mut self, open: bool) {
        self.sidebar_open = open;
    }

    pub fn set_focus_new_collection(&mut self, focus: bool) {
        self.focus_new_collection = focus;
    }
}

impl Default for UIState {
    fn default() -> Self {
        Self::new()
    }
}

