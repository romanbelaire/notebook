use glam::Vec2;
use crate::stylus::StylusEditor;
use crate::ui::components::Renderable;
use crate::ui::text_editor::TextEditor;
use crate::ui::{TextInput, Button, NotepadModal, Toolbar};
use crate::ui::chat_window::MentionEntry;

pub struct NotepadWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub editor: StylusEditor,
    pub title_input: TextInput,
    pub new_button: Button,
    pub save_button: Button,
    pub open_button: Button,
    pub delete_button: Button,
    pub toolbar: Toolbar,
    pub notepad_modal: NotepadModal,
    pub document_title: String,
    /// `@` mention picker (papers, shards, graphs, notepad docs).
    pub mention_popup_open: bool,
    pub mention_selected_index: usize,
    pub mention_filter: String,
    pub mention_rows: Vec<MentionEntry>,
}

impl NotepadWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        use crate::ui::style;
        let padding = style::padding::LARGE;
        
        let mut editor = StylusEditor::new(
            Vec2::new(position.x + padding, position.y + padding),
            Vec2::new(size.x - padding * 2.0, size.y - padding * 2.0),
        );
        
        // Create a new document with an ID
        editor.create_new_document();

        // Title input
        let mut title_input = TextInput::new(
            Vec2::ZERO, // Will be positioned in update_layout
            Vec2::new(300.0, 40.0),
        );
        title_input.placeholder = "Untitled Note".to_string();
        title_input.text = "Untitled Note".to_string();

        // Title row actions (icons; labels are render keys, see notepad.rs render_button)
        const ICON_BTN: f32 = 32.0;
        let icon_sz = Vec2::new(ICON_BTN, ICON_BTN);
        let new_button = Button::new(Vec2::ZERO, icon_sz, "__plus");
        let save_button = Button::new(Vec2::ZERO, icon_sz, "__save");
        let open_button = Button::new(Vec2::ZERO, icon_sz, "__open");
        let delete_button = Button::new(Vec2::ZERO, icon_sz, "__delete");

        // Toolbar (with drop shadow)
        let toolbar = Toolbar::new(Vec2::ZERO, size.x - padding * 2.0)
            .with_shadow(crate::ui::style::elevation::LOW());

        // Modal
        let notepad_modal = NotepadModal::new();

        let mut window = Self {
            position,
            size,
            editor,
            title_input,
            new_button,
            save_button,
            open_button,
            delete_button,
            toolbar,
            notepad_modal,
            document_title: "Untitled Note".to_string(),
            mention_popup_open: false,
            mention_selected_index: 0,
            mention_filter: String::new(),
            mention_rows: Vec::new(),
        };

        window.update_layout();
        window
    }

    pub fn sync_mention_popup_from_editor(
        &mut self,
        papers: &[crate::api::models::ApiPaper],
        graph_state: &crate::state::GraphState,
        conversations: &[crate::state::chat::Conversation],
        notepad_documents: &[(String, String)],
    ) {
        if !self.mention_popup_open {
            return;
        }
        let Some(ref cursor) = self.editor.cursor else {
            self.mention_popup_open = false;
            self.mention_rows.clear();
            return;
        };
        let Some(block) = self.editor.document.get_block(&cursor.block_id) else {
            self.mention_popup_open = false;
            self.mention_rows.clear();
            return;
        };
        let Some(text) = block.content.get_text() else {
            self.mention_popup_open = false;
            self.mention_rows.clear();
            return;
        };
        let cur = cursor.position.min(text.len());
        let before: String = text.chars().take(cur).collect();
        if let Some(last_at) = before.rfind('@') {
            let after = &before[last_at.saturating_add(1)..];
            if after.contains(' ') || after.contains('\n') {
                self.mention_popup_open = false;
                self.mention_rows.clear();
            } else {
                self.mention_filter = after.to_string();
                self.mention_selected_index = 0;
                self.rebuild_mention_rows(papers, graph_state, conversations, notepad_documents);
            }
        } else {
            self.mention_popup_open = false;
            self.mention_rows.clear();
        }
    }

    fn rebuild_mention_rows(
        &mut self,
        papers: &[crate::api::models::ApiPaper],
        graph_state: &crate::state::GraphState,
        conversations: &[crate::state::chat::Conversation],
        notepad_documents: &[(String, String)],
    ) {
        self.mention_rows.clear();
        let fl = self.mention_filter.to_lowercase();
        let matches = |s: &str| fl.is_empty() || s.to_lowercase().contains(&fl);
        for p in papers {
            let label = p.title.as_deref().unwrap_or(p.filename.as_str());
            if matches(label) || matches(&p.filename) {
                self.mention_rows.push(MentionEntry::Paper(p.id));
            }
        }
        if let Some(gid) = graph_state.graph_id.as_ref() {
            for id in graph_state.nodes.keys() {
                if matches(id.as_str()) {
                    self.mention_rows.push(MentionEntry::Shard {
                        graph_id: gid.clone(),
                        shard_id: id.clone(),
                    });
                }
            }
        }
        let mut seen_graph: std::collections::HashSet<String> = std::collections::HashSet::new();
        for c in conversations {
            if let Some(ref gid) = c.graph_id {
                if seen_graph.insert(gid.clone()) && (matches(&c.title) || matches(gid.as_str())) {
                    self.mention_rows.push(MentionEntry::Graph {
                        graph_id: gid.clone(),
                    });
                }
            }
        }
        for (doc_id, title) in notepad_documents {
            if matches(title.as_str()) || matches(doc_id.as_str()) {
                self.mention_rows.push(MentionEntry::Notepad {
                    document_id: doc_id.clone(),
                    title: title.clone(),
                });
            }
        }
        if self.mention_selected_index >= self.mention_rows.len() && !self.mention_rows.is_empty() {
            self.mention_selected_index = self.mention_rows.len() - 1;
        }
        self.mention_rows.truncate(20);
    }

    pub fn mention_popup_rect(&self) -> Option<crate::ui::core::Rect> {
        if !self.mention_popup_open || self.mention_rows.is_empty() {
            return None;
        }
        let n = self.mention_rows.len().min(12);
        let row_h = 28.0;
        let h = n as f32 * row_h + 8.0;
        let ip = self.editor.position;
        let is = self.editor.size;
        Some(crate::ui::core::Rect::new(
            ip.x,
            ip.y - h - 4.0,
            is.x,
            h,
        ))
    }

    pub fn apply_mention_row_selection(&mut self, index: usize) {
        if index >= self.mention_rows.len() {
            return;
        }
        let replacement = match self.mention_rows[index].clone() {
            MentionEntry::Paper(id) => format!("@paper:{} ", id),
            MentionEntry::Shard {
                graph_id,
                shard_id,
            } => format!("@shard:{}:{} ", graph_id, shard_id),
            MentionEntry::Graph { graph_id } => format!("@graph:{} ", graph_id),
            MentionEntry::Notepad { document_id, .. } => format!("@notepad:{} ", document_id),
        };
        self.editor.replace_active_at_mention(&replacement);
        self.mention_popup_open = false;
        self.mention_rows.clear();
    }

    pub fn update_layout(&mut self) {
        use crate::ui::style;
        use crate::ui::core::{Rect, layout};
        use crate::ui::components::Renderable;
        
        let padding = style::padding::LARGE;
        const TITLE_ROW_HEIGHT: f32 = 40.0;
        const TOOLBAR_HEIGHT: f32 = 40.0;
        const ICON_BTN: f32 = 32.0;
        const SPACING: f32 = style::padding::SMALL;

        let window_rect = Rect::new(
            self.position.x,
            self.position.y,
            self.size.x,
            self.size.y,
        );
        let content_rect = window_rect.inset(padding);

        // vstack: title row (title left + icon hstack right), toolbar, editor
        let top_heights = [TITLE_ROW_HEIGHT, TOOLBAR_HEIGHT];
        let top_rects = layout::stack_vertical(&content_rect, &top_heights, SPACING, 0.0);

        let title_row_rect = top_rects[0];
        let actions_width = 4.0 * ICON_BTN + 3.0 * SPACING;
        let title_left_w = (title_row_rect.width - actions_width - SPACING).max(0.0);
        let title_rect = Rect::new(
            title_row_rect.x,
            title_row_rect.y,
            title_left_w,
            TITLE_ROW_HEIGHT,
        );
        self.title_input.update_layout(title_rect, None, None);

        let actions_x = title_row_rect.x + title_left_w + SPACING;
        let actions_row = Rect::new(actions_x, title_row_rect.y, actions_width, TITLE_ROW_HEIGHT);
        let icon_row_y = actions_row.y + (actions_row.height - ICON_BTN) * 0.5;
        let icon_hstack = Rect::new(actions_row.x, icon_row_y, actions_width, ICON_BTN);
        let button_widths = [ICON_BTN, ICON_BTN, ICON_BTN, ICON_BTN];
        let button_rects = layout::stack_horizontal(&icon_hstack, &button_widths, SPACING, 0.0);
        Renderable::update_layout(&mut self.new_button, button_rects[0], None, None);
        Renderable::update_layout(&mut self.save_button, button_rects[1], None, None);
        Renderable::update_layout(&mut self.open_button, button_rects[2], None, None);
        Renderable::update_layout(&mut self.delete_button, button_rects[3], None, None);

        let toolbar_rect = top_rects[1];
        Renderable::update_layout(&mut self.toolbar, toolbar_rect, None, None);

        let editor_top = toolbar_rect.bottom() + SPACING;
        let editor_height = (content_rect.bottom() - editor_top).max(0.0);
        let editor_rect = Rect::new(
            content_rect.x,
            editor_top,
            content_rect.width,
            editor_height,
        );
        self.editor.position = editor_rect.position();
        self.editor.size = editor_rect.size();
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }

    pub fn on_char_received(&mut self, ch: char) {
        // Handle slash commands first
        if ch == '/' && !self.editor.slash_command_active {
            self.editor.handle_slash_command(ch);
            return; // Don't insert the '/' character when starting slash command
        }
        
        if self.editor.slash_command_active {
            // Handle slash command input
            self.editor.handle_slash_command(ch);
            return; // Don't insert characters while in slash command mode
        }
        
        // Regular text input
        self.editor.insert_text(&ch.to_string());
    }

    pub fn on_keyboard(&mut self, event: &winit::event::KeyEvent) {
        use winit::event::ElementState;
        use winit::keyboard::{KeyCode, PhysicalKey};
        
        if event.state == ElementState::Pressed {
            if let PhysicalKey::Code(key_code) = event.physical_key {
                match key_code {
                    KeyCode::Backspace => {
                        self.editor.on_backspace();
                    }
                    KeyCode::Delete => {
                        self.editor.on_delete();
                    }
                    KeyCode::ArrowLeft => {
                        self.editor.move_cursor(crate::stylus::editor::CursorDirection::Left);
                    }
                    KeyCode::ArrowRight => {
                        self.editor.move_cursor(crate::stylus::editor::CursorDirection::Right);
                    }
                    KeyCode::ArrowUp => {
                        self.editor.move_cursor(crate::stylus::editor::CursorDirection::Up);
                    }
                    KeyCode::ArrowDown => {
                        self.editor.move_cursor(crate::stylus::editor::CursorDirection::Down);
                    }
                    KeyCode::Enter => {
                        // Create new paragraph block
                        let cursor_block_id = self.editor.cursor.as_ref().map(|c| c.block_id.clone());
                        if let Some(block_id) = cursor_block_id {
                            let new_block_id = self.editor.create_block(
                                crate::stylus::block::BlockType::Paragraph,
                                Some(&block_id),
                            );
                            self.editor.focus_block(&new_block_id);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn create_new_note(&mut self) {
        self.editor.create_new_document();
        self.document_title = "Untitled Note".to_string();
        self.title_input.text = "Untitled Note".to_string();
        self.title_input.cursor_position = self.title_input.text.chars().count();
    }

    pub fn save_note(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Update document title from input
        let title = if self.title_input.text.trim().is_empty() {
            "Untitled Note".to_string()
        } else {
            self.title_input.text.trim().to_string()
        };
        self.document_title = title.clone();
        self.editor.document.set_title(title);
        
        // Save document
        if let Some(ref doc_id) = self.editor.document_id {
            self.editor.save()?;
        } else {
            // Create new document ID if none exists
            let doc_id = self.editor.create_new_document();
            self.editor.save()?;
        }
        Ok(())
    }

    pub fn load_note(&mut self, document_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.editor.load_document(document_id)?;
        self.document_title = self.editor.document.metadata.title.clone();
        self.title_input.text = self.document_title.clone();
        self.title_input.cursor_position = self.title_input.text.chars().count();
        Ok(())
    }

    pub fn delete_note(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::DocumentPersistence;
        if let Some(ref doc_id) = self.editor.document_id {
            DocumentPersistence::delete_document(doc_id)?;
            // Create a new empty document
            self.create_new_note();
        }
        Ok(())
    }

    pub fn open_modal(&mut self) {
        self.notepad_modal.open();
        // Load document list
        self.refresh_modal_documents();
    }

    pub fn refresh_modal_documents(&mut self) {
        use crate::persistence::DocumentPersistence;
        use crate::ui::library_window::Paper;
        
        if let Ok(document_ids) = DocumentPersistence::list_documents() {
            let mut papers = Vec::new();
            for doc_id in document_ids {
                if let Ok(doc) = DocumentPersistence::load_document(&doc_id) {
                    papers.push(Paper {
                        id: 0, // Not used for documents - we use doc_id from filename
                        title: Some(doc.metadata.title.clone()),
                        filename: format!("{}.json", doc_id),
                        authors: None,
                        year: None,
                        exists: true,
                    });
                }
            }
            self.notepad_modal.set_papers(papers);
        }
    }

    pub fn hit_test(&mut self, pos: Vec2) -> NotepadHit {
        if self.mention_popup_open {
            if let Some(rect) = self.mention_popup_rect() {
                if rect.contains_point(pos) {
                    const ROW: f32 = 28.0;
                    const PAD: f32 = 4.0;
                    let inner_y = pos.y - rect.y - PAD;
                    if inner_y >= 0.0 {
                        let idx = (inner_y / ROW) as usize;
                        if idx < self.mention_rows.len() {
                            return NotepadHit::MentionItem(idx);
                        }
                    }
                    return NotepadHit::Background;
                }
            }
        }
        // Check modal first (if open)
        if self.notepad_modal.is_open {
            if self.notepad_modal.contains(pos) {
                if self.notepad_modal.close_button.contains(pos) {
                    return NotepadHit::ModalClose;
                }
                if self.notepad_modal.import_button.contains(pos) {
                    return NotepadHit::ModalImport;
                }
                if self.notepad_modal.delete_button.contains(pos) {
                    return NotepadHit::ModalDelete;
                }
                if let Some(index) = self.notepad_modal.get_paper_at(pos) {
                    return NotepadHit::ModalPaper(index);
                }
                return NotepadHit::Modal;
            }
        }

        // Check title input
        if self.title_input.contains(pos) {
            return NotepadHit::TitleInput;
        }

        // Check CRUD buttons
        if self.new_button.contains(pos) {
            return NotepadHit::NewButton;
        }
        if self.save_button.contains(pos) {
            return NotepadHit::SaveButton;
        }
        if self.open_button.contains(pos) {
            return NotepadHit::OpenButton;
        }
        if self.delete_button.contains(pos) {
            return NotepadHit::DeleteButton;
        }

        // Check toolbar
        if let Some(button) = self.toolbar.hit_test(pos) {
            return NotepadHit::ToolbarButton(button);
        }

        // Check editor
        if self.editor.position.x <= pos.x && pos.x <= self.editor.position.x + self.editor.size.x &&
           self.editor.position.y <= pos.y && pos.y <= self.editor.position.y + self.editor.size.y {
            return NotepadHit::Editor;
        }

        NotepadHit::Background
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotepadHit {
    MentionItem(usize),
    TitleInput,
    NewButton,
    SaveButton,
    OpenButton,
    DeleteButton,
    ToolbarButton(crate::ui::ToolbarButton),
    Editor,
    Modal,
    ModalClose,
    ModalImport,
    ModalDelete,
    ModalPaper(usize),
    Background,
}

