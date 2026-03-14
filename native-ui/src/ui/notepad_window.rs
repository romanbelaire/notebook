use glam::Vec2;
use crate::stylus::StylusEditor;
use crate::ui::components::Renderable;
use crate::ui::text_editor::TextEditor;
use crate::ui::{TextInput, Button, NotepadModal, Toolbar};

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

        // CRUD buttons
        const BUTTON_HEIGHT: f32 = 32.0;
        const BUTTON_WIDTH: f32 = 80.0;
        let new_button = Button::new(Vec2::ZERO, Vec2::new(BUTTON_WIDTH, BUTTON_HEIGHT), "New");
        let save_button = Button::new(Vec2::ZERO, Vec2::new(BUTTON_WIDTH, BUTTON_HEIGHT), "Save");
        let open_button = Button::new(Vec2::ZERO, Vec2::new(BUTTON_WIDTH, BUTTON_HEIGHT), "Open");
        let delete_button = Button::new(Vec2::ZERO, Vec2::new(BUTTON_WIDTH, BUTTON_HEIGHT), "Delete");

        // Toolbar
        let toolbar = Toolbar::new(Vec2::ZERO, size.x - padding * 2.0);

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
        };

        window.update_layout();
        window
    }

    pub fn update_layout(&mut self) {
        use crate::ui::style;
        use crate::ui::core::{Rect, layout};
        
        let padding = style::padding::LARGE;
        const TITLE_HEIGHT: f32 = 40.0;
        const BUTTON_ROW_HEIGHT: f32 = 36.0;
        const TOOLBAR_HEIGHT: f32 = 36.0;
        const SPACING: f32 = 8.0;

        // Create window rect with padding
        let window_rect = Rect::new(
            self.position.x,
            self.position.y,
            self.size.x,
            self.size.y,
        );
        let content_rect = window_rect.inset(padding);
        
        let mut y_offset = content_rect.y;
        
        // Title input area
        let title_rect = Rect::new(
            content_rect.x,
            y_offset,
            content_rect.width * 0.6, // Title takes 60% of width
            TITLE_HEIGHT,
        );
        self.title_input.update_layout(title_rect, None, None);
        y_offset += TITLE_HEIGHT + SPACING;
        
        // Button row
        let button_row_rect = Rect::new(
            content_rect.x,
            y_offset,
            content_rect.width,
            BUTTON_ROW_HEIGHT,
        );
        let button_widths = [80.0, 80.0, 80.0, 80.0];
        let button_rects = layout::stack_horizontal(&button_row_rect, &button_widths, SPACING, 0.0);
        if let Some(rect) = button_rects.get(0) {
            self.new_button.position = rect.position();
        }
        if let Some(rect) = button_rects.get(1) {
            self.save_button.position = rect.position();
        }
        if let Some(rect) = button_rects.get(2) {
            self.open_button.position = rect.position();
        }
        if let Some(rect) = button_rects.get(3) {
            self.delete_button.position = rect.position();
        }
        y_offset += BUTTON_ROW_HEIGHT + SPACING;
        
        // Toolbar
        let toolbar_rect = Rect::new(
            content_rect.x,
            y_offset,
            content_rect.width,
            TOOLBAR_HEIGHT,
        );
        self.toolbar.position = toolbar_rect.position();
        self.toolbar.size = toolbar_rect.size();
        self.toolbar.update_layout();
        y_offset += TOOLBAR_HEIGHT + SPACING;
        
        // Editor area below toolbar
        let editor_height = content_rect.bottom() - y_offset;
        let editor_rect = Rect::new(
            content_rect.x,
            y_offset,
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
                    });
                }
            }
            self.notepad_modal.set_papers(papers);
        }
    }

    pub fn hit_test(&mut self, pos: Vec2) -> NotepadHit {
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

