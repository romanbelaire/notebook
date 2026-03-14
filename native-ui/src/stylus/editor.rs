use glam::Vec2;
use crate::stylus::block::{Block, BlockType, BlockContent};
use crate::stylus::{Document, Cursor, Selection, SlashCommand};
use crate::stylus::formatting::{TextFormat, FormatSpan};
use crate::ui::text_editor::TextEditor;

pub struct StylusEditor {
    pub document: Document,
    pub document_id: Option<String>,  // ID for persistence
    pub cursor: Option<Cursor>,
    pub selection: Option<Selection>,
    pub position: Vec2,
    pub size: Vec2,
    pub scroll_offset: f32,
    pub focused_block_id: Option<String>,
    pub slash_command_active: bool,
    pub slash_command_query: String,
    pub slash_command_matches: Vec<SlashCommand>,
    pub selected_slash_index: usize,
    pub cursor_blink_time: f32,
    pub last_save_time: f32,  // Time since last save (for debouncing)
    pub has_unsaved_changes: bool,  // Track if document needs saving
    pub is_selecting: bool,  // Track if user is dragging to select text
    pub selection_start_pos: Option<Vec2>,  // Mouse position where selection started
    pub selection_anchor: Option<Cursor>,  // Anchor point for keyboard selection extension
    pub dragging_block_id: Option<String>,  // Block being dragged
    pub drag_start_pos: Option<Vec2>,  // Mouse position where drag started
    pub drag_start_index: Option<usize>,  // Original index of dragged block
    /// Term to highlight (all occurrences) in the notepad; set from selection on mouse up, cleared when selection empty.
    pub highlight_term: Option<String>,
}

impl StylusEditor {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        let mut doc = Document::new();
        // Focus the first block on creation
        let first_block_id = doc.blocks[0].id.clone();
        let mut editor = Self {
            document: doc,
            document_id: None,
            cursor: None,
            selection: None,
            position,
            size,
            scroll_offset: 0.0,
            focused_block_id: None,
            slash_command_active: false,
            slash_command_query: String::new(),
            slash_command_matches: Vec::new(),
            selected_slash_index: 0,
            cursor_blink_time: 0.0,
            last_save_time: 0.0,
            has_unsaved_changes: false,
            is_selecting: false,
            selection_start_pos: None,
            selection_anchor: None,
            dragging_block_id: None,
            drag_start_pos: None,
            drag_start_index: None,
            highlight_term: None,
        };
        editor.focus_block(&first_block_id);
        editor
    }

    /// Create a new editor with a document ID (for loading existing documents)
    pub fn with_document_id(position: Vec2, size: Vec2, document_id: String) -> Self {
        let mut editor = Self::new(position, size);
        editor.document_id = Some(document_id);
        editor
    }

    /// Load a document by ID
    pub fn load_document(&mut self, document_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::DocumentPersistence;
        let doc = DocumentPersistence::load_document(document_id)?;
        self.document = doc;
        self.document_id = Some(document_id.to_string());
        self.has_unsaved_changes = false;
        self.last_save_time = 0.0;
        // Focus the first block
        if let Some(first_block_id) = self.document.blocks.first().map(|b| b.id.clone()) {
            self.focus_block(&first_block_id);
        }
        Ok(())
    }

    /// Save the document (auto-save)
    pub fn save(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::DocumentPersistence;
        if let Some(ref doc_id) = self.document_id {
            self.document.update_metadata();
            DocumentPersistence::save_document(doc_id, &self.document)?;
            self.has_unsaved_changes = false;
            self.last_save_time = 0.0;
        }
        Ok(())
    }

    /// Create a new document and assign an ID
    pub fn create_new_document(&mut self) -> String {
        use uuid::Uuid;
        let doc_id = format!("doc_{}", Uuid::new_v4().to_string().replace("-", ""));
        self.document_id = Some(doc_id.clone());
        self.document = Document::new();
        self.has_unsaved_changes = true;
        self.last_save_time = 0.0;
        // Focus the first block
        if let Some(first_block_id) = self.document.blocks.first().map(|b| b.id.clone()) {
            self.focus_block(&first_block_id);
        }
        doc_id
    }

    /// Load markdown content into the editor
    pub fn load_from_markdown(&mut self, markdown: &str) {
        use pulldown_cmark::{Parser, Event, Tag, TagEnd};
        use crate::stylus::block::{Block, BlockType, BlockContent};
        
        let parser = Parser::new(markdown);
        let mut blocks = Vec::new();
        let mut current_text = String::new();
        let mut current_block_type = BlockType::Paragraph;
        
        for event in parser {
            match event {
                Event::Start(Tag::Heading { level, .. }) => {
                    // Finish current block if it has content
                    if !current_text.trim().is_empty() {
                        let block = Block {
                            id: uuid::Uuid::new_v4().to_string(),
                            block_type: current_block_type.clone(),
                            content: BlockContent::Text {
                                text: current_text.clone(),
                                formats: Vec::new(),
                            },
                            collapsed: false,
                            metadata: serde_json::Value::Null,
                        };
                        blocks.push(block);
                        current_text.clear();
                    }
                    
                    // Set new block type based on heading level
                    current_block_type = match level {
                        pulldown_cmark::HeadingLevel::H1 => BlockType::Heading1,
                        pulldown_cmark::HeadingLevel::H2 => BlockType::Heading2,
                        pulldown_cmark::HeadingLevel::H3 => BlockType::Heading3,
                        pulldown_cmark::HeadingLevel::H4 => BlockType::Heading4,
                        pulldown_cmark::HeadingLevel::H5 => BlockType::Heading5,
                        pulldown_cmark::HeadingLevel::H6 => BlockType::Heading5, // Use Heading5 for level 6
                    };
                }
                Event::End(TagEnd::Heading { .. }) => {
                    // Finish heading block
                    if !current_text.trim().is_empty() {
                        let block = Block {
                            id: uuid::Uuid::new_v4().to_string(),
                            block_type: current_block_type.clone(),
                            content: BlockContent::Text {
                                text: current_text.clone(),
                                formats: Vec::new(),
                            },
                            collapsed: false,
                            metadata: serde_json::Value::Null,
                        };
                        blocks.push(block);
                        current_text.clear();
                    }
                    current_block_type = BlockType::Paragraph;
                }
                Event::Text(text) => {
                    current_text.push_str(&text);
                }
                Event::Code(code) => {
                    // Finish current block and create code block
                    if !current_text.trim().is_empty() {
                        let block = Block {
                            id: uuid::Uuid::new_v4().to_string(),
                            block_type: current_block_type.clone(),
                            content: BlockContent::Text {
                                text: current_text.clone(),
                                formats: Vec::new(),
                            },
                            collapsed: false,
                            metadata: serde_json::Value::Null,
                        };
                        blocks.push(block);
                        current_text.clear();
                    }
                    
                    let code_block = Block {
                        id: uuid::Uuid::new_v4().to_string(),
                        block_type: BlockType::Code,
                        content: BlockContent::Code {
                            code: code.to_string(),
                            language: None,
                        },
                        collapsed: false,
                        metadata: serde_json::Value::Null,
                    };
                    blocks.push(code_block);
                }
                Event::SoftBreak | Event::HardBreak => {
                    current_text.push('\n');
                }
                Event::End(TagEnd::Paragraph) => {
                    if !current_text.trim().is_empty() {
                        let block = Block {
                            id: uuid::Uuid::new_v4().to_string(),
                            block_type: current_block_type.clone(),
                            content: BlockContent::Text {
                                text: current_text.clone(),
                                formats: Vec::new(),
                            },
                            collapsed: false,
                            metadata: serde_json::Value::Null,
                        };
                        blocks.push(block);
                        current_text.clear();
                    }
                }
                _ => {}
            }
        }
        
        // Add any remaining text
        if !current_text.trim().is_empty() {
            let block = Block {
                id: uuid::Uuid::new_v4().to_string(),
                block_type: current_block_type.clone(),
                content: BlockContent::Text {
                    text: current_text.clone(),
                    formats: Vec::new(),
                },
                collapsed: false,
                metadata: serde_json::Value::Null,
            };
            blocks.push(block);
        }
        
        // If no blocks were created, create an empty paragraph
        if blocks.is_empty() {
            blocks.push(Block::new(BlockType::Paragraph));
        }
        
        // Update document
        self.document.blocks = blocks;
        self.has_unsaved_changes = true;
        
        // Focus the first block
        if let Some(first_block_id) = self.document.blocks.first().map(|b| b.id.clone()) {
            self.focus_block(&first_block_id);
        }
    }

    /// Mark document as changed (called when content is modified)
    pub fn mark_changed(&mut self) {
        self.has_unsaved_changes = true;
    }

    pub fn create_block(&mut self, block_type: BlockType, after_id: Option<&str>) -> String {
        let block = Block::new(block_type);
        let id = block.id.clone();
        self.document.insert_block(block, after_id);
        self.mark_changed();
        id
    }

    pub fn delete_block(&mut self, id: &str) -> Option<Block> {
        if self.document.blocks.len() > 1 {
            let result = self.document.remove_block(id);
            if result.is_some() {
                self.mark_changed();
            }
            result
        } else {
            None
        }
    }

    pub fn update_block(&mut self, id: &str, content: crate::stylus::block::BlockContent) {
        if let Some(block) = self.document.get_block_mut(id) {
            block.content = content;
            self.document.update_metadata();
            self.mark_changed();
        }
    }

    pub fn focus_block(&mut self, id: &str) {
        self.focused_block_id = Some(id.to_string());
        if let Some(block) = self.document.get_block(id) {
            if let Some(text) = block.content.get_text() {
                self.cursor = Some(Cursor::at_end(id.to_string(), text.len()));
            } else {
                self.cursor = Some(Cursor::at_start(id.to_string()));
            }
        }
    }

    pub fn insert_text(&mut self, new_text: &str) {
        // Ensure we have a cursor - if not, focus the first block
        if self.cursor.is_none() {
            if !self.document.blocks.is_empty() {
                let first_block_id = self.document.blocks[0].id.clone();
                self.focus_block(&first_block_id);
            } else {
                return; // No blocks to insert into
            }
        }
        
        if let Some(ref mut cursor) = self.cursor {
            if let Some(block) = self.document.get_block_mut(&cursor.block_id) {
                if let BlockContent::Text { ref mut text, ref mut formats } = block.content {
                    // If there's a selection, delete it first
                    if let Some(ref selection) = self.selection {
                        if selection.start.block_id == cursor.block_id && selection.end.block_id == cursor.block_id {
                            let start = selection.start.position.min(selection.end.position);
                            let end = selection.start.position.max(selection.end.position);
                            if end <= text.len() {
                                text.drain(start..end);
                                // Remove format spans that are now invalid
                                formats.retain(|span| {
                                    !(span.start >= start && span.end <= end) && 
                                    !(span.start < end && span.end > start)
                                });
                                // Adjust format span positions
                                for span in formats.iter_mut() {
                                    if span.start > end {
                                        span.start -= end - start;
                                        span.end -= end - start;
                                    } else if span.end > start {
                                        span.end = span.end.min(start) + (span.end - end.max(span.start));
                                    }
                                }
                                cursor.position = start;
                                self.selection = None;
                            }
                        }
                    }
                    
                    let text_len = new_text.len();
                    text.insert_str(cursor.position, new_text);
                    cursor.position += text_len;
                    self.document.update_metadata();
                    self.mark_changed();
                }
            }
        }
    }
    
    /// Apply formatting to selected text or toggle formatting at cursor position
    pub fn apply_format(&mut self, format: TextFormat) {
        if let Some(ref cursor) = self.cursor {
            if let Some(block) = self.document.get_block_mut(&cursor.block_id) {
                if let BlockContent::Text { ref mut text, ref mut formats } = block.content {
                    let (start, end) = if let Some(ref selection) = self.selection {
                        if selection.start.block_id == cursor.block_id && selection.end.block_id == cursor.block_id {
                            let s = selection.start.position.min(selection.end.position);
                            let e = selection.start.position.max(selection.end.position);
                            (s, e)
                        } else {
                            return; // Selection spans multiple blocks - not supported yet
                        }
                    } else {
                        // No selection - toggle format at cursor position
                        let pos = cursor.position;
                        (pos, pos)
                    };
                    
                    // Check if format already exists in this range
                    let has_format = formats.iter().any(|span| {
                        span.overlaps(start, end) && span.format == format
                    });
                    
                    if has_format {
                        // Remove format from this range
                        let mut new_formats = Vec::new();
                        for span in formats.iter() {
                            if span.format == format && span.overlaps(start, end) {
                                // Split or trim the span
                                if span.start < start {
                                    new_formats.push(FormatSpan::new(span.start, start, span.format.clone()));
                                }
                                if span.end > end {
                                    new_formats.push(FormatSpan::new(end, span.end, span.format.clone()));
                                }
                            } else {
                                new_formats.push(span.clone());
                            }
                        }
                        *formats = new_formats;
                    } else {
                        // Add format to this range
                        formats.push(FormatSpan::new(start, end, format));
                    }
                    
                    self.document.update_metadata();
                    self.mark_changed();
                }
            }
        }
    }

    pub fn delete_char(&mut self, forward: bool) {
        if let Some(ref mut cursor) = self.cursor {
            let block_id = cursor.block_id.clone();
            let cursor_pos = cursor.position;
            
            // Get mutable access to block
            let should_delete = if let Some(block) = self.document.get_block_mut(&block_id) {
                if let Some(block_text) = block.content.get_text_mut() {
                    if forward {
                        if cursor_pos < block_text.len() {
                            block_text.remove(cursor_pos);
                            self.mark_changed();
                            false
                        } else if cursor_pos == block_text.len() {
                            // Try to merge with next block or delete current if empty
                            let is_empty = block_text.trim().is_empty();
                            if is_empty {
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else if cursor_pos > 0 {
                        cursor.position -= 1;
                        block_text.remove(cursor.position);
                        self.mark_changed();
                        false
                    } else {
                        // Try to merge with previous block
                        true
                    }
                } else {
                    false
                }
            } else {
                false
            };

            if should_delete {
                if forward {
                    // Delete current block if empty and move to next
                    if let Some(next_id) = Self::get_next_block_id(&self.document, &block_id) {
                        self.delete_block(&block_id);
                        self.focus_block(&next_id);
                        if let Some(ref mut c) = self.cursor {
                            c.position = 0;
                        }
                    }
                } else {
                    // Merge with previous block
                    if let Some(prev_id) = Self::get_prev_block_id(&self.document, &block_id) {
                        // Get text from both blocks before modifying
                        let prev_text = self.document.get_block(&prev_id)
                            .and_then(|b| b.content.get_text().map(|s| s.to_string()));
                        let block_text = self.document.get_block(&block_id)
                            .and_then(|b| b.content.get_text().map(|s| s.to_string()));
                        
                        if let (Some(prev_text), Some(block_text)) = (prev_text, block_text) {
                            let prev_text_len = prev_text.len();
                            let merged = format!("{}{}", prev_text, block_text);
                            self.update_block(&prev_id, BlockContent::text(merged));
                            // Remove block directly to avoid double-marking as changed
                            let _ = self.document.remove_block(&block_id);
                            self.focus_block(&prev_id);
                            if let Some(ref mut c) = self.cursor {
                                c.position = prev_text_len;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn move_cursor(&mut self, direction: CursorDirection) {
        // Clone cursor data first to avoid borrow conflicts
        let (block_id, current_pos) = if let Some(ref cursor) = self.cursor {
            (cursor.block_id.clone(), cursor.position)
        } else {
            return;
        };
        
        // Get text length for current block
        let current_text_len = self.document.get_block(&block_id)
            .and_then(|b| b.content.get_text())
            .map(|t| t.len())
            .unwrap_or(0);
        
        // Collect all needed information before making mutable calls
        // We need to get all block IDs and text lengths first
        let prev_id_opt = Self::get_prev_block_id(&self.document, &block_id);
        let next_id_opt = Self::get_next_block_id(&self.document, &block_id);
        
        let prev_text_len = prev_id_opt.as_ref()
            .and_then(|id| self.document.get_block(id))
            .and_then(|b| b.content.get_text())
            .map(|t| t.len())
            .unwrap_or(0);
        
        let next_text_len = next_id_opt.as_ref()
            .and_then(|id| self.document.get_block(id))
            .and_then(|b| b.content.get_text())
            .map(|t| t.len())
            .unwrap_or(0);
        
        // Now determine new block and position
        let (new_block_id, new_pos) = match direction {
            CursorDirection::Left => {
                if current_pos > 0 {
                    (block_id.clone(), current_pos - 1)
                } else {
                    // Move to previous block
                    if let Some(prev_id) = prev_id_opt {
                        (prev_id, prev_text_len)
                    } else {
                        (block_id.clone(), current_pos)
                    }
                }
            }
            CursorDirection::Right => {
                if current_pos < current_text_len {
                    (block_id.clone(), current_pos + 1)
                } else {
                    // Move to next block
                    if let Some(next_id) = next_id_opt {
                        (next_id, 0)
                    } else {
                        (block_id.clone(), current_pos)
                    }
                }
            }
            CursorDirection::Up => {
                // Move to previous block
                if let Some(prev_id) = prev_id_opt {
                    (prev_id, current_pos.min(prev_text_len))
                } else {
                    (block_id.clone(), current_pos)
                }
            }
            CursorDirection::Down => {
                // Move to next block
                if let Some(next_id) = next_id_opt {
                    (next_id, current_pos.min(next_text_len))
                } else {
                    (block_id.clone(), current_pos)
                }
            }
        };
        
        // Now make mutable calls
        if new_block_id != block_id {
            self.focus_block(&new_block_id);
        }
        if let Some(ref mut cursor) = self.cursor {
            cursor.position = new_pos;
        }
    }

    pub fn handle_slash_command(&mut self, ch: char) {
        if ch == '/' && !self.slash_command_active {
            self.slash_command_active = true;
            self.slash_command_query.clear();
            self.slash_command_matches = SlashCommand::all_commands();
            self.selected_slash_index = 0;
        } else if self.slash_command_active {
            if ch == '\n' || ch == '\r' {
                // Execute selected command
                let cmd_opt = self.slash_command_matches.get(self.selected_slash_index).copied();
                self.slash_command_active = false;
                self.slash_command_query.clear();
                self.slash_command_matches.clear();
                if let Some(cmd) = cmd_opt {
                    self.execute_slash_command(&cmd);
                }
            } else if ch == '\x08' || ch == '\x7f' {
                // Backspace
                if !self.slash_command_query.is_empty() {
                    self.slash_command_query.pop();
                    self.update_slash_matches();
                } else {
                    self.slash_command_active = false;
                }
            } else if ch == '\x1b' {
                // Escape
                self.slash_command_active = false;
                self.slash_command_query.clear();
                self.slash_command_matches.clear();
            } else {
                self.slash_command_query.push(ch);
                self.update_slash_matches();
            }
        }
    }

    fn update_slash_matches(&mut self) {
        if self.slash_command_query.is_empty() {
            self.slash_command_matches = SlashCommand::all_commands();
        } else {
            self.slash_command_matches = SlashCommand::search(&self.slash_command_query);
        }
        self.selected_slash_index = 0;
    }

    fn execute_slash_command(&mut self, cmd: &SlashCommand) {
        if let Some(ref cursor) = self.cursor {
            let block_id = cursor.block_id.clone();
            let is_empty = self.document.get_block(&block_id)
                .map(|b| b.is_empty())
                .unwrap_or(false);
            
            let block_type = cmd.to_block_type();
            let new_block_id = self.create_block(block_type, Some(&block_id));
            
            // If current block is empty, replace it; otherwise insert after
            if is_empty {
                self.delete_block(&block_id);
            }
            
            self.focus_block(&new_block_id);
        }
    }

    fn get_next_block_id(document: &Document, id: &str) -> Option<String> {
        if let Some(index) = document.get_block_index(id) {
            if index + 1 < document.blocks.len() {
                Some(document.blocks[index + 1].id.clone())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_prev_block_id(document: &Document, id: &str) -> Option<String> {
        if let Some(index) = document.get_block_index(id) {
            if index > 0 {
                Some(document.blocks[index - 1].id.clone())
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        pos.x >= self.position.x
            && pos.x <= self.position.x + self.size.x
            && pos.y >= self.position.y
            && pos.y <= self.position.y + self.size.y
    }

    pub fn on_mouse_wheel(&mut self, delta: f32) {
        // Scroll the editor
        let scroll_speed = 20.0;
        self.scroll_offset = (self.scroll_offset - delta * scroll_speed).max(0.0);
        
        // Calculate max scroll based on content height
        let total_height: f32 = self.document.blocks.iter()
            .map(|b| {
                use crate::stylus::renderer::StylusRenderer;
                StylusRenderer::get_block_height_static(b, self.size.x - 20.0) + 8.0
            })
            .sum();
        
        let max_scroll = (total_height - self.size.y).max(0.0);
        self.scroll_offset = self.scroll_offset.min(max_scroll);
    }

    pub fn update(&mut self, dt: f32) {
        // Update cursor blink animation
        self.cursor_blink_time += dt;
        // Reset after 1 second (for 0.5s on, 0.5s off cycle)
        if self.cursor_blink_time > 1.0 {
            self.cursor_blink_time -= 1.0;
        }
    }

    pub fn is_cursor_visible(&self) -> bool {
        // Cursor is visible for the first 0.5 seconds of each cycle
        self.cursor_blink_time < 0.5
    }
    
    /// Handle mouse down - start text selection
    pub fn on_mouse_down(&mut self, pos: Vec2, click_count: u32) {
        if self.contains(pos) {
            self.is_selecting = true;
            self.selection_start_pos = Some(pos);
            // Clear existing selection
            self.selection = None;
            // Set cursor position based on click
            self.set_cursor_from_position(pos);
            
            // Handle double/triple click for word/line selection
            match click_count {
                2 => {
                    // Double click: select word at cursor
                    self.select_word_at_cursor();
                }
                3 => {
                    // Triple click: select line (entire block text)
                    self.select_line_at_cursor();
                }
                _ => {
                    // Single click: just position cursor
                }
            }
        }
    }
    
    /// Select word at cursor position
    fn select_word_at_cursor(&mut self) {
        if let Some(ref cursor) = self.cursor {
            if let Some(block) = self.document.get_block(&cursor.block_id) {
                if let Some(text) = block.content.get_text() {
                    let pos = cursor.position.min(text.len());
                    let chars: Vec<char> = text.chars().collect();
                    
                    // Find word boundaries
                    let mut start = pos;
                    let mut end = pos;
                    
                    // If cursor is on whitespace, select the whitespace
                    if pos < chars.len() && chars[pos].is_whitespace() {
                        while start > 0 && chars[start - 1].is_whitespace() {
                            start -= 1;
                        }
                        while end < chars.len() && chars[end].is_whitespace() {
                            end += 1;
                        }
                    } else {
                        // Select word characters backward
                        while start > 0 && !chars[start - 1].is_whitespace() {
                            start -= 1;
                        }
                        // Select word characters forward
                        while end < chars.len() && !chars[end].is_whitespace() {
                            end += 1;
                        }
                    }
                    
                    // Create selection
                    let start_cursor = Cursor::new(cursor.block_id.clone(), start);
                    let end_cursor = Cursor::new(cursor.block_id.clone(), end);
                    self.selection = Some(Selection::new(start_cursor, end_cursor));
                    if let Some(ref mut c) = self.cursor {
                        c.position = end;
                    }
                }
            }
        }
    }
    
    /// Select entire line (block text) at cursor
    fn select_line_at_cursor(&mut self) {
        if let Some(ref cursor) = self.cursor {
            if let Some(block) = self.document.get_block(&cursor.block_id) {
                if let Some(text) = block.content.get_text() {
                    let start_cursor = Cursor::at_start(cursor.block_id.clone());
                    let end_cursor = Cursor::at_end(cursor.block_id.clone(), text.len());
                    self.selection = Some(Selection::new(start_cursor, end_cursor));
                    if let Some(ref mut c) = self.cursor {
                        c.position = text.len();
                    }
                }
            }
        }
    }
    
    /// Clear the current selection
    fn clear_selection(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
    }
    
    /// Check if there is an active selection
    fn has_selection(&self) -> bool {
        self.selection.is_some()
    }
    
    /// Select all text in the document
    pub fn select_all(&mut self) {
        if self.document.blocks.is_empty() {
            return;
        }
        
        // Find first and last text blocks
        let mut first_block_id = None;
        let mut last_block_id = None;
        let mut last_text_len = 0;
        
        for block in &self.document.blocks {
            if block.content.get_text().is_some() {
                if first_block_id.is_none() {
                    first_block_id = Some(block.id.clone());
                }
                last_block_id = Some(block.id.clone());
                if let Some(text) = block.content.get_text() {
                    last_text_len = text.len();
                }
            }
        }
        
        if let (Some(first_id), Some(last_id)) = (first_block_id, last_block_id) {
            let start_cursor = Cursor::at_start(first_id.clone());
            let end_cursor = Cursor::at_end(last_id.clone(), last_text_len);
            self.selection = Some(Selection::new(start_cursor, end_cursor.clone()));
            
            // Move cursor to end of selection
            self.cursor = Some(end_cursor);
            self.focused_block_id = Some(last_id);
        }
    }
    
    /// Get the selected text as a string
    pub fn get_selected_text(&self) -> String {
        if let Some(ref selection) = self.selection {
            let mut result = String::new();
            
            // Get all blocks in order
            let block_ids: Vec<String> = self.document.blocks.iter()
                .map(|b| b.id.clone())
                .collect();
            
            let start_block_idx = block_ids.iter()
                .position(|id| id == &selection.start.block_id);
            let end_block_idx = block_ids.iter()
                .position(|id| id == &selection.end.block_id);
            
            if let (Some(start_idx), Some(end_idx)) = (start_block_idx, end_block_idx) {
                for (idx, block_id) in block_ids.iter().enumerate() {
                    if idx >= start_idx && idx <= end_idx {
                        if let Some(block) = self.document.get_block(block_id) {
                            if let Some(text) = block.content.get_text() {
                                if idx == start_idx && idx == end_idx {
                                    // Same block - extract range
                                    let start = selection.start.position.min(selection.end.position);
                                    let end = selection.start.position.max(selection.end.position);
                                    if end <= text.len() {
                                        result.push_str(&text[start..end]);
                                    }
                                } else if idx == start_idx {
                                    // First block - from start position to end
                                    let start = selection.start.position;
                                    if start < text.len() {
                                        result.push_str(&text[start..]);
                                    }
                                    result.push('\n');
                                } else if idx == end_idx {
                                    // Last block - from start to end position
                                    let end = selection.end.position.min(text.len());
                                    if end > 0 {
                                        result.push_str(&text[..end]);
                                    }
                                } else {
                                    // Middle block - entire text
                                    result.push_str(text);
                                    result.push('\n');
                                }
                            }
                        }
                    }
                }
            }
            
            result
        } else {
            String::new()
        }
    }
    
    /// Paste text at cursor position, replacing selection if present
    pub fn paste(&mut self, text: &str) {
        // If there's a selection, delete it first
        if let Some(ref selection) = self.selection {
            if let Some(ref cursor) = self.cursor {
                if selection.start.block_id == cursor.block_id && selection.end.block_id == cursor.block_id {
                    let start = selection.start.position.min(selection.end.position);
                    let end = selection.start.position.max(selection.end.position);
                    
                    if let Some(block) = self.document.get_block_mut(&cursor.block_id) {
                        if let Some(block_text) = block.content.get_text_mut() {
                            if end <= block_text.len() {
                                block_text.drain(start..end);
                                // Remove format spans that are now invalid
                                if let BlockContent::Text { ref mut formats, .. } = block.content {
                                    formats.retain(|span| {
                                        !(span.start >= start && span.end <= end) && 
                                        !(span.start < end && span.end > start)
                                    });
                                    // Adjust format span positions
                                    for span in formats.iter_mut() {
                                        if span.start > end {
                                            span.start -= end - start;
                                            span.end -= end - start;
                                        } else if span.start > start {
                                            span.start = start;
                                            if span.end > end {
                                                span.end = start;
                                            }
                                        }
                                    }
                                }
                                if let Some(ref mut c) = self.cursor {
                                    c.position = start;
                                }
                                self.selection = None;
                                self.mark_changed();
                            }
                        }
                    }
                }
            }
        }
        
        // Insert the text
        self.insert_text(text);
    }
    
    /// Update selection from anchor point to current cursor position
    fn update_selection_from_anchor(&mut self) {
        if let Some(ref anchor) = self.selection_anchor {
            if let Some(ref cursor) = self.cursor {
                // Create selection from anchor to current cursor
                let start = anchor.clone();
                let end = cursor.clone();
                self.selection = Some(Selection::new(start, end));
            }
        }
    }
    
    /// Move cursor one word to the left
    fn move_cursor_word_left(&mut self, extend_selection: bool) {
        if let Some(ref mut cursor) = self.cursor {
            let block_id = cursor.block_id.clone();
            let cursor_pos = cursor.position;
            
            if let Some(block) = self.document.get_block(&block_id) {
                if let Some(text) = block.content.get_text() {
                    if cursor_pos == 0 {
                        // At start of block, try to move to previous block
                        if let Some(prev_id) = Self::get_prev_block_id(&self.document, &block_id) {
                            if let Some(prev_block) = self.document.get_block(&prev_id) {
                                if let Some(prev_text) = prev_block.content.get_text() {
                                    let prev_text_len = prev_text.len();
                                    self.focus_block(&prev_id);
                                    if let Some(ref mut c) = self.cursor {
                                        c.position = prev_text_len;
                                    }
                                    if extend_selection {
                                        if self.selection_anchor.is_none() {
                                            self.selection_anchor = Some(Cursor::new(block_id, cursor_pos));
                                        }
                                        self.update_selection_from_anchor();
                                    } else {
                                        self.clear_selection();
                                    }
                                    return;
                                }
                            }
                        }
                        if !extend_selection {
                            self.clear_selection();
                        }
                        return;
                    }
                    
                    let chars: Vec<char> = text.chars().collect();
                    let mut new_pos = cursor_pos.min(chars.len());
                    
                    // Skip whitespace backward
                    while new_pos > 0 && chars[new_pos - 1].is_whitespace() {
                        new_pos -= 1;
                    }
                    
                    // Skip word characters backward
                    while new_pos > 0 && !chars[new_pos - 1].is_whitespace() {
                        new_pos -= 1;
                    }
                    
                    cursor.position = new_pos;
                    if extend_selection {
                        if self.selection_anchor.is_none() {
                            self.selection_anchor = Some(Cursor::new(block_id, cursor_pos));
                        }
                        self.update_selection_from_anchor();
                    } else {
                        self.clear_selection();
                    }
                }
            }
        }
    }
    
    /// Move cursor one word to the right
    fn move_cursor_word_right(&mut self, extend_selection: bool) {
        if let Some(ref mut cursor) = self.cursor {
            let block_id = cursor.block_id.clone();
            let cursor_pos = cursor.position;
            
            if let Some(block) = self.document.get_block(&block_id) {
                if let Some(text) = block.content.get_text() {
                    let text_len = text.len();
                    
                    if cursor_pos >= text_len {
                        // At end of block, try to move to next block
                        if let Some(next_id) = Self::get_next_block_id(&self.document, &block_id) {
                            self.focus_block(&next_id);
                            if let Some(ref mut c) = self.cursor {
                                c.position = 0;
                            }
                            if extend_selection {
                                if self.selection_anchor.is_none() {
                                    self.selection_anchor = Some(Cursor::new(block_id, cursor_pos));
                                }
                                self.update_selection_from_anchor();
                            } else {
                                self.clear_selection();
                            }
                            return;
                        }
                        if !extend_selection {
                            self.clear_selection();
                        }
                        return;
                    }
                    
                    let chars: Vec<char> = text.chars().collect();
                    let mut new_pos = cursor_pos.min(text_len);
                    
                    // Skip word characters forward
                    while new_pos < text_len && !chars[new_pos].is_whitespace() {
                        new_pos += 1;
                    }
                    
                    // Skip whitespace forward
                    while new_pos < text_len && chars[new_pos].is_whitespace() {
                        new_pos += 1;
                    }
                    
                    cursor.position = new_pos;
                    if extend_selection {
                        if self.selection_anchor.is_none() {
                            self.selection_anchor = Some(Cursor::new(block_id, cursor_pos));
                        }
                        self.update_selection_from_anchor();
                    } else {
                        self.clear_selection();
                    }
                }
            }
        }
    }
    
    /// Handle mouse move - update selection or block drag
    pub fn on_mouse_move(&mut self, pos: Vec2) {
        // Handle block dragging
        if let Some(ref dragging_id) = self.dragging_block_id {
            // Calculate which index to move the block to
            let padding = 10.0;
            let mut y_offset = self.position.y + padding - self.scroll_offset;
            let block_spacing = 8.0;
            let mut target_index = 0;
            
            for (i, block) in self.document.blocks.iter().enumerate() {
                let block_height = crate::stylus::renderer::StylusRenderer::get_block_height_static(block, self.size.x - padding * 2.0);
                let block_center = y_offset + block_height / 2.0;
                
                if pos.y < block_center {
                    target_index = i;
                    break;
                }
                
                target_index = i + 1;
                y_offset += block_height + block_spacing;
            }
            
            // Move the block if the index changed
            if let Some(start_index) = self.drag_start_index {
                if target_index != start_index && target_index < self.document.blocks.len() {
                    let _ = self.document.move_block(dragging_id, target_index);
                    self.drag_start_index = Some(target_index);
                    self.mark_changed();
                }
            }
            return;
        }
        
        // Handle text selection
        if self.is_selecting {
            if let Some(start_pos) = self.selection_start_pos {
                // Clamp position to editor bounds for cursor calculation
                let clamped_pos = Vec2::new(
                    pos.x.max(self.position.x).min(self.position.x + self.size.x),
                    pos.y.max(self.position.y).min(self.position.y + self.size.y),
                );
                
                // Update selection based on drag
                let start_cursor = self.get_cursor_from_position(start_pos);
                let end_cursor = self.get_cursor_from_position(clamped_pos);
                if let (Some(start), Some(end)) = (start_cursor, end_cursor) {
                    let end_clone = end.clone();
                    self.selection = Some(Selection::new(start, end));
                    self.cursor = Some(end_clone);
                }
            }
        }
    }
    
    /// Handle mouse up - end text selection or block drag
    pub fn on_mouse_up(&mut self, _pos: Vec2) {
        // End block dragging
        if self.dragging_block_id.is_some() {
            self.dragging_block_id = None;
            self.drag_start_pos = None;
            self.drag_start_index = None;
        }
        
        // Update highlight term from selection: set when selection has text, clear when empty
        self.highlight_term = self.selection.as_ref().map(|_| self.get_selected_text()).filter(|s| !s.is_empty());
        
        // End text selection
        self.is_selecting = false;
        self.selection_start_pos = None;
    }
    
    /// Set cursor position from screen coordinates
    fn set_cursor_from_position(&mut self, pos: Vec2) {
        let padding = 10.0;
        let mut y_offset = self.position.y + padding - self.scroll_offset;
        let line_height = 24.0;
        let block_spacing = 8.0;
        
        for block in &self.document.blocks {
            let block_height = crate::stylus::renderer::StylusRenderer::get_block_height_static(block, self.size.x - padding * 2.0);
            
            if pos.y >= y_offset && pos.y < y_offset + block_height {
                // Clicked on this block
                let block_id = block.id.clone();
                if let Some(text) = block.content.get_text() {
                    // Calculate character position from x coordinate
                    let font_size = crate::stylus::renderer::StylusRenderer::get_font_size_static(&block.block_type);
                    let rel_x = pos.x - (self.position.x + padding);
                    let char_width = font_size * 0.6;
                    let char_pos = (rel_x / char_width).max(0.0) as usize;
                    let cursor_pos = char_pos.min(text.len());
                    
                    self.focus_block(&block_id);
                    if let Some(ref mut cursor) = self.cursor {
                        cursor.position = cursor_pos;
                    }
                } else {
                    self.focus_block(&block_id);
                }
                return;
            }
            
            y_offset += block_height + block_spacing;
        }
    }
    
    /// Get cursor from screen coordinates
    fn get_cursor_from_position(&self, pos: Vec2) -> Option<Cursor> {
        let padding = 10.0;
        let mut y_offset = self.position.y + padding - self.scroll_offset;
        let line_height = 24.0;
        let block_spacing = 8.0;
        
        for block in &self.document.blocks {
            let block_height = crate::stylus::renderer::StylusRenderer::get_block_height_static(block, self.size.x - padding * 2.0);
            
            if pos.y >= y_offset && pos.y < y_offset + block_height {
                // Clicked on this block
                if let Some(text) = block.content.get_text() {
                    let font_size = crate::stylus::renderer::StylusRenderer::get_font_size_static(&block.block_type);
                    let rel_x = pos.x - (self.position.x + padding);
                    let char_width = font_size * 0.6;
                    let char_pos = (rel_x / char_width).max(0.0) as usize;
                    let cursor_pos = char_pos.min(text.len());
                    return Some(Cursor::new(block.id.clone(), cursor_pos));
                } else {
                    return Some(Cursor::at_start(block.id.clone()));
                }
            }
            
            y_offset += block_height + block_spacing;
        }
        None
    }

    /// Set cursor to a character index in a block (used after glyph-based hit test in renderer).
    pub fn set_cursor_from_character_index(&mut self, block_id: &str, char_idx: usize) {
        self.focus_block(block_id);
        if let Some(ref mut cursor) = self.cursor {
            cursor.position = char_idx;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorDirection {
    Left,
    Right,
    Up,
    Down,
}

impl TextEditor for StylusEditor {
    fn on_char_received(&mut self, ch: char) {
        // Handle slash commands first
        if self.slash_command_active || ch == '/' {
            self.handle_slash_command(ch);
            return;
        }
        
        // Normal text insertion
        self.insert_text(&ch.to_string());
    }
    
    fn on_backspace(&mut self) {
        // If there's a selection, delete it first
        if let Some(ref selection) = self.selection {
            if let Some(ref cursor) = self.cursor {
                if selection.start.block_id == cursor.block_id && selection.end.block_id == cursor.block_id {
                    let start = selection.start.position.min(selection.end.position);
                    let end = selection.start.position.max(selection.end.position);
                    
                    if let Some(block) = self.document.get_block_mut(&cursor.block_id) {
                        if let Some(block_text) = block.content.get_text_mut() {
                            if end <= block_text.len() {
                                block_text.drain(start..end);
                                // Remove format spans that are now invalid
                                if let BlockContent::Text { ref mut formats, .. } = block.content {
                                    formats.retain(|span| {
                                        !(span.start >= start && span.end <= end) && 
                                        !(span.start < end && span.end > start)
                                    });
                                    // Adjust format span positions
                                    for span in formats.iter_mut() {
                                        if span.start > end {
                                            span.start -= end - start;
                                            span.end -= end - start;
                                        } else if span.start > start {
                                            span.start = start;
                                            if span.end > end {
                                                span.end = start;
                                            }
                                        }
                                    }
                                }
                                if let Some(ref mut c) = self.cursor {
                                    c.position = start;
                                }
                                self.selection = None;
                                self.mark_changed();
                                return;
                            }
                        }
                    }
                }
            }
        }
        
        // No selection - delete single character
        self.delete_char(false);
    }
    
    fn on_backspace_word(&mut self) {
        // If there's a selection, delete it
        if let Some(ref selection) = self.selection {
            if let Some(ref cursor) = self.cursor {
                if selection.start.block_id == cursor.block_id && selection.end.block_id == cursor.block_id {
                    let start = selection.start.position.min(selection.end.position);
                    let end = selection.start.position.max(selection.end.position);
                    
                    if let Some(block) = self.document.get_block_mut(&cursor.block_id) {
                        if let Some(block_text) = block.content.get_text_mut() {
                            if end <= block_text.len() {
                                block_text.drain(start..end);
                                // Remove format spans that are now invalid
                                if let BlockContent::Text { ref mut formats, .. } = block.content {
                                    formats.retain(|span| {
                                        !(span.start >= start && span.end <= end) && 
                                        !(span.start < end && span.end > start)
                                    });
                                    // Adjust format span positions
                                    for span in formats.iter_mut() {
                                        if span.start > end {
                                            span.start -= end - start;
                                            span.end -= end - start;
                                        } else if span.start > start {
                                            span.start = start;
                                            if span.end > end {
                                                span.end = start;
                                            }
                                        }
                                    }
                                }
                                if let Some(ref mut c) = self.cursor {
                                    c.position = start;
                                }
                                self.selection = None;
                                self.mark_changed();
                                return;
                            }
                        }
                    }
                }
            }
        }
        
        // No selection - delete word backward
        if let Some(ref mut cursor) = self.cursor {
            let block_id = cursor.block_id.clone();
            let cursor_pos = cursor.position;
            
            if let Some(block) = self.document.get_block_mut(&block_id) {
                if let Some(block_text) = block.content.get_text_mut() {
                    if cursor_pos == 0 {
                        // At start of block, try to merge with previous block
                        return; // Let delete_char handle block merging
                    }
                    
                    let chars: Vec<char> = block_text.chars().collect();
                    let mut new_pos = cursor_pos.min(chars.len());
                    
                    // Skip whitespace backward
                    while new_pos > 0 && chars[new_pos - 1].is_whitespace() {
                        new_pos -= 1;
                    }
                    
                    // Skip word characters backward
                    while new_pos > 0 && !chars[new_pos - 1].is_whitespace() {
                        new_pos -= 1;
                    }
                    
                    // Delete from new_pos to cursor_pos
                    if new_pos < cursor_pos {
                        let mut chars: Vec<char> = block_text.chars().collect();
                        chars.drain(new_pos..cursor_pos);
                        *block_text = chars.into_iter().collect();
                        
                        // Adjust format spans
                        if let BlockContent::Text { ref mut formats, .. } = block.content {
                            let deleted_len = cursor_pos - new_pos;
                            formats.retain(|span| {
                                !(span.start >= new_pos && span.end <= cursor_pos) && 
                                !(span.start < cursor_pos && span.end > new_pos)
                            });
                            // Adjust format span positions
                            for span in formats.iter_mut() {
                                if span.start > cursor_pos {
                                    span.start -= deleted_len;
                                    span.end -= deleted_len;
                                } else if span.start > new_pos {
                                    span.start = new_pos;
                                    if span.end > cursor_pos {
                                        span.end = new_pos;
                                    }
                                }
                            }
                        }
                        
                        cursor.position = new_pos;
                        self.mark_changed();
                    }
                }
            }
        }
    }
    
    fn on_delete(&mut self) {
        // If there's a selection, delete it first
        if let Some(ref selection) = self.selection {
            if let Some(ref cursor) = self.cursor {
                if selection.start.block_id == cursor.block_id && selection.end.block_id == cursor.block_id {
                    let start = selection.start.position.min(selection.end.position);
                    let end = selection.start.position.max(selection.end.position);
                    
                    if let Some(block) = self.document.get_block_mut(&cursor.block_id) {
                        if let Some(block_text) = block.content.get_text_mut() {
                            if end <= block_text.len() {
                                block_text.drain(start..end);
                                // Remove format spans that are now invalid
                                if let BlockContent::Text { ref mut formats, .. } = block.content {
                                    formats.retain(|span| {
                                        !(span.start >= start && span.end <= end) && 
                                        !(span.start < end && span.end > start)
                                    });
                                    // Adjust format span positions
                                    for span in formats.iter_mut() {
                                        if span.start > end {
                                            span.start -= end - start;
                                            span.end -= end - start;
                                        } else if span.start > start {
                                            span.start = start;
                                            if span.end > end {
                                                span.end = start;
                                            }
                                        }
                                    }
                                }
                                if let Some(ref mut c) = self.cursor {
                                    c.position = start;
                                }
                                self.selection = None;
                                self.mark_changed();
                                return;
                            }
                        }
                    }
                }
            }
        }
        
        // No selection - delete single character forward
        self.delete_char(true);
    }
    
    fn on_arrow_left(&mut self, shift: bool, ctrl: bool) {
        if ctrl {
            // Word movement
            self.move_cursor_word_left(shift);
        } else {
            // Character movement
            if shift {
                // Extend selection
                if self.selection_anchor.is_none() {
                    if let Some(ref cursor) = self.cursor {
                        self.selection_anchor = Some(cursor.clone());
                    }
                }
                self.move_cursor(CursorDirection::Left);
                self.update_selection_from_anchor();
            } else {
                // Clear selection and move
                self.clear_selection();
                self.move_cursor(CursorDirection::Left);
            }
        }
    }
    
    fn on_arrow_right(&mut self, shift: bool, ctrl: bool) {
        if ctrl {
            // Word movement
            self.move_cursor_word_right(shift);
        } else {
            // Character movement
            if shift {
                // Extend selection
                if self.selection_anchor.is_none() {
                    if let Some(ref cursor) = self.cursor {
                        self.selection_anchor = Some(cursor.clone());
                    }
                }
                self.move_cursor(CursorDirection::Right);
                self.update_selection_from_anchor();
            } else {
                // Clear selection and move
                self.clear_selection();
                self.move_cursor(CursorDirection::Right);
            }
        }
    }
    
    fn on_arrow_up(&mut self, shift: bool) {
        if shift {
            // Extend selection
            if self.selection_anchor.is_none() {
                if let Some(ref cursor) = self.cursor {
                    self.selection_anchor = Some(cursor.clone());
                }
            }
            self.move_cursor(CursorDirection::Up);
            self.update_selection_from_anchor();
        } else {
            // Clear selection and move
            self.clear_selection();
            self.move_cursor(CursorDirection::Up);
        }
    }
    
    fn on_arrow_down(&mut self, shift: bool) {
        if shift {
            // Extend selection
            if self.selection_anchor.is_none() {
                if let Some(ref cursor) = self.cursor {
                    self.selection_anchor = Some(cursor.clone());
                }
            }
            self.move_cursor(CursorDirection::Down);
            self.update_selection_from_anchor();
        } else {
            // Clear selection and move
            self.clear_selection();
            self.move_cursor(CursorDirection::Down);
        }
    }
    
    fn on_home(&mut self, shift: bool) {
        if shift {
            // Extend selection to start of line
            if self.selection_anchor.is_none() {
                if let Some(ref cursor) = self.cursor {
                    self.selection_anchor = Some(cursor.clone());
                }
            }
            // Move cursor to start of current block
            if let Some(ref mut cursor) = self.cursor {
                cursor.position = 0;
            }
            self.update_selection_from_anchor();
        } else {
            // Clear selection and move to start
            self.clear_selection();
            if let Some(ref mut cursor) = self.cursor {
                cursor.position = 0;
            }
        }
    }
    
    fn on_end(&mut self, shift: bool) {
        if shift {
            // Extend selection to end of line
            if self.selection_anchor.is_none() {
                if let Some(ref cursor) = self.cursor {
                    self.selection_anchor = Some(cursor.clone());
                }
            }
            // Move cursor to end of current block
            if let Some(ref cursor) = self.cursor {
                if let Some(block) = self.document.get_block(&cursor.block_id) {
                    if let Some(text) = block.content.get_text() {
                        if let Some(ref mut c) = self.cursor {
                            c.position = text.len();
                        }
                    }
                }
            }
            self.update_selection_from_anchor();
        } else {
            // Clear selection and move to end
            self.clear_selection();
            if let Some(ref cursor) = self.cursor {
                if let Some(block) = self.document.get_block(&cursor.block_id) {
                    if let Some(text) = block.content.get_text() {
                        if let Some(ref mut c) = self.cursor {
                            c.position = text.len();
                        }
                    }
                }
            }
        }
    }
    
    fn get_cursor_position(&self) -> usize {
        self.cursor.as_ref().map(|c| c.position).unwrap_or(0)
    }
    
    fn is_focused(&self) -> bool {
        self.cursor.is_some()
    }
    
    fn on_mouse_down(&mut self, pos: Vec2, click_count: u32) {
        StylusEditor::on_mouse_down(self, pos, click_count);
    }
    
    fn on_mouse_move(&mut self, pos: Vec2) {
        self.on_mouse_move(pos);
    }
    
    fn on_mouse_up(&mut self) {
        self.on_mouse_up(Vec2::ZERO);
    }
    
    fn contains(&self, pos: Vec2) -> bool {
        self.contains(pos)
    }
    
    fn focus(&mut self) {
        // Focus the first block if no block is focused
        if self.cursor.is_none() && !self.document.blocks.is_empty() {
            let first_block_id = self.document.blocks[0].id.clone();
            self.focus_block(&first_block_id);
        }
    }
    
    fn blur(&mut self) {
        // Clear cursor and selection
        self.cursor = None;
        self.selection = None;
        self.focused_block_id = None;
    }
}

