use glam::{Vec2, Vec4};
use crate::ui::{TextInput, ScrollView, ScrollHit, Dropdown, DropdownItem};
use crate::ui::core::{Rect, layout};
use crate::ui::components::VStack;
use std::cell::RefCell;
use std::collections::HashSet;
use std::collections::HashMap;
use crate::state::shard::Shard;

use serde::{Serialize, Deserialize};

/// Viewport over the constellation (2D world).
/// Screen position = (world - camera) * scale + viewport_center.
#[derive(Debug, Clone)]
pub struct ConstellationView {
    pub position: Vec2,
    pub size: Vec2,
    /// Camera position in world space. Viewport center shows this point.
    pub camera_position: Vec2,
    /// Animated camera position for smooth recenter (lerps toward camera_position).
    pub camera_position_animated: Vec2,
    /// Zoom scale target (1.0 = 1:1). Clamped to [0.25, 2.0]. Updated on wheel.
    pub scale: f32,
    /// Animated scale for smooth zoom (lerps toward scale each frame).
    pub scale_animated: f32,
    /// When panning: screen position where drag started.
    pub pan_drag_start: Option<Vec2>,
}

impl ConstellationView {
    pub const MIN_SCALE: f32 = 0.25;
    pub const MAX_SCALE: f32 = 2.0;

    pub fn new(position: Vec2, size: Vec2) -> Self {
        Self {
            position,
            size,
            camera_position: Vec2::ZERO,
            camera_position_animated: Vec2::ZERO,
            scale: 1.0,
            scale_animated: 1.0,
            pan_drag_start: None,
        }
    }

    pub fn viewport_center(&self) -> Vec2 {
        self.position + self.size * 0.5
    }

    /// World position -> screen position. Uses animated position and scale.
    pub fn world_to_screen(&self, world: Vec2) -> Vec2 {
        (world - self.camera_position_animated) * self.scale_animated + self.viewport_center()
    }

    /// Screen position -> world position.
    pub fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        (screen - self.viewport_center()) / self.scale_animated + self.camera_position_animated
    }

    /// World size -> screen size (for node rendering).
    pub fn world_size_to_screen(&self, world_size: Vec2) -> Vec2 {
        world_size * self.scale_animated
    }

    pub fn contains_screen(&self, screen: Vec2) -> bool {
        screen.x >= self.position.x
            && screen.x <= self.position.x + self.size.x
            && screen.y >= self.position.y
            && screen.y <= self.position.y + self.size.y
    }

    /// Center the camera on the given world position (so that point is at viewport center).
    /// Sets target; actual position animates smoothly via update().
    pub fn center_camera_on(&mut self, world_pos: Vec2) {
        self.camera_position = world_pos;
    }

    /// Reset zoom to 1:1 (e.g. when refocusing on a shard after pan/zoom).
    /// Sets target only; scale_animated lerps smoothly in update(dt).
    pub fn reset_zoom_to_normal(&mut self) {
        self.scale = 1.0;
    }

    /// Pan the camera by a delta in screen space. Converts to world space using scale.
    pub fn pan(&mut self, screen_delta: Vec2) {
        let world_delta = screen_delta / self.scale_animated;
        self.camera_position -= world_delta;
        self.camera_position_animated -= world_delta;
    }

    /// Zoom by factor (e.g. 1.1 for in, 0.9 for out). Clamped. Updates target; scale_animated lerps in update().
    pub fn zoom(&mut self, factor: f32) {
        self.scale = (self.scale * factor).clamp(Self::MIN_SCALE, Self::MAX_SCALE);
    }

    /// Fit the given world-space AABB in view with margin. Sets camera and scale.
    pub fn fit_in_view(&mut self, min: Vec2, max: Vec2, margin: f32) {
        let size = max - min;
        if size.x <= 0.0 || size.y <= 0.0 {
            return;
        }
        let center = (min + max) * 0.5;
        self.camera_position = center;
        self.camera_position_animated = center;
        let avail_w = (self.size.x - margin * 2.0).max(1.0);
        let avail_h = (self.size.y - margin * 2.0).max(1.0);
        let scale_x = avail_w / size.x;
        let scale_y = avail_h / size.y;
        self.scale = scale_x.min(scale_y).clamp(Self::MIN_SCALE, Self::MAX_SCALE);
        self.scale_animated = self.scale;
    }

    /// Smooth camera and zoom animation toward targets. Call each frame with dt.
    pub fn update(&mut self, dt: f32) {
        const LERP_SPEED: f32 = 8.0;
        let t = (LERP_SPEED * dt).min(1.0);
        self.camera_position_animated = self.camera_position_animated.lerp(self.camera_position, t);
        self.scale_animated = self.scale_animated + (self.scale - self.scale_animated) * t;
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ActionButtonType {
    Edit,
    Delete,
    Mute,
    AddNote,
}

/// Visual wrapper around Shard for rendering purposes
/// Maintains backward compatibility with old serialization format
#[derive(Clone, Debug)]
pub struct ChatMessage {
    /// Reference to the underlying shard ID
    /// If None, this is a legacy message that hasn't been migrated yet
    pub shard_id: Option<String>,
    
    /// Legacy fields - kept for backward compatibility and direct access
    /// These are computed from shard when shard_id is Some
    pub role: MessageRole,
    pub content: String,
    pub contexts: Vec<String>,
    pub citations: Vec<Citation>,
    /// User-attached notes (comments) on this message
    pub notes: Vec<String>,
}

impl ChatMessage {
    /// Create a ChatMessage from a Shard
    pub fn from_shard(shard: &Shard) -> Self {
        Self {
            shard_id: Some(shard.id.clone()),
            role: shard.metadata.role.clone(),
            content: shard.text.clone(),
            contexts: shard.metadata.contexts.clone(),
            citations: shard.metadata.citations.clone(),
            notes: shard.metadata.notes.clone(),
        }
    }
    
    /// Create a ChatMessage from legacy data (for migration)
    pub fn from_legacy(role: MessageRole, content: String, contexts: Vec<String>, citations: Vec<Citation>) -> Self {
        Self {
            shard_id: None,
            role,
            content,
            contexts,
            citations,
            notes: Vec::new(),
        }
    }
    
    /// Convert this ChatMessage to a Shard
    /// If shard_id is None, creates a new shard
    pub fn to_shard(&self) -> Shard {
        if let Some(ref id) = self.shard_id {
            // If we have a shard_id, we should look it up from state
            // For now, create a new shard with the same data
            let mut shard = Shard::new(self.content.clone(), self.role.clone());
            shard.id = id.clone();
            shard.metadata.contexts = self.contexts.clone();
            shard.metadata.citations = self.citations.clone();
            shard.metadata.notes = self.notes.clone();
            shard
        } else {
            // Legacy message - create new shard
            let mut shard = Shard::new(self.content.clone(), self.role.clone());
            shard.metadata.contexts = self.contexts.clone();
            shard.metadata.citations = self.citations.clone();
            shard.metadata.notes = self.notes.clone();
            shard
        }
    }
    
    /// Check if this is a legacy message (not yet migrated to shard)
    pub fn is_legacy(&self) -> bool {
        self.shard_id.is_none()
    }
}

// Custom serialization for backward compatibility
impl Serialize for ChatMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ChatMessage", 5)?;
        state.serialize_field("role", &self.role)?;
        state.serialize_field("content", &self.content)?;
        state.serialize_field("contexts", &self.contexts)?;
        state.serialize_field("citations", &self.citations)?;
        state.serialize_field("notes", &self.notes)?;
        state.end()
    }
}

// Custom deserialization for backward compatibility
impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ChatMessageHelper {
            role: MessageRole,
            content: String,
            contexts: Option<Vec<String>>,
            citations: Option<Vec<Citation>>,
            notes: Option<Vec<String>>,
        }
        
        let helper = ChatMessageHelper::deserialize(deserializer)?;
        Ok(ChatMessage {
            shard_id: None, // Legacy messages don't have shard_id
            role: helper.role,
            content: helper.content,
            contexts: helper.contexts.unwrap_or_default(),
            citations: helper.citations.unwrap_or_default(),
            notes: helper.notes.unwrap_or_default(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Citation {
    pub text: String,
    pub source: String,
    pub title: Option<String>,
    pub year: Option<String>,
    pub section: Option<String>,
    pub page: Option<u32>,
}

pub struct ChatWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub message_list: ScrollView,
    /// Constellation view (2D world + camera). Used when graph is active.
    pub constellation_view: ConstellationView,
    pub input_field: TextInput,
    pub send_button_position: Vec2,
    pub send_button_size: Vec2,
    pub context_pool_button_position: Vec2,
    pub context_pool_button_size: Vec2,
    pub context_pool_dropdown: Dropdown,
    pub messages: Vec<ChatMessage>,
    pub selected_collection_id: Option<i32>,
    pub editing_message_idx: Option<usize>,
    pub edit_textarea: TextInput,  // Textarea for editing messages
    pub delete_confirm_idx: Option<usize>,
    pub muted_messages: HashSet<usize>,
    pub is_sending: bool,
    pub highlight_term: Option<String>,
    pub citations_expanded: HashSet<usize>,  // Track which messages have expanded citations
    /// When Some(msg_idx), user is adding a note to that message; note_input holds the text.
    pub adding_note_msg_idx: Option<usize>,
    /// Text input for adding or editing a note (positioned when adding_note_msg_idx or editing_note is set).
    pub note_input: TextInput,
    /// When Some((msg_idx, note_idx)), user is editing that note; note_input holds the text.
    pub editing_note: Option<(usize, usize)>,
    /// Cached (user_size, assistant_size) per node id from last render; used for hit test to match render layout.
    pub constellation_layout_cache: RefCell<Option<HashMap<String, (Vec2, Vec2)>>>,
    /// Per-node per-bubble scroll (user, assistant) for constellation; only text inside bubbles scrolls.
    pub constellation_scroll_offsets: RefCell<HashMap<String, (f32, f32)>>,
    /// Target scroll for smooth lerp each frame (user, assistant) per node.
    pub constellation_scroll_targets: RefCell<HashMap<String, (f32, f32)>>,
}

impl ChatWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        use crate::ui::style;
        
        let input_container_height = 60.0;  // Total height including padding
        let padding = style::padding::MEDIUM;
        
        // Input and button will be positioned relative to viewport bottom
        // We'll update their positions in update_layout()
        let input_width = size.x - 100.0 - padding * 2.0;
        let input = TextInput::new(
            Vec2::new(position.x + padding, 0.0),  // Will be updated
            Vec2::new(input_width, style::input_height::NORMAL),
        );

        let send_button_size = Vec2::new(80.0, style::button_height::NORMAL);
        let send_button_position = Vec2::new(
            position.x + size.x - send_button_size.x - padding,
            0.0,  // Will be updated
        );

        let context_pool_button_size = Vec2::new(40.0, style::button_height::NORMAL);
        let context_pool_button_position = Vec2::new(
            position.x + padding,
            0.0,  // Will be updated
        );

        let mut context_pool_dropdown = Dropdown::new(
            context_pool_button_position,
            context_pool_button_size,
        );
        // Initialize with "All papers" option
        context_pool_dropdown.items.push(DropdownItem {
            id: None,
            label: "All papers".to_string(),
        });

        let message_list_height = size.y - input_container_height - padding;
        let message_list = ScrollView::new(
            Vec2::new(position.x, position.y + padding),
            Vec2::new(size.x, message_list_height),
        );
        let constellation_view = ConstellationView::new(
            Vec2::new(position.x, position.y + padding),
            Vec2::new(size.x, message_list_height),
        );

        // Edit textarea (initially hidden, positioned dynamically when editing)
        let edit_textarea = TextInput::new(
            Vec2::new(0.0, 0.0),  // Will be positioned when editing
            Vec2::new(400.0, 100.0),  // Default size, will be adjusted
        );
        let note_input = TextInput::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(300.0, style::input_height::NORMAL),
        );

        let mut window = Self {
            position,
            size,
            message_list,
            constellation_view,
            input_field: input,
            send_button_position,
            send_button_size,
            context_pool_button_position,
            context_pool_button_size,
            context_pool_dropdown,
            messages: Vec::new(),
            selected_collection_id: None,
            editing_message_idx: None,
            edit_textarea,
            delete_confirm_idx: None,
            muted_messages: HashSet::new(),
            is_sending: false,
            highlight_term: None,
            citations_expanded: HashSet::new(),
            adding_note_msg_idx: None,
            note_input,
            editing_note: None,
            constellation_layout_cache: RefCell::new(None),
            constellation_scroll_offsets: RefCell::new(HashMap::new()),
            constellation_scroll_targets: RefCell::new(HashMap::new()),
        };
        
        window.update_layout();
        window
    }

    pub fn update_layout(&mut self) {
        use crate::ui::style;
        
        let padding = style::padding::MEDIUM;
        // Calculate input container height from input height + padding on both sides
        let input_container_height = style::input_height::NORMAL + padding * 2.0;
        
        // Position input and button at bottom of chat window (which is at bottom of viewport)
        let input_y = self.position.y + self.size.y - input_container_height;
        
        // Use horizontal stack to position context pool button, input, and send button
        use crate::ui::core::{layout, Rect};
        let input_area = Rect::new(
            self.position.x + padding,
            input_y + padding,
            self.size.x - padding * 2.0,
            style::input_height::NORMAL,
        );
        
        // Calculate widths for horizontal stack
        let input_width = input_area.width - self.context_pool_button_size.x - self.send_button_size.x - padding * 2.0;
        let component_widths = [
            self.context_pool_button_size.x,
            input_width,
            self.send_button_size.x,
        ];
        
        // Use stack_horizontal to position all three components
        let component_rects = layout::stack_horizontal(
            &input_area,
            &component_widths,
            padding,
            0.0,
        );
        
        // Set positions from layout
        if let Some(context_pool_rect) = component_rects.get(0) {
            self.context_pool_button_position = context_pool_rect.position();
            // Update dropdown anchor rect to button position
            self.context_pool_dropdown.anchor_rect = *context_pool_rect;
            // Update dropdown layout if it's open
            if self.context_pool_dropdown.is_open {
                self.context_pool_dropdown.update_layout();
            }
        }
        
        // Input field position
        if let Some(input_rect) = component_rects.get(1) {
            self.input_field.position = input_rect.position();
            self.input_field.size.x = input_rect.width;
        }
        
        // Send button position
        if let Some(send_rect) = component_rects.get(2) {
            self.send_button_position = send_rect.position();
        }
        
        // Update message list and constellation view height (leave space for input at bottom)
        let message_list_height = self.size.y - input_container_height - padding;
        self.message_list.size.y = message_list_height;
        self.message_list.position = Vec2::new(self.position.x, self.position.y + padding);
        self.constellation_view.position = Vec2::new(self.position.x, self.position.y + padding);
        self.constellation_view.size = Vec2::new(self.size.x, message_list_height);
    }

    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
        // Update content height when message is added
        self.update_content_height(|text, size| {
            // Simple approximation - in real usage, this would use the renderer's measure_text
            Vec2::new(text.len() as f32 * size * 0.6, size)
        });
        self.message_list.scroll_to_bottom();
    }

    pub fn update_content_height(&mut self, measure_text_fn: impl Fn(&str, f32) -> Vec2) {
        let message_spacing = 16.0;
        let padding = 12.0;
        let max_bubble_width = self.size.x * 0.7;
        const FONT_SIZE: f32 = 16.0;
        let line_height = FONT_SIZE * 1.2;
        
        let mut total_height = padding;  // Top padding

        for msg in &self.messages {
            // Calculate actual wrapped text height using the same logic as get_message_bubbles
            let words: Vec<&str> = msg.content.split_whitespace().collect();
            let mut current_line = String::new();
            let mut line_count = 0;
            
            for word in words {
                let test_line = if current_line.is_empty() {
                    word.to_string()
                } else {
                    format!("{} {}", current_line, word)
                };
                
                let test_line_size = measure_text_fn(&test_line, FONT_SIZE);
                
                if test_line_size.x > max_bubble_width && !current_line.is_empty() {
                    line_count += 1;
                    current_line = word.to_string();
                } else {
                    current_line = test_line;
                }
            }
            
            if !current_line.is_empty() {
                line_count += 1;
            }
            
            let mut bubble_height = if line_count == 0 {
                line_height + (padding * 2.0)
            } else {
                (line_count as f32 * line_height) + (padding * 2.0)
            };
            // Approximate space for citations (collapsed) and notes and action buttons
            const NOTE_LINE_H: f32 = 18.0;
            bubble_height += (msg.notes.len() as f32 * NOTE_LINE_H) + 20.0 + 25.0; // notes + pin area + action row
            
            total_height += bubble_height + message_spacing;
        }

        total_height += padding;  // Bottom padding
        self.message_list.set_content_height(total_height);
    }

    pub fn get_message_bubbles(&self, mut measure_text_fn: impl FnMut(&str, f32) -> Vec2) -> Vec<MessageBubble> {
        let padding = 12.0;
        let message_spacing = 16.0;
        let bubble_margin = 8.0; // Small margin around each bubble
        let max_bubble_width = self.size.x * 0.7;
        const FONT_SIZE: f32 = 16.0;
        let line_height = FONT_SIZE * 1.2; // Match text_line_height
        let scroll_offset = self.message_list.scroll_offset;
        
        // Start from message list position (not chat window position)
        let message_list_top = self.message_list.position.y;
        
        let mut bubbles = Vec::new();
        let mut y_offset = padding - scroll_offset;

        for msg in &self.messages {
            // Create VStack for message bubble content to use wrap_content()
            let mut bubble_content = VStack::new(0.0, padding);
            
            // Add message text to VStack
            use crate::ui::text::TextAlignment;
            use crate::ui::style;
            bubble_content.add_text_styled(
                &msg.content,
                FONT_SIZE,
                style::text::PRIMARY,
                TextAlignment::Left,
            );
            
            // Add citations if present (as text items in VStack)
            let is_citations_expanded = self.citations_expanded.contains(&bubbles.len());
            if !msg.citations.is_empty() {
                if is_citations_expanded {
                    // Add each citation as a text item
                    for (i, citation) in msg.citations.iter().enumerate() {
                        let mut citation_text = format!("[{}] ", i + 1);
                        if let Some(ref title) = citation.title {
                            citation_text.push_str(title);
                        }
                        citation_text.push_str(" (");
                        citation_text.push_str(&citation.source);
                        if let Some(ref year) = citation.year {
                            citation_text.push_str(", ");
                            citation_text.push_str(year);
                        }
                        citation_text.push(')');
                        if let Some(ref section) = citation.section {
                            citation_text.push_str(" – ");
                            citation_text.push_str(section);
                        }
                        if let Some(page) = citation.page {
                            citation_text.push_str(&format!(", p.{}", page));
                        }
                        bubble_content.add_text_styled(
                            &citation_text,
                            FONT_SIZE * 0.85,
                            style::text::SECONDARY,
                            TextAlignment::Left,
                        );
                    }
                } else {
                    // Add collapsed "Sources" summary
                    let citation_text = format!("Sources ({})", msg.citations.len());
                    bubble_content.add_text_styled(
                        &citation_text,
                        FONT_SIZE * 0.85,
                        style::text::SECONDARY,
                        TextAlignment::Left,
                    );
                }
            }
            // Add notes (user comments on this message)
            for note in &msg.notes {
                bubble_content.add_text_styled(
                    &format!("• {}", note),
                    FONT_SIZE * 0.85,
                    style::text::SECONDARY,
                    TextAlignment::Left,
                );
            }
            
            // Compute bubble content size using wrap_content with word wrapping
            // Available width for content (accounting for padding)
            let content_max_width = max_bubble_width - padding * 2.0;
            // Create and use closure sequentially to avoid multiple borrows
            let content_size = {
                let mut measure_wrapper: Box<dyn FnMut(&str, f32) -> Vec2> = Box::new(|text: &str, font_size: f32| -> Vec2 {
                    measure_text_fn(text, font_size)
                });
                bubble_content.wrap_content(
                    Some(content_max_width),
                    Some(measure_wrapper.as_mut())
                )
            };
            
            // Calculate bubble dimensions from content size
            let bubble_width = content_size.x.max(padding * 2.0 + FONT_SIZE); // Minimum width
            let mut bubble_height = content_size.y;
            
            // Citations are already included in bubble_content, so no need to add extra space
            
            // Add space for pin button (only for assistant messages)
            let pin_button_size = Vec2::new(20.0, 20.0);
            let pin_button_padding = 5.0;
            if matches!(msg.role, MessageRole::Assistant) {
                bubble_height += pin_button_size.y + pin_button_padding;
            }
            
            // Add space for action buttons at bottom (Edit, Add note, Mute, Delete)
            let action_button_size = Vec2::new(20.0, 20.0);
            let action_button_spacing = 5.0;
            const NOTE_LINE_H: f32 = 18.0;
            bubble_height += action_button_size.y + action_button_spacing;
            
            // Determine bubble position based on role
            let bubble_x = match msg.role {
                MessageRole::User => {
                    // Right-aligned with margin
                    self.position.x + self.size.x - bubble_width - padding - bubble_margin
                }
                MessageRole::Assistant => {
                    // Left-aligned with margin
                    self.position.x + padding + bubble_margin
                }
            };

            // Calculate citation positions using layout functions
            // Citations are now part of the VStack, so we need to compute their positions
            // based on the text content layout
            let mut citation_positions = Vec::new();
            let is_citations_expanded = self.citations_expanded.contains(&bubbles.len());
            if !msg.citations.is_empty() {
                let citation_item_height = 20.0;
                let citation_spacing = 0.0;
                let citation_padding = padding;
                
                // Calculate where citations start (after message text)
                // Create a VStack with just the message text to measure its height
                let mut message_only = VStack::new(0.0, padding);
                message_only.add_text_styled(
                    &msg.content,
                    FONT_SIZE,
                    style::text::PRIMARY,
                    TextAlignment::Left,
                );
                // Create and use closure sequentially to avoid multiple borrows
                let message_text_size = {
                    let mut measure_wrapper: Box<dyn FnMut(&str, f32) -> Vec2> = Box::new(|text: &str, font_size: f32| -> Vec2 {
                        measure_text_fn(text, font_size)
                    });
                    message_only.wrap_content(
                        Some(content_max_width),
                        Some(measure_wrapper.as_mut())
                    )
                };
                let citation_start_y = message_list_top + y_offset + message_text_size.y;
                
                // Create rect for citations area
                let citations_area = Rect::new(
                    bubble_x + citation_padding,
                    citation_start_y,
                    bubble_width - citation_padding * 2.0,
                    if is_citations_expanded {
                        msg.citations.len() as f32 * citation_item_height
                    } else {
                        citation_item_height
                    },
                );
                
                if is_citations_expanded {
                    // Show all citations with full details using stack_vertical
                    let citation_heights: Vec<f32> = (0..msg.citations.len()).map(|_| citation_item_height).collect();
                    let citation_rects = layout::stack_vertical(&citations_area, &citation_heights, citation_spacing, 0.0);
                    
                    for (i, citation) in msg.citations.iter().enumerate() {
                        if let Some(citation_rect) = citation_rects.get(i) {
                            // Format: Title (Source, Year) – Section, p.Page
                            let mut citation_text = format!("[{}] ", i + 1);
                            if let Some(ref title) = citation.title {
                                citation_text.push_str(title);
                            }
                            citation_text.push_str(" (");
                            citation_text.push_str(&citation.source);
                            if let Some(ref year) = citation.year {
                                citation_text.push_str(", ");
                                citation_text.push_str(year);
                            }
                            citation_text.push(')');
                            if let Some(ref section) = citation.section {
                                citation_text.push_str(" – ");
                                citation_text.push_str(section);
                            }
                            if let Some(page) = citation.page {
                                citation_text.push_str(&format!(", p.{}", page));
                            }
                            let citation_width = measure_text_fn(&citation_text, FONT_SIZE * 0.85).x;
                            citation_positions.push((
                                citation_rect.position(),
                                Vec2::new(citation_width + 25.0, citation_item_height), // Extra space for magnify icon
                                i,
                            ));
                        }
                    }
                } else {
                    // Show collapsed "Sources" summary
                    let citation_text = format!("Sources ({})", msg.citations.len());
                    let citation_width = measure_text_fn(&citation_text, FONT_SIZE * 0.85).x;
                    citation_positions.push((
                        citations_area.position(),
                        Vec2::new(citation_width + 25.0, citation_item_height),
                        0, // Use 0 as index for summary
                    ));
                }
            }
            
            // Calculate pin button position (top-right of assistant messages) using layout functions
            let pin_button_position = match msg.role {
                MessageRole::Assistant => {
                    let bubble_rect = Rect::new(
                        bubble_x,
                        message_list_top + y_offset,
                        bubble_width,
                        bubble_height,
                    );
                    Some(Vec2::new(
                        layout::align_right(&bubble_rect, pin_button_size.x, pin_button_padding),
                        bubble_rect.y + pin_button_padding,
                    ))
                }
                MessageRole::User => None,
            };
            
            // Calculate action buttons (edit, delete, mute) - positioned at bottom of bubble using layout functions
            let action_button_size = Vec2::new(20.0, 20.0);
            let action_button_spacing = 5.0;
            let action_buttons_padding = 5.0;
            
            // Create rect for action buttons area at bottom of bubble
            let action_buttons_area = Rect::new(
                bubble_x + padding,
                message_list_top + y_offset + bubble_height - action_button_size.y - action_buttons_padding,
                bubble_width - padding * 2.0,
                action_button_size.y,
            );
            
            // Four buttons: Edit, Add note, Mute, Delete
            let button_widths = [
                action_button_size.x,
                action_button_size.x,
                action_button_size.x,
                action_button_size.x,
            ];
            let button_rects = layout::stack_horizontal(&action_buttons_area, &button_widths, action_button_spacing, 0.0);
            
            let edit_button_position = button_rects.get(0).map(|r| r.position());
            let add_note_button_position = button_rects.get(1).map(|r| r.position());
            let mute_button_position = button_rects.get(2).map(|r| r.position());
            let delete_button_position = button_rects.get(3).map(|r| r.position());

            // Note remove and edit button positions (right side of bubble, one per note)
            let note_start_y = message_list_top + y_offset + bubble_height
                - action_button_size.y - action_buttons_padding
                - (msg.notes.len() as f32 * NOTE_LINE_H);
            let note_remove_positions: Vec<(Vec2, Vec2, usize)> = msg.notes.iter().enumerate()
                .map(|(i, _)| (
                    Vec2::new(bubble_x + bubble_width - 24.0, note_start_y + i as f32 * NOTE_LINE_H),
                    Vec2::new(20.0, NOTE_LINE_H),
                    i,
                ))
                .collect();
            let note_edit_positions: Vec<(Vec2, Vec2, usize)> = msg.notes.iter().enumerate()
                .map(|(i, _)| (
                    Vec2::new(bubble_x + bubble_width - 48.0, note_start_y + i as f32 * NOTE_LINE_H),
                    Vec2::new(20.0, NOTE_LINE_H),
                    i,
                ))
                .collect();

            let msg_idx = bubbles.len();
            let is_muted = self.muted_messages.contains(&msg_idx);
            
            bubbles.push(MessageBubble {
                position: Vec2::new(bubble_x, message_list_top + y_offset),
                size: Vec2::new(bubble_width, bubble_height),
                role: msg.role.clone(),
                content: msg.content.clone(),
                text_start_y: message_list_top + y_offset + padding,
                message: Some(msg.clone()),
                citation_positions,
                pin_button_position,
                pin_button_size,
                is_muted,
                message_idx: msg_idx,
                edit_button_position,
                delete_button_position,
                mute_button_position,
                action_button_size,
                add_note_button_position,
                notes: msg.notes.clone(),
                note_remove_positions,
                note_edit_positions,
            });

            y_offset += bubble_height + message_spacing + bubble_margin;
        }

        // Note: Content height should be updated separately via update_content_height
        // We don't modify self here to allow this to work with &self

        bubbles
    }

    pub fn hit_test(&mut self, pos: Vec2, graph_state: &crate::state::GraphState) -> ChatHit {
        if self.input_field.contains(pos) {
            return ChatHit::Input;
        }

        // Check context pool dropdown menu first (if open)
        if self.context_pool_dropdown.is_open {
            use crate::ui::core::Rect;
            
            // Check "Create new collection" button first (more specific)
            let menu_padding = 10.0;
            let item_height = 30.0;
            let menu_spacing = 5.0;
            let create_button_y = self.context_pool_dropdown.menu_rect.y + menu_padding + 
                (self.context_pool_dropdown.items.len() as f32 * item_height) + menu_spacing;
            let create_button_rect = Rect::new(
                self.context_pool_dropdown.menu_rect.x + menu_padding,
                create_button_y,
                self.context_pool_dropdown.menu_rect.width - menu_padding * 2.0,
                30.0,
            );
            
            if create_button_rect.contains_point(pos) {
                return ChatHit::ContextPoolCreate;
            }
            
            // Check menu items
            if self.context_pool_dropdown.contains_menu(pos) {
                if let Some(index) = self.context_pool_dropdown.get_menu_item_at(pos) {
                    return ChatHit::ContextPoolItem(index);
                }
                return ChatHit::ContextPoolMenu;
            }
        }

        // Check context pool button
        if self.context_pool_dropdown.contains(pos) {
            return ChatHit::ContextPoolButton;
        }

        let send_button_rect = (
            self.send_button_position,
            self.send_button_position + self.send_button_size,
        );
        if pos.x >= send_button_rect.0.x && pos.x <= send_button_rect.1.x &&
           pos.y >= send_button_rect.0.y && pos.y <= send_button_rect.1.y {
            return ChatHit::SendButton;
        }

        // Constellation view: when graph is active, hit-test edit/note inputs first, then nodes
        if graph_state.graph_id.is_some() && self.constellation_view.contains_screen(pos) {
            if self.editing_message_idx.is_some() && self.edit_textarea.contains(pos) {
                return ChatHit::ConstellationEditTextarea;
            }
            if (self.adding_note_msg_idx.is_some() || self.editing_note.is_some()) && self.note_input.contains(pos) {
                return ChatHit::ConstellationNoteInput;
            }
            let world = self.constellation_view.screen_to_world(pos);
            let scale = self.constellation_view.scale_animated;
            let pad = 8.0f32 * scale;
            let row_pad = 6.0f32 * scale;
            let btn = 18.0f32 * scale;
            let btn_space = 4.0f32 * scale;
            let msg_btn = 14.0f32 * scale;
            let bubble_spacing = 6.0f32 * scale;
            let citation_line_height = (14.4f32) * scale;
            let citation_gap = 4.0f32 * scale;
            let max_content_base = 80.0f32 * scale;
            let cache = self.constellation_layout_cache.borrow();
            let cache = cache.as_ref();
            // Check nodes (last matching node wins = drawn on top)
            let mut hit_id = None;
            let mut hit_button = None;
            let mut node_user_content_rect: Option<Rect> = None;
            let mut node_assistant_content_rect: Option<Rect> = None;
            for (id, node) in &graph_state.nodes {
                let r0 = node.position;
                let r1 = node.position + node.size;
                if world.x >= r0.x && world.x <= r1.x && world.y >= r0.y && world.y <= r1.y {
                    hit_id = Some(id.clone());
                    hit_button = None;
                    let screen_pos = self.constellation_view.world_to_screen(node.position);
                    let screen_size = self.constellation_view.world_size_to_screen(node.size);

                    // Fixed layout (no scroll) for hit test; only bubble content areas are scrollable.
                    let max_content_width = (screen_size.x * 0.7 - pad * 2.0).max(max_content_base);
                    let (user_size, assistant_size) = match cache.and_then(|c| c.get(id)) {
                        Some(&(u, a)) => (u, a),
                        None => {
                            let mut measure = |t: &str, s: f32| -> Vec2 {
                                Vec2::new(t.chars().count() as f32 * s * 0.6, s * 1.2)
                            };
                            let font_size = (16.0f32 * scale).max(8.0);
                            let u = node.shard.user_content.as_deref()
                                .filter(|s| !s.is_empty())
                                .map(|u| Self::measure_wrapped_block(&mut measure, u, max_content_width, font_size))
                                .unwrap_or(Vec2::ZERO);
                            let a = node.shard.assistant_content.as_deref()
                                .filter(|s| !s.is_empty())
                                .map(|a| Self::measure_wrapped_block(&mut measure, a, max_content_width, font_size))
                                .unwrap_or(Vec2::ZERO);
                            (u, a)
                        }
                    };
                    // Minimum bubble size when node has content so hide/unhide button always has a hit rect (matches render placeholder)
                    const BUTTON_ROW_RESERVE: f32 = 22.0;
                    let button_reserve = BUTTON_ROW_RESERVE * scale;
                    let min_content_h = 20.0f32 * scale;
                    let min_content_w = max_content_base;
                    let has_user_bubble = node.shard.user_content.as_deref().map_or(false, |s| !s.is_empty());
                    let has_assistant_bubble = node.shard.assistant_content.as_deref().map_or(false, |s| !s.is_empty());
                    let user_size_eff = if has_user_bubble {
                        Vec2::new(user_size.x.max(min_content_w), user_size.y.max(min_content_h))
                    } else {
                        user_size
                    };
                    let assistant_size_eff = if has_assistant_bubble {
                        Vec2::new(assistant_size.x.max(min_content_w), assistant_size.y.max(min_content_h))
                    } else {
                        assistant_size
                    };
                    let mut y = screen_pos.y + pad;
                    let mut hit_user_content_rect: Option<Rect> = None;
                    let mut hit_assistant_content_rect: Option<Rect> = None;
                    if user_size_eff.x > 0.0 && user_size_eff.y > 0.0 {
                        let bubble_w = user_size_eff.x + pad * 2.0;
                        let bubble_h = user_size_eff.y + pad * 2.0 + button_reserve;
                        let bubble_x = screen_pos.x + screen_size.x - pad - bubble_w;
                        let bubble_y = y;
                        hit_user_content_rect = Some(Rect::new(
                            bubble_x + pad,
                            bubble_y + pad,
                            (bubble_w - pad * 2.0).max(0.0),
                            (bubble_h - pad * 2.0 - button_reserve).max(0.0),
                        ));
                        let hit_expand = 2.0 * scale;
                        let edit_rect = Rect::new(bubble_x + bubble_w - msg_btn * 2.0 - 4.0 * scale - hit_expand, y + bubble_h - msg_btn - 4.0 * scale - hit_expand, msg_btn + hit_expand * 2.0, msg_btn + hit_expand * 2.0);
                        let hide_rect = Rect::new(bubble_x + bubble_w - msg_btn - 4.0 * scale - hit_expand, y + bubble_h - msg_btn - 4.0 * scale - hit_expand, msg_btn + hit_expand * 2.0, msg_btn + hit_expand * 2.0);
                        if edit_rect.contains_point(pos) {
                            return ChatHit::ConstellationMessageEditButton(id.clone(), MessagePart::User);
                        }
                        if hide_rect.contains_point(pos) {
                            return ChatHit::ConstellationMessageHideButton(id.clone(), MessagePart::User);
                        }
                        y += bubble_h + bubble_spacing;
                        node_user_content_rect = hit_user_content_rect;
                    }
                    let is_editing_assistant = self.messages.iter().position(|m| m.shard_id.as_deref() == Some(id.as_str()))
                        .map(|idx| self.editing_message_idx == Some(idx))
                        .unwrap_or(false);
                    let (ast_bubble_w, ast_bubble_h) = if is_editing_assistant {
                        (self.edit_textarea.size.x + pad * 2.0, self.edit_textarea.size.y + pad * 2.0)
                    } else if assistant_size_eff.x > 0.0 && assistant_size_eff.y > 0.0 {
                        let cit_h = if node.shard.citations.is_empty() { 0.0 } else { citation_gap + node.shard.citations.len() as f32 * citation_line_height };
                        (assistant_size_eff.x + pad * 2.0, assistant_size_eff.y + pad * 2.0 + cit_h + button_reserve)
                    } else {
                        (0.0, 0.0)
                    };
                    if ast_bubble_w > 0.0 && ast_bubble_h > 0.0 {
                        let bubble_w = ast_bubble_w;
                        let bubble_h = ast_bubble_h;
                        let bubble_x = screen_pos.x + pad;
                        let bubble_y = y;
                        let hit_expand = 2.0 * scale;
                        let edit_rect = Rect::new(bubble_x + bubble_w - msg_btn * 2.0 - 4.0 * scale - hit_expand, bubble_y + bubble_h - msg_btn - 4.0 * scale - hit_expand, msg_btn + hit_expand * 2.0, msg_btn + hit_expand * 2.0);
                        let hide_rect = Rect::new(bubble_x + bubble_w - msg_btn - 4.0 * scale - hit_expand, bubble_y + bubble_h - msg_btn - 4.0 * scale - hit_expand, msg_btn + hit_expand * 2.0, msg_btn + hit_expand * 2.0);
                        if edit_rect.contains_point(pos) {
                            return ChatHit::ConstellationMessageEditButton(id.clone(), MessagePart::Assistant);
                        }
                        if hide_rect.contains_point(pos) {
                            return ChatHit::ConstellationMessageHideButton(id.clone(), MessagePart::Assistant);
                        }
                        hit_assistant_content_rect = Some(Rect::new(
                            bubble_x + pad,
                            bubble_y + pad,
                            (bubble_w - pad * 2.0).max(0.0),
                            (bubble_h - pad * 2.0 - button_reserve).max(0.0),
                        ));
                        node_assistant_content_rect = hit_assistant_content_rect;
                    }
                    // Citation regions (same layout as render_constellation)
                    if !is_editing_assistant && assistant_size_eff.x > 0.0 && assistant_size_eff.y > 0.0 && !node.shard.citations.is_empty() {
                        let bubble_x = screen_pos.x + pad;
                        let bubble_y = y;
                        let bubble_w = assistant_size_eff.x + pad * 2.0;
                        let text_start_y = bubble_y + pad;
                        for (citation_idx, _) in node.shard.citations.iter().enumerate() {
                            let citation_y = text_start_y + assistant_size_eff.y + citation_gap + citation_idx as f32 * citation_line_height;
                            let rect = Rect::new(bubble_x + pad, citation_y, bubble_w - pad * 2.0, citation_line_height);
                            if rect.contains_point(pos) {
                                return ChatHit::ConstellationCitation(id.clone(), citation_idx);
                            }
                        }
                    }
                    // Note edit/remove (same layout as render_constellation)
                    let note_line_h = 18.0f32 * scale;
                    let notes_gap = 4.0f32 * scale;
                    if !is_editing_assistant && assistant_size_eff.x > 0.0 && assistant_size_eff.y > 0.0 && !node.shard.notes.is_empty() {
                        let citations_height = if node.shard.citations.is_empty() {
                            0.0
                        } else {
                            citation_gap + node.shard.citations.len() as f32 * citation_line_height
                        };
                        let ast_bubble_h = assistant_size_eff.y + pad * 2.0 + citations_height;
                        let assistant_bottom_y = y + ast_bubble_h + bubble_spacing;
                        let notes_start_y = assistant_bottom_y + notes_gap;
                        for (note_idx, _) in node.shard.notes.iter().enumerate() {
                            let line_y = notes_start_y + note_idx as f32 * note_line_h;
                            let edit_w = 20.0f32 * scale;
                            let edit_rect = Rect::new(screen_pos.x + screen_size.x - pad - 48.0 * scale, line_y, edit_w, note_line_h);
                            let remove_rect = Rect::new(screen_pos.x + screen_size.x - pad - 24.0 * scale, line_y, edit_w, note_line_h);
                            if edit_rect.contains_point(pos) {
                                return ChatHit::ConstellationEditNote(id.clone(), note_idx);
                            }
                            if remove_rect.contains_point(pos) {
                                return ChatHit::ConstellationRemoveNote(id.clone(), note_idx);
                            }
                        }
                    }
                    // Pin button (top-right)
                    let pin_rect = Rect::new(
                        screen_pos.x + screen_size.x - row_pad - btn,
                        screen_pos.y + row_pad,
                        btn,
                        btn,
                    );
                    if pin_rect.contains_point(pos) {
                        hit_button = Some(ConstellationButtonHit::Pin);
                    } else {
                        // Action buttons (bottom row): hide, note, add context, more
                        let action_y = screen_pos.y + screen_size.y - row_pad - btn;
                        let start_x = screen_pos.x + pad;
                        for i in 0..4 {
                            let rect = Rect::new(start_x + (btn + btn_space) * i as f32, action_y, btn, btn);
                            if rect.contains_point(pos) {
                                hit_button = Some(match i {
                                    0 => ConstellationButtonHit::Hide,
                                    1 => ConstellationButtonHit::Note,
                                    2 => ConstellationButtonHit::AddContext,
                                    _ => ConstellationButtonHit::More,
                                });
                                break;
                            }
                        }
                    }
                }
            }
            if let (Some(id), Some(btn)) = (hit_id.as_ref(), hit_button.as_ref()) {
                return match btn {
                    ConstellationButtonHit::Pin => ChatHit::ConstellationPinButton(id.clone()),
                    ConstellationButtonHit::Hide => ChatHit::ConstellationHideButton(id.clone()),
                    ConstellationButtonHit::Note => ChatHit::ConstellationNoteButton(id.clone()),
                    ConstellationButtonHit::AddContext => ChatHit::ConstellationAddContextButton(id.clone()),
                    ConstellationButtonHit::More => ChatHit::ConstellationMoreButton(id.clone()),
                };
            }
            if let Some(id) = hit_id {
                if let Some(ur) = node_user_content_rect {
                    if ur.contains_point(pos) {
                        return ChatHit::ConstellationUserBubbleContent(id);
                    }
                }
                if let Some(ar) = node_assistant_content_rect {
                    if ar.contains_point(pos) {
                        return ChatHit::ConstellationAssistantBubbleContent(id);
                    }
                }
                return ChatHit::ConstellationNode(id);
            }
            return ChatHit::ConstellationBackground;
        }

        // Check message bubbles for action buttons, citations, and pin buttons
        // We need to update content height first, then get bubbles
        let bubbles = {
            let measure_fn = |text: &str, size: f32| -> Vec2 {
                // Simple approximation for hit testing
                Vec2::new(text.len() as f32 * size * 0.6, size)
            };
            self.update_content_height(measure_fn);
            self.get_message_bubbles(measure_fn)
        };
        
        // Check action buttons first (they're on top)
        if let Some((msg_idx, action_type)) = self.get_action_button_at(pos, &bubbles) {
            return match action_type {
                ActionButtonType::Edit => ChatHit::EditButton(msg_idx),
                ActionButtonType::Delete => ChatHit::DeleteButton(msg_idx),
                ActionButtonType::Mute => ChatHit::MuteButton(msg_idx),
                ActionButtonType::AddNote => ChatHit::AddNoteButton(msg_idx),
            };
        }
        
        // Check note edit and remove buttons (edit first so edit rect takes priority)
        if let Some((msg_idx, note_idx)) = self.get_note_edit_at(pos, &bubbles) {
            return ChatHit::EditNote(msg_idx, note_idx);
        }
        if let Some((msg_idx, note_idx)) = self.get_note_remove_at(pos, &bubbles) {
            return ChatHit::RemoveNote(msg_idx, note_idx);
        }
        
        // Check pin button
        if let Some(msg_idx) = self.get_pin_button_at(pos, &bubbles) {
            return ChatHit::PinButton(msg_idx);
        }
        
        // Check citations
        if let Some((msg_idx, citation_idx)) = self.get_citation_at(pos, &bubbles) {
            return ChatHit::Citation(msg_idx, citation_idx);
        }

        let scroll_hit = self.message_list.hit_test(pos - self.position);
        if scroll_hit != ScrollHit::Outside {
            return ChatHit::MessageList;
        }

        ChatHit::Background
    }
    
    pub fn get_citation_at(&self, pos: Vec2, bubbles: &[MessageBubble]) -> Option<(usize, usize)> {
        for (msg_idx, bubble) in bubbles.iter().enumerate() {
            // Check citations
            for (citation_pos, citation_size, citation_idx) in &bubble.citation_positions {
                if pos.x >= citation_pos.x && pos.x <= citation_pos.x + citation_size.x &&
                   pos.y >= citation_pos.y && pos.y <= citation_pos.y + citation_size.y {
                    return Some((msg_idx, *citation_idx));
                }
            }
        }
        None
    }
    
    pub fn toggle_citations(&mut self, msg_idx: usize) {
        if self.citations_expanded.contains(&msg_idx) {
            self.citations_expanded.remove(&msg_idx);
        } else {
            self.citations_expanded.insert(msg_idx);
        }
    }
    
    pub fn set_highlight_term(&mut self, term: Option<String>) {
        // Limit to 60 characters
        self.highlight_term = term.and_then(|t| {
            let trimmed = t.trim();
            if trimmed.len() > 60 {
                Some(trimmed[..60].to_string())
            } else if !trimmed.is_empty() {
                Some(trimmed.to_string())
            } else {
                None
            }
        });
    }
    
    pub fn get_pin_button_at(&self, pos: Vec2, bubbles: &[MessageBubble]) -> Option<usize> {
        for (msg_idx, bubble) in bubbles.iter().enumerate() {
            if let Some(pin_pos) = bubble.pin_button_position {
                if pos.x >= pin_pos.x && pos.x <= pin_pos.x + bubble.pin_button_size.x &&
                   pos.y >= pin_pos.y && pos.y <= pin_pos.y + bubble.pin_button_size.y {
                    return Some(msg_idx);
                }
            }
        }
        None
    }
    
    pub fn get_action_button_at(&self, pos: Vec2, bubbles: &[MessageBubble]) -> Option<(usize, ActionButtonType)> {
        for (msg_idx, bubble) in bubbles.iter().enumerate() {
            if let Some(edit_pos) = bubble.edit_button_position {
                if pos.x >= edit_pos.x && pos.x <= edit_pos.x + bubble.action_button_size.x &&
                   pos.y >= edit_pos.y && pos.y <= edit_pos.y + bubble.action_button_size.y {
                    return Some((msg_idx, ActionButtonType::Edit));
                }
            }
            if let Some(add_note_pos) = bubble.add_note_button_position {
                if pos.x >= add_note_pos.x && pos.x <= add_note_pos.x + bubble.action_button_size.x &&
                   pos.y >= add_note_pos.y && pos.y <= add_note_pos.y + bubble.action_button_size.y {
                    return Some((msg_idx, ActionButtonType::AddNote));
                }
            }
            if let Some(delete_pos) = bubble.delete_button_position {
                if pos.x >= delete_pos.x && pos.x <= delete_pos.x + bubble.action_button_size.x &&
                   pos.y >= delete_pos.y && pos.y <= delete_pos.y + bubble.action_button_size.y {
                    return Some((msg_idx, ActionButtonType::Delete));
                }
            }
            if let Some(mute_pos) = bubble.mute_button_position {
                if pos.x >= mute_pos.x && pos.x <= mute_pos.x + bubble.action_button_size.x &&
                   pos.y >= mute_pos.y && pos.y <= mute_pos.y + bubble.action_button_size.y {
                    return Some((msg_idx, ActionButtonType::Mute));
                }
            }
        }
        None
    }

    pub fn get_note_remove_at(&self, pos: Vec2, bubbles: &[MessageBubble]) -> Option<(usize, usize)> {
        for (msg_idx, bubble) in bubbles.iter().enumerate() {
            for (note_pos, note_size, note_idx) in &bubble.note_remove_positions {
                if pos.x >= note_pos.x && pos.x <= note_pos.x + note_size.x &&
                   pos.y >= note_pos.y && pos.y <= note_pos.y + note_size.y {
                    return Some((msg_idx, *note_idx));
                }
            }
        }
        None
    }

    pub fn get_note_edit_at(&self, pos: Vec2, bubbles: &[MessageBubble]) -> Option<(usize, usize)> {
        for (msg_idx, bubble) in bubbles.iter().enumerate() {
            for (note_pos, note_size, note_idx) in &bubble.note_edit_positions {
                if pos.x >= note_pos.x && pos.x <= note_pos.x + note_size.x &&
                   pos.y >= note_pos.y && pos.y <= note_pos.y + note_size.y {
                    return Some((msg_idx, *note_idx));
                }
            }
        }
        None
    }

    pub fn send_message(&mut self) -> Option<String> {
        if !self.input_field.text.trim().is_empty() {
            let text = self.input_field.text.clone();
            self.input_field.clear();
            Some(text)
        } else {
            None
        }
    }

    pub fn start_editing_message(&mut self, msg_idx: usize) {
        if msg_idx < self.messages.len() {
            self.editing_message_idx = Some(msg_idx);
            self.edit_textarea.text = self.messages[msg_idx].content.clone();
            self.edit_textarea.on_focus();
        }
    }

    pub fn save_edited_message(&mut self) -> bool {
        if let Some(msg_idx) = self.editing_message_idx {
            if msg_idx < self.messages.len() {
                let new_content = self.edit_textarea.text.trim().to_string();
                if !new_content.is_empty() {
                    self.messages[msg_idx].content = new_content;
                    self.editing_message_idx = None;
                    self.edit_textarea.on_blur();
                    self.edit_textarea.text.clear();
                    return true;
                }
            }
        }
        false
    }

    pub fn cancel_editing_message(&mut self) {
        self.editing_message_idx = None;
        self.edit_textarea.on_blur();
        self.edit_textarea.text.clear();
    }

    /// Update edit_textarea position and size when editing so hit-test and layout match the rendered rect.
    /// For linear view only; use update_edit_textarea_rect_constellation when in graph mode.
    pub fn update_edit_textarea_rect(&mut self, mut measure_fn: impl FnMut(&str, f32) -> Vec2) {
        let Some(msg_idx) = self.editing_message_idx else { return };
        let bubbles = self.get_message_bubbles(&mut measure_fn);
        let bubble = match bubbles.get(msg_idx) {
            Some(b) => b,
            None => return,
        };
        let padding = 12.0;
        let edit_max_width = bubble.size.x - padding * 2.0;
        const EDIT_FONT_SIZE: f32 = 16.0;
        let edit_line_height = EDIT_FONT_SIZE * 1.2;
        let words: Vec<&str> = self.edit_textarea.text.split_whitespace().collect();
        let mut line_count = 0u32;
        let mut current_line = String::new();
        for word in words {
            let test_line = if current_line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", current_line, word)
            };
            let test_width = measure_fn(&test_line, EDIT_FONT_SIZE).x;
            if test_width > edit_max_width && !current_line.is_empty() {
                line_count += 1;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }
        if !current_line.is_empty() {
            line_count += 1;
        }
        let line_count = line_count.max(1).min(20);
        let edit_height = (line_count as f32 * edit_line_height) + padding * 2.0;
        let edit_height = edit_height.clamp(edit_line_height * 2.0 + padding * 2.0, 400.0);
        self.edit_textarea.position = Vec2::new(
            bubble.position.x + padding,
            bubble.text_start_y,
        );
        self.edit_textarea.size = Vec2::new(edit_max_width, edit_height);
    }

    /// Update edit_textarea for constellation: position in-place on the shard body (assistant bubble area).
    pub fn update_edit_textarea_rect_constellation(
        &mut self,
        mut measure_fn: impl FnMut(&str, f32) -> Vec2,
        graph_state: &crate::state::GraphState,
    ) {
        let Some(msg_idx) = self.editing_message_idx else { return };
        let shard_id = match self.messages.get(msg_idx).and_then(|m| m.shard_id.as_ref()) {
            Some(id) => id,
            None => return,
        };
        let node = match graph_state.get_node(shard_id) {
            Some(n) => n,
            None => return,
        };
        let screen_pos = self.constellation_view.world_to_screen(node.position);
        let screen_size = self.constellation_view.world_size_to_screen(node.size);
        const PAD: f32 = 8.0;
        const BUBBLE_SPACING: f32 = 6.0;
        let max_content_width = (screen_size.x * 0.7 - PAD * 2.0).max(80.0);
        const FONT_SIZE: f32 = 16.0;
        let line_height = FONT_SIZE * 1.2;
        let user_size = node.shard.user_content.as_deref()
            .filter(|s| !s.is_empty())
            .map(|u| Self::measure_wrapped_block(&mut measure_fn, u, max_content_width, FONT_SIZE))
            .unwrap_or(Vec2::ZERO);
        let mut y = screen_pos.y + PAD;
        if user_size.x > 0.0 && user_size.y > 0.0 {
            y += user_size.y + PAD * 2.0 + BUBBLE_SPACING;
        }
        let edit_max_width = max_content_width;
        let words: Vec<&str> = self.edit_textarea.text.split_whitespace().collect();
        let mut line_count = 0u32;
        let mut current_line = String::new();
        for word in words {
            let test_line = if current_line.is_empty() { word.to_string() } else { format!("{} {}", current_line, word) };
            let test_w = measure_fn(&test_line, FONT_SIZE).x;
            if test_w > edit_max_width && !current_line.is_empty() {
                line_count += 1;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }
        if !current_line.is_empty() {
            line_count += 1;
        }
        let line_count = line_count.max(1).min(20);
        let edit_height = (line_count as f32 * line_height) + PAD * 2.0;
        let edit_height = edit_height.clamp(line_height * 2.0 + PAD * 2.0, 400.0);
        self.edit_textarea.position = Vec2::new(screen_pos.x + PAD, y);
        self.edit_textarea.size = Vec2::new(edit_max_width, edit_height);
    }

    pub fn measure_wrapped_block(measure: &mut impl FnMut(&str, f32) -> Vec2, text: &str, max_width: f32, font_size: f32) -> Vec2 {
        let line_height = font_size * 1.2;
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_line = String::new();
        let mut max_w = 0.0f32;
        let mut line_count = 0u32;
        for word in words {
            let test_line = if current_line.is_empty() { word.to_string() } else { format!("{} {}", current_line, word) };
            let test_size = measure(&test_line, font_size);
            if test_size.x > max_width && !current_line.is_empty() {
                max_w = max_w.max(measure(&current_line, font_size).x);
                line_count += 1;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }
        if !current_line.is_empty() {
            line_count += 1;
            max_w = max_w.max(measure(&current_line, font_size).x);
        }
        let h = (line_count.max(1) as f32) * line_height;
        Vec2::new(max_w, h)
    }

    /// Update note_input position for constellation: popup below shard (so the quick-view notes list
    /// at the bottom of the shard body stays visible above the input), 2/3 width, adaptive height,
    /// bump above shard if placing below would overflow viewport.
    pub fn update_note_input_rect_constellation(
        &mut self,
        mut measure_fn: impl FnMut(&str, f32) -> Vec2,
        graph_state: &crate::state::GraphState,
        viewport_bottom: f32,
    ) {
        let Some(msg_idx) = self.adding_note_msg_idx.or(self.editing_note.map(|(m, _)| m)) else { return };
        let Some(shard_id) = self.messages.get(msg_idx).and_then(|m| m.shard_id.as_ref()) else { return };
        let Some(node) = graph_state.get_node(shard_id) else { return };
        let screen_pos = self.constellation_view.world_to_screen(node.position);
        let screen_size = self.constellation_view.world_size_to_screen(node.size);
        const GAP: f32 = 6.0;
        const PAD: f32 = 8.0;
        const FONT_SIZE: f32 = 16.0;
        let line_height = FONT_SIZE * 1.2;
        let width = screen_size.x * 2.0 / 3.0;
        let max_content_width = width - PAD * 2.0;
        let words: Vec<&str> = self.note_input.text.split_whitespace().collect();
        let mut line_count = 0u32;
        let mut current_line = String::new();
        for word in words {
            let test_line = if current_line.is_empty() { word.to_string() } else { format!("{} {}", current_line, word) };
            let test_w = measure_fn(&test_line, FONT_SIZE).x;
            if test_w > max_content_width && !current_line.is_empty() {
                line_count += 1;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }
        if !current_line.is_empty() {
            line_count += 1;
        }
        let line_count = line_count.max(1);
        let height = (line_count as f32 * line_height) + PAD * 2.0;
        let note_y_below = screen_pos.y + screen_size.y + GAP;
        let note_bottom_below = note_y_below + height;
        let (note_x, note_y) = if note_bottom_below > viewport_bottom {
            let note_y_above = screen_pos.y - height - GAP;
            (screen_pos.x + (screen_size.x - width) * 0.5, note_y_above)
        } else {
            (screen_pos.x + (screen_size.x - width) * 0.5, note_y_below)
        };
        self.note_input.position = Vec2::new(note_x, note_y);
        self.note_input.size = Vec2::new(width, height);
    }

    pub fn delete_message(&mut self, msg_idx: usize) {
        if msg_idx < self.messages.len() {
            self.messages.remove(msg_idx);
            self.muted_messages.remove(&msg_idx);
            // Adjust indices for muted_messages after deletion
            let mut new_muted = HashSet::new();
            for &idx in &self.muted_messages {
                if idx > msg_idx {
                    new_muted.insert(idx - 1);
                } else if idx < msg_idx {
                    new_muted.insert(idx);
                }
            }
            self.muted_messages = new_muted;
            // Adjust citations_expanded indices
            let mut new_expanded = HashSet::new();
            for &idx in &self.citations_expanded {
                if idx > msg_idx {
                    new_expanded.insert(idx - 1);
                } else if idx < msg_idx {
                    new_expanded.insert(idx);
                }
            }
            self.citations_expanded = new_expanded;
            self.delete_confirm_idx = None;
        }
    }

    pub fn toggle_mute_message(&mut self, msg_idx: usize) {
        if self.muted_messages.contains(&msg_idx) {
            self.muted_messages.remove(&msg_idx);
        } else {
            self.muted_messages.insert(msg_idx);
        }
    }

    /// Start adding a note to the message at msg_idx. Shows note_input.
    pub fn start_adding_note(&mut self, msg_idx: usize) {
        if msg_idx < self.messages.len() {
            self.adding_note_msg_idx = Some(msg_idx);
            self.editing_note = None;
            self.note_input.text.clear();
            self.note_input.on_focus();
        }
    }

    /// Save the current note (append if adding, update if editing). Returns true if saved.
    pub fn save_note(&mut self) -> bool {
        let text = self.note_input.text.trim().to_string();
        if text.is_empty() {
            return false;
        }
        if let Some((msg_idx, note_idx)) = self.editing_note {
            if msg_idx < self.messages.len() && note_idx < self.messages[msg_idx].notes.len() {
                self.messages[msg_idx].notes[note_idx] = text;
                self.editing_note = None;
                self.adding_note_msg_idx = None;
                self.note_input.on_blur();
                self.note_input.text.clear();
                return true;
            }
        }
        if let Some(msg_idx) = self.adding_note_msg_idx {
            if msg_idx < self.messages.len() {
                self.messages[msg_idx].notes.push(text);
                self.adding_note_msg_idx = None;
                self.note_input.on_blur();
                self.note_input.text.clear();
                return true;
            }
        }
        false
    }

    /// Cancel adding or editing a note.
    pub fn cancel_note(&mut self) {
        self.adding_note_msg_idx = None;
        self.editing_note = None;
        self.note_input.on_blur();
        self.note_input.text.clear();
    }

    /// Remove the note at note_idx from the message at msg_idx.
    pub fn remove_note(&mut self, msg_idx: usize, note_idx: usize) {
        if msg_idx < self.messages.len() && note_idx < self.messages[msg_idx].notes.len() {
            self.messages[msg_idx].notes.remove(note_idx);
            if self.editing_note == Some((msg_idx, note_idx)) {
                self.cancel_note();
            } else if let Some((m, n)) = self.editing_note {
                if m == msg_idx && n > note_idx {
                    self.editing_note = Some((m, n - 1));
                }
            }
        }
    }

    /// Start editing the note at (msg_idx, note_idx). Shows note_input prefilled.
    pub fn start_editing_note(&mut self, msg_idx: usize, note_idx: usize) {
        if msg_idx < self.messages.len() && note_idx < self.messages[msg_idx].notes.len() {
            self.editing_note = Some((msg_idx, note_idx));
            self.adding_note_msg_idx = None;
            self.note_input.text = self.messages[msg_idx].notes[note_idx].clone();
            self.note_input.on_focus();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessagePart {
    User,
    Assistant,
}

#[derive(Debug)]
enum ConstellationButtonHit {
    Pin,
    Hide,
    Note,
    AddContext,
    More,
}

#[derive(Debug)]
pub enum ChatHit {
    Input,
    SendButton,
    ContextPoolButton,
    ContextPoolMenu,
    ContextPoolItem(usize),
    ContextPoolCreate,  // "Create new collection" button in dropdown
    MessageList,
    /// Constellation view: hit on a node (node_id).
    ConstellationNode(String),
    /// Constellation: over user bubble content area (scrollable text); node_id.
    ConstellationUserBubbleContent(String),
    /// Constellation: over assistant bubble content area (scrollable text); node_id.
    ConstellationAssistantBubbleContent(String),
    /// Constellation view: hit on background (for pan).
    ConstellationBackground,
    /// Constellation: click on in-place edit textarea
    ConstellationEditTextarea,
    /// Constellation: click on note input popup
    ConstellationNoteInput,
    /// Constellation node buttons (node_id)
    ConstellationPinButton(String),
    ConstellationHideButton(String),
    ConstellationNoteButton(String),
    ConstellationAddContextButton(String),
    ConstellationMoreButton(String),
    /// Per-message edit (opens modal): (node_id, "user"|"assistant")
    ConstellationMessageEditButton(String, MessagePart),
    /// Per-message hide: (node_id, "user"|"assistant")
    ConstellationMessageHideButton(String, MessagePart),
    /// Constellation: click on citation text (node_id, citation_index)
    ConstellationCitation(String, usize),
    /// Constellation: edit note on node (node_id, note_index)
    ConstellationEditNote(String, usize),
    /// Constellation: remove note on node (node_id, note_index)
    ConstellationRemoveNote(String, usize),
    Background,
    Citation(usize, usize),  // (message_index, citation_index)
    PinButton(usize),  // message_index
    EditButton(usize),  // message_index
    DeleteButton(usize),  // message_index
    MuteButton(usize),  // message_index
    AddNoteButton(usize),  // message_index
    RemoveNote(usize, usize),  // (message_index, note_index)
    EditNote(usize, usize),    // (message_index, note_index)
}

pub struct MessageBubble {
    pub position: Vec2,
    pub size: Vec2,
    pub role: MessageRole,
    pub content: String,
    pub text_start_y: f32,
    pub message: Option<ChatMessage>,  // Reference to full message for citations
    pub citation_positions: Vec<(Vec2, Vec2, usize)>,  // (position, size, citation_index)
    pub pin_button_position: Option<Vec2>,
    pub pin_button_size: Vec2,
    pub is_muted: bool,
    pub message_idx: usize,
    pub edit_button_position: Option<Vec2>,
    pub delete_button_position: Option<Vec2>,
    pub mute_button_position: Option<Vec2>,
    pub action_button_size: Vec2,
    pub add_note_button_position: Option<Vec2>,
    pub notes: Vec<String>,
    /// (position, size, note_index) for remove button hit-test
    pub note_remove_positions: Vec<(Vec2, Vec2, usize)>,
    /// (position, size, note_index) for edit button hit-test
    pub note_edit_positions: Vec<(Vec2, Vec2, usize)>,
}


