use glam::{Vec2, Vec4};
use crate::gfx::components::chat::markdown::measure_message_markdown;
use crate::gfx::renderer::Renderer;
use crate::ui::{TextInput, ScrollView, ScrollHit, Dropdown, DropdownItem};
use crate::ui::core::{Rect, layout};
use crate::ui::components::VStack;
use std::cell::RefCell;
use std::collections::HashSet;
use std::collections::HashMap;
use crate::state::shard::Shard;
use crate::api::models::GraphMention;

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
        const LERP_SPEED: f32 = 4.0;  // Slower for smoother zoom interpolation
        let t = (LERP_SPEED * dt).min(1.0);
        self.camera_position_animated = self.camera_position_animated.lerp(self.camera_position, t);
        self.scale_animated = self.scale_animated + (self.scale - self.scale_animated) * t;
    }

    pub fn effective_message_font_pt(&self) -> f32 {
        crate::ui::style::font_size::MESSAGE_BODY * self.scale_animated
    }

    pub fn macro_mode_active(&self) -> bool {
        self.scale_animated <= Self::MIN_SCALE
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

/// Per-bubble 2D scroll offset for constellation shard content.
/// `user` / `assistant` each store (x, y) in screen pixels.
#[derive(Default, Clone, Copy, Debug)]
pub struct BubbleScroll {
    pub user: Vec2,
    pub assistant: Vec2,
}

pub struct ChatWindow {
    pub position: Vec2,
    pub size: Vec2,
    pub message_list: ScrollView,
    /// Constellation view (2D world + camera). Used when graph is active.
    pub constellation_view: ConstellationView,
    /// When true, `update_layout` does not overwrite `constellation_view` (full-bleed graph is set from `App`).
    pub graph_viewport_full_bleed: bool,
    /// Top Y of the composer strip; set in `update_layout` from `input_y`.
    pub composer_top_y: f32,
    /// Height of composer block (input + padding + optional pill row); for chassis backplate.
    pub composer_block_height: f32,
    /// Graph mode: pan / empty background hit-testing excludes header, sidebar, and composer.
    pub constellation_interactive_rect: Rect,
    pub input_field: TextInput,
    pub send_button_position: Vec2,
    pub send_button_size: Vec2,
    pub context_pool_button_position: Vec2,
    pub context_pool_button_size: Vec2,
    pub context_pool_dropdown: Dropdown,
    /// Slash-command palette for named system prompts (anchored to input field).
    pub system_prompt_dropdown: Dropdown,
    pub messages: Vec<ChatMessage>,
    pub selected_collection_id: Option<i32>,
    pub editing_message_idx: Option<usize>,
    pub edit_textarea: TextInput,  // Textarea for editing messages
    pub delete_confirm_idx: Option<usize>,
    pub muted_messages: HashSet<usize>,
    pub muted_shard_ids: HashSet<String>,
    pub is_sending: bool,
    pub highlight_term: Option<String>,
    pub citations_expanded: HashSet<usize>,  // Track which messages have expanded citations
    pub citations_expanded_shards: HashSet<String>,
    /// When Some(msg_idx), user is adding a note to that message; note_input holds the text.
    pub adding_note_msg_idx: Option<usize>,
    /// Text input for adding or editing a note (positioned when adding_note_msg_idx or editing_note is set).
    pub note_input: TextInput,
    /// Send button next to constellation note_input (same row).
    pub note_send_button_rect: Rect,
    /// When Some((msg_idx, note_idx)), user is editing that note; note_input holds the text.
    pub editing_note: Option<(usize, usize)>,
    /// Cached (user_size, assistant_size) per node id from last render; used for hit test to match render layout.
    pub constellation_layout_cache: RefCell<Option<HashMap<String, (Vec2, Vec2)>>>,
    /// Per-node per-bubble 2D scroll for constellation; only text inside bubbles scrolls.
    pub constellation_scroll_offsets: RefCell<HashMap<String, BubbleScroll>>,
    /// Target scroll for smooth lerp each frame per node.
    pub constellation_scroll_targets: RefCell<HashMap<String, BubbleScroll>>,
    /// Node id under cursor in constellation (for hover highlight). None when not over a shard.
    pub hovered_node_id: Option<String>,
    /// Macro constellation only: selected nodes set (single/ctrl/shift behavior).
    pub macro_selected_node_ids: HashSet<String>,
    /// Macro constellation only: range-selection anchor.
    pub macro_selection_anchor: Option<String>,
    /// Last focused leaf before entering macro mode; used by Escape restore behavior.
    pub last_active_node_before_macro: Option<String>,
    /// `@` mention picker for papers (and extended for shards/graphs).
    pub mention_popup_open: bool,
    pub mention_selected_index: usize,
    pub mention_filter: String,
    pub mention_rows: Vec<MentionEntry>,
    /// Slash-selected system prompt (shown as pill; not left as `/name` in the input).
    pub pending_system_prompt: Option<PendingSystemPrompt>,
    /// @-mentions as structured data + display labels for pills.
    pub pending_mentions: Vec<PendingMention>,
    /// Layout for composer pills (refreshed in `update_layout`).
    pub composer_pill_items: Vec<ComposerPillItem>,
    pub composer_pill_row_rect: Rect,
    /// Linear list bubbles from last [`Self::refresh_linear_message_layout`] (markdown-aware widths/heights).
    pub linear_message_bubbles_cache: Vec<MessageBubble>,
}

/// One row in the @-mention popup (papers, shards in current graph, or other graphs).
#[derive(Clone, Debug)]
pub enum MentionEntry {
    Paper(i32),
    Shard { graph_id: String, shard_id: String },
    Graph { graph_id: String },
    /// Local notepad document (`data/documents/{id}.json`).
    Notepad {
        document_id: String,
        title: String,
    },
}

/// Slash-selected system prompt shown as a composer pill (not stored as `/name` in the input).
#[derive(Clone, Debug)]
pub struct PendingSystemPrompt {
    pub name: String,
    pub content: String,
}

/// Resolved @-mention for send + pill label; canonical tokens are not kept in the text field.
#[derive(Clone, Debug)]
pub struct PendingMention {
    pub mention: GraphMention,
    pub label: String,
}

fn same_graph_mention(a: &GraphMention, b: &GraphMention) -> bool {
    match (a, b) {
        (GraphMention::Paper { paper_id: a }, GraphMention::Paper { paper_id: b }) => a == b,
        (
            GraphMention::Shard {
                graph_id: ga,
                shard_id: sa,
            },
            GraphMention::Shard {
                graph_id: gb,
                shard_id: sb,
            },
        ) => ga == gb && sa == sb,
        (GraphMention::Graph { graph_id: ga }, GraphMention::Graph { graph_id: gb }) => ga == gb,
        (
            GraphMention::Notepad { document_id: a },
            GraphMention::Notepad { document_id: b },
        ) => a == b,
        _ => false,
    }
}

impl ChatWindow {
    pub fn new(position: Vec2, size: Vec2) -> Self {
        use crate::ui::style;
        
        let input_container_height = 60.0;  // Total height including padding
        let padding = style::hero::MAIN_VIEWPORT_GUTTER;
        
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
            slash_name: None,
        });

        let mut system_prompt_dropdown = Dropdown::new(
            input.position,
            input.size,
        );
        system_prompt_dropdown.show_create_footer = false;

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
        let note_send_button_rect = Rect::new(0.0, 0.0, 0.0, 0.0);

        let mut window = Self {
            position,
            size,
            message_list,
            constellation_view,
            graph_viewport_full_bleed: false,
            composer_top_y: 0.0,
            composer_block_height: 0.0,
            constellation_interactive_rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            input_field: input,
            send_button_position,
            send_button_size,
            context_pool_button_position,
            context_pool_button_size,
            context_pool_dropdown,
            system_prompt_dropdown,
            messages: Vec::new(),
            selected_collection_id: None,
            editing_message_idx: None,
            edit_textarea,
            delete_confirm_idx: None,
            muted_messages: HashSet::new(),
            muted_shard_ids: HashSet::new(),
            is_sending: false,
            highlight_term: None,
            citations_expanded: HashSet::new(),
            citations_expanded_shards: HashSet::new(),
            adding_note_msg_idx: None,
            note_input,
            note_send_button_rect,
            editing_note: None,
            constellation_layout_cache: RefCell::new(None),
            constellation_scroll_offsets: RefCell::new(HashMap::new()),
            constellation_scroll_targets: RefCell::new(HashMap::new()),
            hovered_node_id: None,
            macro_selected_node_ids: HashSet::new(),
            macro_selection_anchor: None,
            last_active_node_before_macro: None,
            mention_popup_open: false,
            mention_selected_index: 0,
            mention_filter: String::new(),
            mention_rows: Vec::new(),
            pending_system_prompt: None,
            pending_mentions: Vec::new(),
            composer_pill_items: Vec::new(),
            composer_pill_row_rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            linear_message_bubbles_cache: Vec::new(),
        };
        
        window.update_layout();
        window
    }

    pub fn has_composer_pills(&self) -> bool {
        self.pending_system_prompt.is_some() || !self.pending_mentions.is_empty()
    }

    pub fn constellation_macro_active(&self) -> bool {
        self.constellation_view.macro_mode_active()
    }

    pub fn macro_selection_sorted_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.macro_selected_node_ids.iter().cloned().collect();
        ids.sort();
        ids
    }

    fn push_pending_mention(&mut self, pending: PendingMention) {
        if self
            .pending_mentions
            .iter()
            .any(|p| same_graph_mention(&p.mention, &pending.mention))
        {
            return;
        }
        self.pending_mentions.push(pending);
    }

    /// Remove leading `/token` from input (after picking a slash prompt or paste).
    pub fn strip_leading_slash_token(&mut self) {
        let t = self.input_field.text.clone();
        let chars: Vec<char> = t.chars().collect();
        let old_cursor = self.input_field.cursor_position;
        let mut start = 0usize;
        while start < chars.len() && chars[start].is_whitespace() {
            start += 1;
        }
        if start >= chars.len() || chars[start] != '/' {
            return;
        }
        let mut end = start + 1;
        while end < chars.len() && !chars[end].is_whitespace() {
            end += 1;
        }
        let mut new_chars: Vec<char> = Vec::new();
        new_chars.extend(chars[..start].iter().cloned());
        new_chars.extend(chars[end..].iter().cloned());
        self.input_field.text = new_chars.into_iter().collect();
        self.input_field.ensure_cursor_valid();
        let new_len = self.input_field.text.chars().count();
        let stripped = end - start;
        if old_cursor > end {
            self.input_field.cursor_position = (old_cursor - stripped).min(new_len);
        } else if old_cursor > start {
            self.input_field.cursor_position = start.min(new_len);
        } else {
            self.input_field.cursor_position = old_cursor.min(new_len);
        }
    }

    fn remove_active_at_mention(&mut self) {
        let t = self.input_field.text.clone();
        let cur = self.input_field.cursor_position.min(t.chars().count());
        let before: String = t.chars().take(cur).collect();
        let after: String = t.chars().skip(cur).collect();
        if let Some(last_at) = before.rfind('@') {
            let prefix = before[..last_at].to_string();
            self.input_field.text = format!("{}{}", prefix, after);
            self.input_field.cursor_position = prefix.chars().count();
        }
    }

    fn refresh_composer_pill_layout(&mut self) {
        self.composer_pill_items.clear();
        if !self.has_composer_pills() {
            self.composer_pill_row_rect = Rect::new(0.0, 0.0, 0.0, 0.0);
            return;
        }
        let input_rect = Rect::from_pos_size(self.input_field.position, self.input_field.size);
        const PILL_H: f32 = 26.0;
        const GAP: f32 = 6.0;
        const CLOSE_W: f32 = 22.0;
        const PAD_X: f32 = 8.0;
        const CHAR_W: f32 = 7.0;
        const MAX_LABEL_W: f32 = 160.0;
        let pill_y = input_rect.y - GAP - PILL_H;
        self.composer_pill_row_rect = Rect::new(input_rect.x, pill_y, input_rect.width, PILL_H);
        let mut x = input_rect.x + 4.0;
        let max_x = input_rect.right() - 4.0;
        if let Some(ref sp) = self.pending_system_prompt {
            let label = format!("/{}", sp.name);
            let label_w = (label.chars().count() as f32 * CHAR_W).min(MAX_LABEL_W);
            let pill_w = PAD_X + label_w + PAD_X + CLOSE_W + 4.0;
            if x + pill_w <= max_x {
                let body_rect = Rect::new(x, pill_y, pill_w - CLOSE_W, PILL_H);
                let close_rect = Rect::new(x + pill_w - CLOSE_W, pill_y, CLOSE_W, PILL_H);
                self.composer_pill_items.push(ComposerPillItem {
                    body_rect,
                    close_rect,
                    label,
                    hit_body: ChatHit::ComposerSystemPromptPillBody,
                    hit_close: ChatHit::ComposerSystemPromptPillClose,
                });
                x += pill_w + 6.0;
            }
        }
        for (i, pm) in self.pending_mentions.iter().enumerate() {
            let label = pm.label.clone();
            let label_w = (label.chars().count() as f32 * CHAR_W).min(MAX_LABEL_W);
            let pill_w = PAD_X + label_w + PAD_X + CLOSE_W + 4.0;
            if x + pill_w > max_x {
                break;
            }
            let body_rect = Rect::new(x, pill_y, pill_w - CLOSE_W, PILL_H);
            let close_rect = Rect::new(x + pill_w - CLOSE_W, pill_y, CLOSE_W, PILL_H);
            self.composer_pill_items.push(ComposerPillItem {
                body_rect,
                close_rect,
                label,
                hit_body: ChatHit::ComposerMentionPillBody(i),
                hit_close: ChatHit::ComposerMentionPillClose(i),
            });
            x += pill_w + 6.0;
        }
    }

    pub fn update_layout(&mut self) {
        use crate::ui::style;
        
        let padding = style::hero::MAIN_VIEWPORT_GUTTER;
        const PILL_H: f32 = 26.0;
        const PILL_GAP: f32 = 6.0;
        let pill_row_extra = if self.has_composer_pills() {
            PILL_H + PILL_GAP
        } else {
            0.0
        };
        // Input height + padding + optional composer pill row above the input
        let input_container_height = style::input_height::NORMAL + padding * 2.0 + pill_row_extra;
        
        // Position input and button at bottom of chat window (which is at bottom of viewport)
        let input_y = self.position.y + self.size.y - input_container_height;
        
        // Use horizontal stack to position context pool button, input, and send button
        use crate::ui::core::{layout, Rect};
        let input_area = Rect::new(
            self.position.x + padding,
            input_y + padding + pill_row_extra,
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
            self.system_prompt_dropdown.anchor_rect = Rect::from_pos_size(self.input_field.position, self.input_field.size);
            self.system_prompt_dropdown.button_size = self.input_field.size;
            if self.system_prompt_dropdown.is_open {
                self.system_prompt_dropdown.update_layout();
            }
        }
        
        // Send button position
        if let Some(send_rect) = component_rects.get(2) {
            self.send_button_position = send_rect.position();
        }
        
        // Update message list and constellation view height (leave space for input at bottom)
        let message_list_height = self.size.y - input_container_height - padding;
        self.message_list.size.y = message_list_height;
        self.message_list.position = Vec2::new(self.position.x, self.position.y + padding);
        if !self.graph_viewport_full_bleed {
            self.constellation_view.position = Vec2::new(self.position.x, self.position.y + padding);
            self.constellation_view.size = Vec2::new(self.size.x, message_list_height);
        }
        self.composer_top_y = input_y;
        self.composer_block_height = input_container_height;

        self.refresh_composer_pill_layout();
    }

    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
        // Scroll height and bubble cache are recomputed on the next frame in [`Renderer::render`].
        self.message_list.scroll_to_bottom();
    }

    /// Citations + notes stacked for layout (same order as rendered under the message body).
    fn build_aux_vstack(&self, msg_idx: usize, msg: &ChatMessage) -> VStack {
        let mut aux = VStack::new(0.0, 0.0);
        use crate::ui::text::TextAlignment;
        use crate::ui::style;
        let is_citations_expanded = self.citations_expanded.contains(&msg_idx);
        if !msg.citations.is_empty() {
            if is_citations_expanded {
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
                    aux.add_text_styled(
                        &citation_text,
                        style::font_size::MESSAGE_BODY * 0.85,
                        style::text::SECONDARY(),
                        TextAlignment::Left,
                    );
                }
            } else {
                let citation_text = format!("Sources ({})", msg.citations.len());
                aux.add_text_styled(
                    &citation_text,
                    style::font_size::MESSAGE_BODY * 0.85,
                    style::text::SECONDARY(),
                    TextAlignment::Left,
                );
            }
        }
        for note in &msg.notes {
            aux.add_text_styled(
                &format!("• {}", note),
                style::font_size::MESSAGE_BODY * 0.85,
                style::text::SECONDARY(),
                TextAlignment::Left,
            );
        }
        aux
    }

    pub fn refresh_linear_message_layout(&mut self, renderer: &mut Renderer) {
        self.update_content_height_markdown(renderer);
        self.linear_message_bubbles_cache = self.compute_message_bubbles(renderer);
    }

    pub fn update_content_height(&mut self, measure_text_fn: impl Fn(&str, f32) -> Vec2) {
        let message_spacing = 16.0;
        let padding = 12.0;
        let max_bubble_width = self.size.x * 2.0 / 3.0;
        const FONT_SIZE: f32 = crate::ui::style::font_size::MESSAGE_BODY;
        let line_height = FONT_SIZE * crate::ui::style::font_size::LINE_HEIGHT_RATIO;
        
        let mut total_height = padding;

        for msg in &self.messages {
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
            const NOTE_LINE_H: f32 = 18.0;
            bubble_height += (msg.notes.len() as f32 * NOTE_LINE_H) + 20.0 + 25.0;
            
            total_height += bubble_height + message_spacing;
        }

        total_height += padding;
        self.message_list.set_content_height(total_height);
    }

    pub fn update_content_height_markdown(&mut self, renderer: &mut Renderer) {
        let message_spacing = 16.0;
        let bubble_margin = 8.0;
        let padding = 12.0;
        let max_bubble_width = self.size.x * 2.0 / 3.0;
        const FONT_SIZE: f32 = crate::ui::style::font_size::MESSAGE_BODY;
        let content_max_width = max_bubble_width - padding * 2.0;
        let pin_extra = 20.0 + 5.0;
        let action_row = 20.0 + 5.0;
        let mut total_height = padding;

        for (msg_idx, msg) in self.messages.iter().enumerate() {
            let msg_sz = measure_message_markdown(renderer, &msg.content, content_max_width, FONT_SIZE);
            let aux = self.build_aux_vstack(msg_idx, msg);
            let aux_size = {
                let mut measure_wrapper: Box<dyn FnMut(&str, f32) -> Vec2> = Box::new(|text: &str, font_size: f32| -> Vec2 {
                    renderer.measure_text(text, font_size)
                });
                aux.wrap_content(Some(content_max_width), Some(measure_wrapper.as_mut()))
            };
            let mut bubble_height = padding * 2.0 + msg_sz.y + aux_size.y;
            if matches!(msg.role, MessageRole::Assistant) {
                bubble_height += pin_extra;
            }
            bubble_height += action_row;

            total_height += bubble_height + message_spacing + bubble_margin;
        }

        total_height += padding;
        self.message_list.set_content_height(total_height);
    }

    pub fn compute_message_bubbles(&self, renderer: &mut Renderer) -> Vec<MessageBubble> {
        let padding = 12.0;
        let message_spacing = 16.0;
        let bubble_margin = 8.0;
        let max_bubble_width = self.size.x * 2.0 / 3.0;
        const FONT_SIZE: f32 = crate::ui::style::font_size::MESSAGE_BODY;
        let scroll_offset = self.message_list.scroll_offset;
        
        let message_list_top = self.message_list.position.y;
        
        let mut bubbles = Vec::new();
        let mut y_offset = padding - scroll_offset;

        for (msg_i, msg) in self.messages.iter().enumerate() {
            let content_max_width = max_bubble_width - padding * 2.0;
            let msg_sz = measure_message_markdown(renderer, &msg.content, content_max_width, FONT_SIZE);
            let aux = self.build_aux_vstack(msg_i, msg);
            let aux_size = {
                let mut measure_wrapper: Box<dyn FnMut(&str, f32) -> Vec2> = Box::new(|text: &str, font_size: f32| -> Vec2 {
                    renderer.measure_text(text, font_size)
                });
                aux.wrap_content(Some(content_max_width), Some(measure_wrapper.as_mut()))
            };
            let inner_w = msg_sz.x.max(aux_size.x).min(content_max_width);
            let content_height = padding * 2.0 + msg_sz.y + aux_size.y;
            let content_size = Vec2::new(inner_w + padding * 2.0, content_height);
            
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
            let is_citations_expanded = self.citations_expanded.contains(&msg_i);
            if !msg.citations.is_empty() {
                let citation_item_height = 20.0;
                let citation_spacing = 0.0;
                let citation_padding = padding;
                let citation_start_y = message_list_top + y_offset + padding + msg_sz.y;
                
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
                            let citation_width = renderer.measure_text(&citation_text, FONT_SIZE * 0.85).x;
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
                    let citation_width = renderer.measure_text(&citation_text, FONT_SIZE * 0.85).x;
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

            let is_muted = self.muted_messages.contains(&msg_i);
            
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
                message_idx: msg_i,
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

        bubbles
    }

    pub fn hit_test(&mut self, pos: Vec2, graph_state: &crate::state::GraphState) -> ChatHit {
        if self.mention_popup_open {
            if let Some(rect) = self.mention_popup_rect() {
                if rect.contains_point(pos) {
                    const ROW: f32 = 28.0;
                    const PAD: f32 = 4.0;
                    let inner_y = pos.y - rect.y - PAD;
                    if inner_y >= 0.0 {
                        let idx = (inner_y / ROW) as usize;
                        if idx < self.mention_rows.len() {
                            return ChatHit::MentionItem(idx);
                        }
                    }
                    return ChatHit::Background;
                }
            }
        }

        if self.system_prompt_dropdown.is_open {
            if self.system_prompt_dropdown.contains_menu(pos) {
                if let Some(index) = self.system_prompt_dropdown.get_menu_item_at(pos) {
                    return ChatHit::SystemPromptItem(index);
                }
                return ChatHit::SystemPromptMenu;
            }
        }

        for item in self.composer_pill_items.iter().rev() {
            if item.close_rect.contains_point(pos) {
                return item.hit_close.clone();
            }
            if item.body_rect.contains_point(pos) {
                return item.hit_body.clone();
            }
        }

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
            
            if self.context_pool_dropdown.show_create_footer && create_button_rect.contains_point(pos) {
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
        if graph_state.constellation_view_active() && self.constellation_view.contains_screen(pos) {
            if self.editing_message_idx.is_some() && self.edit_textarea.contains(pos) {
                return ChatHit::ConstellationEditTextarea;
            }
            if (self.adding_note_msg_idx.is_some() || self.editing_note.is_some())
                && self.note_send_button_rect.contains_point(pos)
            {
                return ChatHit::ConstellationNoteSendButton;
            }
            if (self.adding_note_msg_idx.is_some() || self.editing_note.is_some()) && self.note_input.contains(pos) {
                return ChatHit::ConstellationNoteInput;
            }
            let world = self.constellation_view.screen_to_world(pos);
            let scale = self.constellation_view.scale_animated;
            let pad = crate::ui::style::padding::SMALL * scale;
            let row_pad = crate::ui::style::constellation::ACTION_ROW_PADDING * scale;
            let btn = crate::ui::style::constellation::ACTION_BUTTON_SIZE * scale;
            let btn_space = crate::ui::style::constellation::ACTION_BUTTON_SPACING * scale;
            let msg_btn = crate::ui::style::constellation::MESSAGE_ACTION_BUTTON_SIZE * scale;
            let bubble_spacing = crate::ui::style::constellation::BUBBLE_SPACING * scale;
            let citation_line_height = (crate::ui::style::font_size::SMALL
                * crate::ui::style::font_size::LINE_HEIGHT_RATIO)
                * scale;
            let citation_gap = crate::ui::style::constellation::CITATION_GAP * scale;
            let max_content_base = crate::ui::style::constellation::BUBBLE_MIN_CONTENT_WIDTH * scale;
            let cache = self.constellation_layout_cache.borrow();
            let cache = cache.as_ref();
            if self.constellation_macro_active() {
                let mut best: Option<(String, f32)> = None;
                let dot_r = crate::ui::style::constellation::MACRO_NODE_RADIUS_PX * scale;
                let hit_r = dot_r + crate::ui::style::constellation::MACRO_NODE_HIT_RADIUS_PAD_PX * scale;
                for (id, node) in &graph_state.nodes {
                    let center_world = node.position + node.size * 0.5;
                    let center_screen = self.constellation_view.world_to_screen(center_world);
                    let d = (pos - center_screen).length();
                    if d <= hit_r {
                        if let Some((_, best_d)) = &best {
                            if d < *best_d {
                                best = Some((id.clone(), d));
                            }
                        } else {
                            best = Some((id.clone(), d));
                        }
                    }
                }
                if let Some((id, _)) = best {
                    return ChatHit::ConstellationNode(id);
                }
                if self.constellation_interactive_rect.contains_point(pos) {
                    return ChatHit::ConstellationBackground;
                }
                return ChatHit::Background;
            }

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

                    // Handles take precedence: top bar = move, bottom-right = resize
                    const MOVE_HANDLE_H: f32 = crate::ui::style::constellation::MOVE_HANDLE_HEIGHT;
                    const RESIZE_HANDLE_SZ: f32 = crate::ui::style::constellation::RESIZE_HANDLE_SIZE;
                    let move_rect = Rect::new(screen_pos.x, screen_pos.y, screen_size.x, MOVE_HANDLE_H * scale);
                    if move_rect.contains_point(pos) {
                        return ChatHit::ConstellationMoveHandle(id.clone());
                    }
                    let resize_rect = Rect::new(
                        screen_pos.x + screen_size.x - RESIZE_HANDLE_SZ * scale,
                        screen_pos.y + screen_size.y - RESIZE_HANDLE_SZ * scale,
                        RESIZE_HANDLE_SZ * scale,
                        RESIZE_HANDLE_SZ * scale,
                    );
                    if resize_rect.contains_point(pos) {
                        return ChatHit::ConstellationResizeHandle(id.clone());
                    }

                    // Fixed layout (no scroll) for hit test; only bubble content areas are scrollable.
                    let shard_msg_inset =
                        crate::ui::style::padding::SHARD_MESSAGE_INSET * scale;
                    let max_content_width = (screen_size.x * crate::ui::style::constellation::BUBBLE_MAX_WIDTH_RATIO
                        - (pad + shard_msg_inset) * 2.0)
                        .max(max_content_base);
                    let (user_size, assistant_size) = match cache.and_then(|c| c.get(id)) {
                        Some(&(u, a)) => (u, a),
                        None => {
                            // Sizes will be populated from graph_state.update_node_sizes; use zero
                            // here so hit-testing falls back to the current node sizes.
                            (Vec2::ZERO, Vec2::ZERO)
                        }
                    };
                    // Minimum bubble size when node has content so hide/unhide button always has a hit rect (matches render placeholder)
                    let button_reserve = crate::ui::style::constellation::BUTTON_ROW_RESERVE * scale;
                    let min_content_h = crate::ui::style::constellation::HIDDEN_PLACEHOLDER_HEIGHT * scale;
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
                    let mut y = screen_pos.y + pad + shard_msg_inset;
                    let mut hit_user_content_rect: Option<Rect> = None;
                    let mut hit_assistant_content_rect: Option<Rect> = None;
                    let content_bottom = screen_pos.y + screen_size.y - row_pad - btn - shard_msg_inset;
                    if user_size_eff.x > 0.0 && user_size_eff.y > 0.0 {
                        let bubble_w = user_size_eff.x + pad * 2.0;
                        let bubble_h = user_size_eff.y + pad * 2.0 + button_reserve;
                        let bubble_x = screen_pos.x + screen_size.x - pad - shard_msg_inset - bubble_w;
                        let bubble_y = y;
                        hit_user_content_rect = Some(Rect::new(
                            bubble_x + pad,
                            bubble_y + pad,
                            (bubble_w - pad * 2.0).max(0.0),
                            (bubble_h - pad * 2.0 - button_reserve).max(0.0),
                        ));
                        let hit_expand = crate::ui::style::constellation::MESSAGE_HIT_EXPAND * scale;
                        let visible_bottom = (bubble_y + bubble_h).min(content_bottom);
                        let edit_rect = Rect::new(
                            bubble_x + bubble_w - msg_btn * 2.0 - crate::ui::style::constellation::MESSAGE_ACTION_INSET * scale - hit_expand,
                            visible_bottom - msg_btn - crate::ui::style::constellation::MESSAGE_ACTION_INSET * scale - hit_expand,
                            msg_btn + hit_expand * 2.0,
                            msg_btn + hit_expand * 2.0,
                        );
                        let hide_rect = Rect::new(
                            bubble_x + bubble_w - msg_btn - crate::ui::style::constellation::MESSAGE_ACTION_INSET * scale - hit_expand,
                            visible_bottom - msg_btn - crate::ui::style::constellation::MESSAGE_ACTION_INSET * scale - hit_expand,
                            msg_btn + hit_expand * 2.0,
                            msg_btn + hit_expand * 2.0,
                        );
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
                        let bubble_x = screen_pos.x + pad + shard_msg_inset;
                        let bubble_y = y;
                        let hit_expand = crate::ui::style::constellation::MESSAGE_HIT_EXPAND * scale;
                        let visible_bottom = (bubble_y + bubble_h).min(content_bottom);
                        let edit_rect = Rect::new(
                            bubble_x + bubble_w - msg_btn * 2.0 - crate::ui::style::constellation::MESSAGE_ACTION_INSET * scale - hit_expand,
                            visible_bottom - msg_btn - crate::ui::style::constellation::MESSAGE_ACTION_INSET * scale - hit_expand,
                            msg_btn + hit_expand * 2.0,
                            msg_btn + hit_expand * 2.0,
                        );
                        let hide_rect = Rect::new(
                            bubble_x + bubble_w - msg_btn - crate::ui::style::constellation::MESSAGE_ACTION_INSET * scale - hit_expand,
                            visible_bottom - msg_btn - crate::ui::style::constellation::MESSAGE_ACTION_INSET * scale - hit_expand,
                            msg_btn + hit_expand * 2.0,
                            msg_btn + hit_expand * 2.0,
                        );
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
                        let bubble_x = screen_pos.x + pad + shard_msg_inset;
                        let bubble_y = y;
                        let bubble_w = assistant_size_eff.x + pad * 2.0;
                        let text_start_y = bubble_y + pad;
                        let citation_line_count = if self.constellation_citations_expanded(id) {
                            node.shard.citations.len()
                        } else {
                            1
                        };
                        for citation_idx in 0..citation_line_count {
                            let citation_y = text_start_y + assistant_size_eff.y + citation_gap + citation_idx as f32 * citation_line_height;
                            let rect = Rect::new(bubble_x + pad, citation_y, bubble_w - pad * 2.0, citation_line_height);
                            if rect.contains_point(pos) {
                                return ChatHit::ConstellationCitation(id.clone(), citation_idx);
                            }
                        }
                    }
                    // Note edit/remove (same layout as render_constellation)
                    let note_line_h = crate::ui::style::constellation::NOTE_LINE_HEIGHT * scale;
                    let notes_gap = crate::ui::style::constellation::CITATION_GAP * scale;
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
                            let edit_w = crate::ui::style::constellation::NOTE_ICON_WIDTH * scale;
                            let edit_rect = Rect::new(
                                screen_pos.x + screen_size.x - pad - crate::ui::style::constellation::NOTE_EDIT_RIGHT_OFFSET * scale,
                                line_y,
                                edit_w,
                                note_line_h,
                            );
                            let remove_rect = Rect::new(
                                screen_pos.x + screen_size.x - pad - crate::ui::style::constellation::NOTE_REMOVE_RIGHT_OFFSET * scale,
                                line_y,
                                edit_w,
                                note_line_h,
                            );
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
                        let action_area_w = screen_size.x - pad * 2.0;
                        let (btn_row, row_space) = crate::ui::style::constellation::fit_shard_action_row(
                            action_area_w,
                            btn,
                            btn_space,
                        );
                        let action_y = screen_pos.y + screen_size.y - row_pad - btn_row;
                        let start_x = screen_pos.x + pad;
                        for i in 0..4 {
                            let rect = Rect::new(
                                start_x + (btn_row + row_space) * i as f32,
                                action_y,
                                btn_row,
                                btn_row,
                            );
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
            if self.constellation_interactive_rect.contains_point(pos) {
                return ChatHit::ConstellationBackground;
            }
            return ChatHit::Background;
        }

        // Linear bubble geometry matches last frame’s [`Self::refresh_linear_message_layout`].
        let bubbles = &self.linear_message_bubbles_cache;
        
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

    /// Returns (user message text for display/API, optional system prompt override, graph mentions).
    pub fn send_message(
        &mut self,
        prompts: &[crate::state::SystemPromptEntry],
    ) -> Option<(String, Option<String>, Vec<crate::api::models::GraphMention>)> {
        let raw = self.input_field.text.clone();
        let text_empty = raw.trim().is_empty();
        if text_empty
            && self.pending_system_prompt.is_none()
            && self.pending_mentions.is_empty()
        {
            return None;
        }
        let (user_draft, system_from_parse) =
            crate::state::parse_slash_system_prompt(&raw, prompts);
        let system_prompt = self
            .pending_system_prompt
            .as_ref()
            .map(|p| p.content.clone())
            .or(system_from_parse);
        let mut mentions: Vec<GraphMention> = self
            .pending_mentions
            .iter()
            .map(|p| p.mention.clone())
            .collect();
        for m in crate::api::models::parse_graph_mentions_from_draft(&user_draft) {
            if !mentions.iter().any(|x| same_graph_mention(x, &m)) {
                mentions.push(m);
            }
        }
        let draft_clean = crate::api::models::strip_graph_mention_tokens(&user_draft);
        if draft_clean.trim().is_empty()
            && mentions.is_empty()
            && system_prompt.is_none()
        {
            return None;
        }
        self.input_field.clear();
        self.pending_system_prompt = None;
        self.pending_mentions.clear();
        self.system_prompt_dropdown.close();
        self.mention_popup_open = false;
        Some((draft_clean, system_prompt, mentions))
    }

    /// Full composer text for clipboard: tokens + body (round-trippable with `absorb_tokens_from_pasted_text`).
    pub fn serialize_composer_for_clipboard(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(p) = &self.pending_system_prompt {
            parts.push(format!("/{}", p.name));
        }
        for pm in &self.pending_mentions {
            let token = match &pm.mention {
                GraphMention::Paper { paper_id } => format!("@paper:{}", paper_id),
                GraphMention::Shard {
                    graph_id,
                    shard_id,
                } => format!("@shard:{}:{}", graph_id, shard_id),
                GraphMention::Graph { graph_id } => format!("@graph:{}", graph_id),
                GraphMention::Notepad { document_id } => format!("@notepad:{}", document_id),
            };
            parts.push(token);
        }
        let body = self.input_field.text.trim();
        if !body.is_empty() {
            parts.push(body.to_string());
        }
        parts.join(" ")
    }

    /// After paste: absorb `/name` and `@…` tokens into structured state and strip from the field.
    pub fn absorb_tokens_from_pasted_text(
        &mut self,
        prompts: &[crate::state::SystemPromptEntry],
    ) {
        self.try_absorb_leading_slash_prompt(prompts);
        let t = self.input_field.text.clone();
        let mentions = crate::api::models::parse_graph_mentions_from_draft(&t);
        let cleaned = crate::api::models::strip_graph_mention_tokens(&t);
        if cleaned != t {
            self.input_field.text = cleaned;
            self.input_field.ensure_cursor_valid();
        }
        for m in mentions {
            let label = match &m {
                GraphMention::Paper { paper_id } => format!("Paper {}", paper_id),
                GraphMention::Shard { shard_id, .. } => shard_id.clone(),
                GraphMention::Graph { graph_id } => graph_id.clone(),
                GraphMention::Notepad { document_id } => document_id.clone(),
            };
            self.push_pending_mention(PendingMention {
                mention: m,
                label,
            });
        }
    }

    fn try_absorb_leading_slash_prompt(&mut self, prompts: &[crate::state::SystemPromptEntry]) {
        let t = self.input_field.text.clone();
        let chars: Vec<char> = t.chars().collect();
        let mut start = 0usize;
        while start < chars.len() && chars[start].is_whitespace() {
            start += 1;
        }
        if start >= chars.len() || chars[start] != '/' {
            return;
        }
        let mut end = start + 1;
        while end < chars.len() && !chars[end].is_whitespace() {
            end += 1;
        }
        let token: String = chars[start + 1..end].iter().collect();
        if token.is_empty() {
            return;
        }
        let token_lower = token.to_lowercase();
        for p in prompts {
            if p.name.to_lowercase() == token_lower {
                self.pending_system_prompt = Some(PendingSystemPrompt {
                    name: p.name.clone(),
                    content: p.content.clone(),
                });
                self.strip_leading_slash_token();
                return;
            }
        }
    }

    /// After input changes: keep mention popup in sync with `@query` before cursor.
    pub fn sync_mention_popup_from_input(
        &mut self,
        papers: &[crate::api::models::ApiPaper],
        graph_state: &crate::state::GraphState,
        conversations: &[crate::state::chat::Conversation],
        notepad_documents: &[(String, String)],
    ) {
        if !self.mention_popup_open {
            return;
        }
        let t = &self.input_field.text;
        let cur = self.input_field.cursor_position.min(t.chars().count());
        let before: String = t.chars().take(cur).collect();
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
        let ip = self.input_field.position;
        let is = self.input_field.size;
        Some(crate::ui::core::Rect::new(
            ip.x,
            ip.y - h - 4.0,
            is.x,
            h,
        ))
    }

    pub fn insert_paper_mention_with_label(&mut self, paper_id: i32, label: String) {
        self.remove_active_at_mention();
        self.push_pending_mention(PendingMention {
            mention: GraphMention::Paper { paper_id },
            label,
        });
        self.mention_popup_open = false;
        self.mention_rows.clear();
    }

    pub fn insert_shard_mention_with_label(
        &mut self,
        graph_id: &str,
        shard_id: &str,
        label: String,
    ) {
        self.remove_active_at_mention();
        self.push_pending_mention(PendingMention {
            mention: GraphMention::Shard {
                graph_id: graph_id.to_string(),
                shard_id: shard_id.to_string(),
            },
            label,
        });
        self.mention_popup_open = false;
        self.mention_rows.clear();
    }

    pub fn insert_graph_mention_with_label(&mut self, graph_id: &str, label: String) {
        self.remove_active_at_mention();
        self.push_pending_mention(PendingMention {
            mention: GraphMention::Graph {
                graph_id: graph_id.to_string(),
            },
            label,
        });
        self.mention_popup_open = false;
        self.mention_rows.clear();
    }

    pub fn insert_notepad_mention_with_label(&mut self, document_id: &str, label: String) {
        self.remove_active_at_mention();
        self.push_pending_mention(PendingMention {
            mention: GraphMention::Notepad {
                document_id: document_id.to_string(),
            },
            label,
        });
        self.mention_popup_open = false;
        self.mention_rows.clear();
    }

    pub fn apply_mention_row_selection(
        &mut self,
        index: usize,
        papers: &[crate::api::models::ApiPaper],
        graph_state: &crate::state::GraphState,
        conversations: &[crate::state::chat::Conversation],
    ) {
        if index >= self.mention_rows.len() {
            return;
        }
        match self.mention_rows[index].clone() {
            MentionEntry::Paper(id) => {
                let label = papers
                    .iter()
                    .find(|p| p.id == id)
                    .map(|p| {
                        p.title
                            .as_deref()
                            .unwrap_or(p.filename.as_str())
                            .to_string()
                    })
                    .unwrap_or_else(|| format!("Paper {}", id));
                self.insert_paper_mention_with_label(id, label);
            }
            MentionEntry::Shard {
                graph_id,
                shard_id,
            } => {
                let label = format!("{}", shard_id);
                self.insert_shard_mention_with_label(&graph_id, &shard_id, label);
            }
            MentionEntry::Graph { graph_id } => {
                let label = conversations
                    .iter()
                    .find(|c| c.graph_id.as_ref() == Some(&graph_id))
                    .map(|c| c.title.clone())
                    .unwrap_or_else(|| graph_id.clone());
                self.insert_graph_mention_with_label(&graph_id, label);
            }
            MentionEntry::Notepad {
                document_id,
                title,
            } => {
                self.insert_notepad_mention_with_label(&document_id, title);
            }
        }
    }

    pub fn sync_system_prompt_dropdown(&mut self, prompts: &[crate::state::SystemPromptEntry]) {
        let t = self.input_field.text.trim();
        if !t.starts_with('/') || t.contains(' ') {
            self.system_prompt_dropdown.close();
            return;
        }
        let prefix = &t[1..];
        self.system_prompt_dropdown.items.clear();
        for p in prompts {
            if p.name.to_lowercase().starts_with(&prefix.to_lowercase()) {
                let preview: String = p.content.chars().take(40).collect();
                let preview = if p.content.chars().count() > 40 {
                    format!("{}…", preview)
                } else {
                    preview
                };
                self.system_prompt_dropdown.items.push(DropdownItem {
                    id: None,
                    label: format!("{} — {}", p.name, preview),
                    slash_name: Some(p.name.clone()),
                });
            }
        }
        if self.system_prompt_dropdown.items.is_empty() {
            self.system_prompt_dropdown.close();
            return;
        }
        self.system_prompt_dropdown.is_open = true;
        self.system_prompt_dropdown.open_animation.target = 1.0;
        self.system_prompt_dropdown.update_layout();
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
        let bubble = match self.linear_message_bubbles_cache.get(msg_idx) {
            Some(b) => b,
            None => return,
        };
        let padding = 12.0;
        let edit_max_width = bubble.size.x - padding * 2.0;
        const EDIT_FONT_SIZE: f32 = crate::ui::style::font_size::MESSAGE_BODY;
        let edit_line_height = EDIT_FONT_SIZE * crate::ui::style::font_size::LINE_HEIGHT_RATIO;
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
        const FONT_SIZE: f32 = crate::ui::style::font_size::MESSAGE_BODY;
        let line_height = FONT_SIZE * crate::ui::style::font_size::LINE_HEIGHT_RATIO;
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
        let line_height = font_size * crate::ui::style::font_size::LINE_HEIGHT_RATIO;
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
        const SEND_GAP: f32 = 8.0;
        const SEND_W: f32 = 56.0;
        const FONT_SIZE: f32 = crate::ui::style::font_size::MESSAGE_BODY;
        let line_height = FONT_SIZE * crate::ui::style::font_size::LINE_HEIGHT_RATIO;
        let total_w = screen_size.x * 2.0 / 3.0;
        let field_w = total_w - SEND_W - SEND_GAP;
        let max_content_width = field_w - PAD * 2.0;
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
            (screen_pos.x + (screen_size.x - total_w) * 0.5, note_y_above)
        } else {
            (screen_pos.x + (screen_size.x - total_w) * 0.5, note_y_below)
        };
        self.note_input.position = Vec2::new(note_x, note_y);
        self.note_input.size = Vec2::new(field_w, height);
        self.note_send_button_rect = Rect::new(
            note_x + field_w + SEND_GAP,
            note_y,
            SEND_W,
            height,
        );
    }

    pub fn delete_message(&mut self, msg_idx: usize) {
        if msg_idx < self.messages.len() {
            let removed_shard_id = self.messages[msg_idx].shard_id.clone();
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
            if let Some(shard_id) = removed_shard_id {
                let shard_still_present = self.messages.iter().any(|m| m.shard_id.as_deref() == Some(shard_id.as_str()));
                if !shard_still_present {
                    self.muted_shard_ids.remove(&shard_id);
                    self.citations_expanded_shards.remove(&shard_id);
                }
            }
            self.delete_confirm_idx = None;
        }
    }

    pub fn toggle_mute_message(&mut self, msg_idx: usize) {
        if self.muted_messages.contains(&msg_idx) {
            self.muted_messages.remove(&msg_idx);
        } else {
            self.muted_messages.insert(msg_idx);
        }
        if let Some(shard_id) = self.messages.get(msg_idx).and_then(|m| m.shard_id.as_ref()) {
            if self.muted_shard_ids.contains(shard_id) {
                self.muted_shard_ids.remove(shard_id);
            } else {
                self.muted_shard_ids.insert(shard_id.clone());
            }
        }
    }

    pub fn toggle_mute_shard(&mut self, shard_id: &str) {
        if self.muted_shard_ids.contains(shard_id) {
            self.muted_shard_ids.remove(shard_id);
            return;
        }
        self.muted_shard_ids.insert(shard_id.to_string());
    }

    pub fn is_shard_muted(&self, shard_id: &str) -> bool {
        self.muted_shard_ids.contains(shard_id)
    }

    pub fn toggle_constellation_citations(&mut self, shard_id: &str) {
        if self.citations_expanded_shards.contains(shard_id) {
            self.citations_expanded_shards.remove(shard_id);
            return;
        }
        self.citations_expanded_shards.insert(shard_id.to_string());
    }

    pub fn constellation_citations_expanded(&self, shard_id: &str) -> bool {
        self.citations_expanded_shards.contains(shard_id)
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

#[derive(Debug, Clone)]
pub enum ChatHit {
    Input,
    SendButton,
    SystemPromptMenu,
    SystemPromptItem(usize),
    ContextPoolButton,
    ContextPoolMenu,
    ContextPoolItem(usize),
    ContextPoolCreate,  // "Create new collection" button in dropdown
    MessageList,
    /// Constellation view: hit on a node (node_id).
    ConstellationNode(String),
    /// Constellation: top bar of shard (drag to move).
    ConstellationMoveHandle(String),
    /// Constellation: bottom-right corner of shard (drag to resize).
    ConstellationResizeHandle(String),
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
    /// Constellation: Send button next to note input
    ConstellationNoteSendButton,
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
    /// Slash system prompt pill (opens settings / preview — body opens modal in app).
    ComposerSystemPromptPillBody,
    ComposerSystemPromptPillClose,
    ComposerMentionPillBody(usize),
    ComposerMentionPillClose(usize),
    /// @-mention popup row index (papers / shards / graphs).
    MentionItem(usize),
}

impl ChatHit {
    /// Whether wheel events should zoom/pan the constellation viewport (graph mode), not composer chrome.
    pub fn is_constellation_wheel_viewport_target(&self) -> bool {
        match self {
            ChatHit::ConstellationUserBubbleContent(_)
            | ChatHit::ConstellationAssistantBubbleContent(_)
            | ChatHit::ConstellationBackground
            | ChatHit::ConstellationNode(_)
            | ChatHit::ConstellationMoveHandle(_)
            | ChatHit::ConstellationResizeHandle(_)
            | ChatHit::ConstellationPinButton(_)
            | ChatHit::ConstellationHideButton(_)
            | ChatHit::ConstellationNoteButton(_)
            | ChatHit::ConstellationAddContextButton(_)
            | ChatHit::ConstellationMoreButton(_)
            | ChatHit::ConstellationMessageEditButton(_, _)
            | ChatHit::ConstellationMessageHideButton(_, _)
            | ChatHit::ConstellationCitation(_, _)
            | ChatHit::ConstellationEditNote(_, _)
            | ChatHit::ConstellationRemoveNote(_, _)
            | ChatHit::ConstellationEditTextarea
            | ChatHit::ConstellationNoteInput
            | ChatHit::ConstellationNoteSendButton => true,
            _ => false,
        }
    }
}

/// One pill chip above the input: layout + hit targets for body vs close.
#[derive(Clone, Debug)]
pub struct ComposerPillItem {
    pub body_rect: Rect,
    pub close_rect: Rect,
    pub label: String,
    pub hit_body: ChatHit,
    pub hit_close: ChatHit,
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


