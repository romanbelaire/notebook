use glam::Vec2;
use crate::utils::animation::SpringAnimation;
use crate::ui::{SectionList, InsightsPanel};

pub struct SidebarWindow {
    pub position: Vec2,
    pub target_width: f32,
    pub current_width: f32,
    pub height: f32,
    pub is_open: bool,
    pub width_animation: SpringAnimation,
    pub conversations_list: SectionList,
    pub documents_list: SectionList,
    pub insights_panel: InsightsPanel,
    pub selected_conversation_id: Option<String>,
    pub selected_document_id: Option<String>,
    pub selected_insight_id: Option<String>,
    pub hovered_conversation_index: Option<usize>,
    pub hovered_document_index: Option<usize>,
    pub hovered_insight_id: Option<String>,
    pub new_conversation_button: crate::ui::Button,
    pub new_document_button: crate::ui::Button,
    pub delete_conversation_button: crate::ui::Button,
    pub delete_document_button: crate::ui::Button,
}

impl SidebarWindow {
    pub const OPEN_WIDTH: f32 = 288.0;  // 18rem = 288px
    const CLOSED_WIDTH: f32 = 1.0;  // 1px for shadow visibility

    pub fn new(position: Vec2, height: f32) -> Self {
        let mut width_animation = SpringAnimation::new(Self::OPEN_WIDTH);
        width_animation.target = Self::OPEN_WIDTH;

        let title_height = 40.0; // Section title height
        let item_height = 40.0;
        let max_items_visible = 6; // Based on max_content_height of 250px / 40px item height
        let section_spacing = 16.0; // style::padding::LARGE
        
        // Calculate section heights to match the Section system
        let conversations_content_height = (max_items_visible as f32 * item_height).min(250.0);
        let conversations_total = title_height + conversations_content_height;
        
        let documents_content_height = (max_items_visible as f32 * item_height).min(250.0);
        let documents_total = title_height + documents_content_height;
        
        // SectionList positions must match Section layout exactly:
        // Section 0 (Conversations): y_offset=0, content at position.y + 0 + title_height
        let conv_pos_y = position.y + title_height;
        let conversations_list = SectionList::new(
            Vec2::new(position.x, conv_pos_y),
            Vec2::new(Self::OPEN_WIDTH, conversations_content_height),
            item_height,
        );

        // Section 1 (Documents): y_offset = conversations_total + spacing
        let documents_y_offset = conversations_total + section_spacing;
        let documents_list = SectionList::new(
            Vec2::new(position.x, position.y + documents_y_offset + title_height),
            Vec2::new(Self::OPEN_WIDTH, documents_content_height),
            item_height,
        );

        // Section 2 (Insights): y_offset = documents_y_offset + documents_total + spacing
        let insights_y_offset = documents_y_offset + documents_total + section_spacing;
        let insights_panel = InsightsPanel::new(
            Vec2::new(position.x, position.y + insights_y_offset + title_height),
            Vec2::new(Self::OPEN_WIDTH, height - insights_y_offset - title_height),
        );

        let start_y = position.y + title_height;
        let docs_start_y = position.y + documents_y_offset + title_height;
        let button_size = Vec2::new(30.0, 30.0);
        let button_padding = 10.0; // Padding for button positioning
        
        let new_conv_button = crate::ui::Button::new(
            Vec2::new(position.x + button_padding, start_y - 35.0),
            button_size,
            "+",
        );
        let new_doc_button = crate::ui::Button::new(
            Vec2::new(position.x + button_padding, docs_start_y - 35.0),
            button_size,
            "+",
        );
        let delete_conv_button = crate::ui::Button::new(
            Vec2::new(position.x + Self::OPEN_WIDTH - button_padding - button_size.x, start_y - 35.0),
            button_size,
            "×",
        );
        let delete_doc_button = crate::ui::Button::new(
            Vec2::new(position.x + Self::OPEN_WIDTH - button_padding - button_size.x, docs_start_y - 35.0),
            button_size,
            "×",
        );

        Self {
            position,
            target_width: Self::OPEN_WIDTH,
            current_width: Self::OPEN_WIDTH,
            height,
            is_open: true,
            width_animation,
            conversations_list,
            documents_list,
            insights_panel,
            selected_conversation_id: None,
            selected_document_id: None,
            selected_insight_id: None,
            hovered_conversation_index: None,
            hovered_document_index: None,
            hovered_insight_id: None,
            new_conversation_button: new_conv_button,
            new_document_button: new_doc_button,
            delete_conversation_button: delete_conv_button,
            delete_document_button: delete_doc_button,
        }
    }

    pub fn toggle(&mut self) {
        self.is_open = !self.is_open;
        self.target_width = if self.is_open {
            Self::OPEN_WIDTH
        } else {
            Self::CLOSED_WIDTH
        };
        self.width_animation.target = self.target_width;
    }

    pub fn set_open(&mut self, open: bool) {
        if self.is_open != open {
            self.toggle();
        }
    }

    pub fn update(&mut self, dt: f32, conversation_count: usize, document_count: usize, insights_count: usize, conversations: &[crate::state::chat::Conversation], document_ids: &[String], insights: &[crate::api::models::Insight]) {
        self.width_animation.update(dt);
        self.current_width = self.width_animation.value;
        self.conversations_list.update(dt, conversation_count);
        self.documents_list.update(dt, document_count);
        self.insights_panel.insights_list.update(dt, insights_count);
        self.update_selection_borders(conversations, document_ids, insights);
    }

    /// True if sidebar width or any list scroll/expand animation is active (needs continuous redraw).
    pub fn has_active_animation(&self) -> bool {
        if !self.width_animation.is_at_target() {
            return true;
        }
        if self.conversations_list.has_active_animation() {
            return true;
        }
        if self.documents_list.has_active_animation() {
            return true;
        }
        if self.insights_panel.insights_list.has_active_animation() {
            return true;
        }
        false
    }

    pub fn update_layout(&mut self, header_height: f32, conversations: &[crate::state::chat::Conversation], documents: &[String], insights: &[crate::api::models::Insight]) {
        // Match the Section system layout exactly
        let title_height = 40.0;
        let item_height = 40.0;
        let max_items_visible = 6;
        let section_spacing = 16.0; // style::padding::LARGE
        
        // Calculate section heights to match Section system
        let conversations_content_height = (max_items_visible as f32 * item_height).min(250.0);
        let conversations_total = title_height + conversations_content_height;
        
        let documents_content_height = (max_items_visible as f32 * item_height).min(250.0);
        let documents_total = title_height + documents_content_height;
        
        // Section 0 (Conversations): y_offset=0, content at position.y + 0 + title_height
        let conv_y = self.position.y + title_height;
        self.conversations_list.set_position_size(
            Vec2::new(self.position.x, conv_y),
            Vec2::new(self.current_width, conversations_content_height),
        );

        // Section 1 (Documents): y_offset = conversations_total + spacing
        let documents_y_offset = conversations_total + section_spacing;
        self.documents_list.set_position_size(
            Vec2::new(self.position.x, self.position.y + documents_y_offset + title_height),
            Vec2::new(self.current_width, documents_content_height),
        );

        // Update button positions to match rendered positions using layout functions
        use crate::ui::core::{Rect, layout};
        let button_size = Vec2::new(30.0, 30.0);
        let padding = 16.0; // style::padding::MEDIUM
        
        // For conversations section: button is positioned using stack_horizontal in render
        // We approximate title_text_width (actual measurement happens in render, but this is close enough)
        let title_text_width_approx = 120.0; // Approximate width of "Conversations" text
        let spacing = padding; // stack_horizontal uses padding as spacing
        
        // Create title rect for conversations section
        let conversations_title_rect = Rect::new(
            self.position.x,
            self.position.y,
            self.current_width,
            title_height,
        );
        
        // Position new conversation button using stack_horizontal logic
        let title_container = Rect::new(
            conversations_title_rect.x + padding,
            conversations_title_rect.y,
            conversations_title_rect.width - padding * 2.0,
            conversations_title_rect.height,
        );
        let title_rects = layout::stack_horizontal(
            &title_container,
            &[title_text_width_approx, button_size.x],
            spacing,
            0.0,
        );
        
        if let Some(button_rect) = title_rects.get(1) {
            // Center button vertically in title area
            let button_y = layout::center_y(&conversations_title_rect, button_size.y);
            self.new_conversation_button.position = Vec2::new(button_rect.x, button_y);
        }
        
        // Delete buttons are not rendered in title area anymore (they're in expanded item actions)
        // But keep their positions updated for hover detection if needed
        let delete_button_x = layout::align_right(&conversations_title_rect, button_size.x, padding);
        let delete_button_y = layout::center_y(&conversations_title_rect, button_size.y);
        self.delete_conversation_button.position = Vec2::new(delete_button_x, delete_button_y);
        
        // For documents section: button uses title_rect.right() - button_size.x - padding
        let documents_y_offset = conversations_total + section_spacing;
        let documents_title_rect = Rect::new(
            self.position.x,
            self.position.y + documents_y_offset,
            self.current_width,
            title_height,
        );
        
        let new_doc_button_x = layout::align_right(&documents_title_rect, button_size.x, padding);
        let new_doc_button_y = layout::center_y(&documents_title_rect, button_size.y);
        self.new_document_button.position = Vec2::new(new_doc_button_x, new_doc_button_y);
        
        self.delete_document_button.position = Vec2::new(new_doc_button_x, new_doc_button_y);

        // Insights section: match Section layout so hit-test aligns with render
        const INSIGHTS_ITEM_HEIGHT: f32 = 35.0;
        let insights_y_offset = documents_y_offset + documents_total + section_spacing;
        let insights_pos = Vec2::new(self.position.x, self.position.y + insights_y_offset + title_height);
        let insights_size = Vec2::new(
            self.current_width,
            (self.height - insights_y_offset - title_height).max(0.0),
        );
        self.insights_panel.update_layout(insights_pos, insights_size);
    }

    pub fn get_conversation_at(&self, pos: Vec2, conversations: &[crate::state::chat::Conversation]) -> Option<usize> {
        self.conversations_list.get_item_at(pos, conversations.len())
    }

    pub fn get_document_at(&self, pos: Vec2, document_ids: &[String]) -> Option<usize> {
        self.documents_list.get_item_at(pos, document_ids.len())
    }

    pub fn hit_test(&self, pos: Vec2) -> bool {
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.current_width && rel.y >= 0.0 && rel.y <= self.height
    }

    pub fn bounds(&self) -> (Vec2, Vec2) {
        (self.position, Vec2::new(self.current_width, self.height))
    }

    pub fn get_new_conversation_button_at(&self, pos: Vec2) -> bool {
        self.new_conversation_button.contains(pos)
    }

    pub fn get_new_document_button_at(&self, pos: Vec2) -> bool {
        self.new_document_button.contains(pos)
    }

    pub fn get_delete_conversation_button_at(&self, pos: Vec2) -> bool {
        self.delete_conversation_button.contains(pos)
    }

    pub fn get_delete_document_button_at(&self, pos: Vec2) -> bool {
        self.delete_document_button.contains(pos)
    }

    pub fn get_insight_at(&self, pos: Vec2, insights: &[crate::api::models::Insight]) -> Option<usize> {
        self.insights_panel.get_insight_at(pos, insights)
    }

    pub fn update_hover_state(&mut self, mouse_pos: Vec2, conversations: &[crate::state::chat::Conversation], document_ids: &[String], insights: &[crate::api::models::Insight]) {
        if let Some(index) = self.get_conversation_at(mouse_pos, conversations) {
            self.hovered_conversation_index = Some(index);
            let highlight_y = self.conversations_list.item_y_for_index(index);
            self.conversations_list.set_highlight_target(highlight_y);
        } else {
            self.hovered_conversation_index = None;
            if !self.conversations_list.contains(mouse_pos) {
                self.conversations_list.clear_highlight();
            }
        }
        if let Some(index) = self.get_document_at(mouse_pos, document_ids) {
            self.hovered_document_index = Some(index);
            let highlight_y = self.documents_list.item_y_for_index(index);
            self.documents_list.set_highlight_target(highlight_y);
        } else {
            self.hovered_document_index = None;
            if !self.documents_list.contains(mouse_pos) {
                self.documents_list.clear_highlight();
            }
        }
        if let Some(index) = self.insights_panel.get_insight_at(mouse_pos, insights) {
            self.hovered_insight_id = Some(insights[index].id.clone());
            let highlight_y = self.insights_panel.insights_list.item_y_for_index(index);
            self.insights_panel.insights_list.set_highlight_target(highlight_y);
        } else {
            self.hovered_insight_id = None;
            if !self.insights_panel.insights_list.contains(mouse_pos) {
                self.insights_panel.insights_list.clear_highlight();
            }
        }
    }

    /// Update selection border for all three lists based on selected ids.
    pub fn update_selection_borders(&mut self, conversations: &[crate::state::chat::Conversation], document_ids: &[String], insights: &[crate::api::models::Insight]) {
        if let Some(ref selected_id) = self.selected_conversation_id {
            if let Some(index) = conversations.iter().position(|c| c.id == *selected_id) {
                let y = self.conversations_list.item_y_for_index(index);
                self.conversations_list.set_selection_border_target(y);
            } else {
                self.conversations_list.clear_selection_border();
            }
        } else {
            self.conversations_list.clear_selection_border();
        }
        if let Some(ref selected_id) = self.selected_document_id {
            if let Some(index) = document_ids.iter().position(|id| id == selected_id) {
                let y = self.documents_list.item_y_for_index(index);
                self.documents_list.set_selection_border_target(y);
            } else {
                self.documents_list.clear_selection_border();
            }
        } else {
            self.documents_list.clear_selection_border();
        }
        if let Some(ref selected_id) = self.selected_insight_id {
            if let Some(index) = insights.iter().position(|i| i.id == *selected_id) {
                let y = self.insights_panel.insights_list.item_y_for_index(index);
                self.insights_panel.insights_list.set_selection_border_target(y);
            } else {
                self.insights_panel.insights_list.clear_selection_border();
            }
        } else {
            self.insights_panel.insights_list.clear_selection_border();
        }
    }
}

