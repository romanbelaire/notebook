/// List components for dynamic content (conversations, messages, documents)
/// These components cache item components and update them when data changes
use glam::Vec2;
use crate::ui::core::Rect;
use crate::gfx::types::Vertex;
use crate::gfx::renderer::Renderer;
use crate::app::App;
use crate::ui::components::Renderable;
use std::collections::HashMap;

/// Conversation list component that caches conversation item components
pub struct ConversationListComponent {
    component_id: String,
    item_components: HashMap<String, Box<dyn Renderable>>, // Map conversation ID to component
    last_conversation_ids: Vec<String>, // Track last known IDs for diffing
}

impl ConversationListComponent {
    pub fn new() -> Self {
        Self {
            component_id: "conversation_list".to_string(),
            item_components: HashMap::new(),
            last_conversation_ids: Vec::new(),
        }
    }
    
    /// Update the list based on current conversations
    /// Creates new components for new conversations, removes components for deleted ones
    pub fn update(&mut self, conversations: &[crate::state::chat::Conversation]) {
        let current_ids: Vec<String> = conversations.iter().map(|c| c.id.clone()).collect();
        
        // Remove components for conversations that no longer exist
        let ids_to_remove: Vec<String> = self.last_conversation_ids.iter()
            .filter(|id| !current_ids.contains(id))
            .cloned()
            .collect();
        for id in ids_to_remove {
            self.item_components.remove(&id);
        }
        
        // Create components for new conversations
        for conv in conversations {
            if !self.item_components.contains_key(&conv.id) {
                // Create new conversation item component
                let item = ConversationItemComponent::new(conv.id.clone());
                self.item_components.insert(conv.id.clone(), Box::new(item));
            }
        }
        
        self.last_conversation_ids = current_ids;
    }
}

impl Renderable for ConversationListComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("sidebar_content"), "ConversationListComponent");
        renderer.push_parent(self.component_id.clone());
        for (id, item) in &self.item_components {
            let item_id = format!("conversation_item_{}", id);
            renderer.validate_component(&item_id, Some(&self.component_id), "ConversationItemComponent");
            renderer.push_parent(item_id.clone());
            item.render(renderer, app, vertices, dirty_rect);
            renderer.pop_parent();
        }
        renderer.pop_parent();
    }
    
    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }
    
    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

impl ConversationListComponent {
    /// Update the component based on current app state
    /// Should be called before rendering when data may have changed
    pub fn update_from_app(&mut self, app: &App) {
        self.update(&app.chat_state.conversations);
    }
}

/// Individual conversation item component
struct ConversationItemComponent {
    conversation_id: String,
    component_id: String,
}

impl ConversationItemComponent {
    fn new(conversation_id: String) -> Self {
        Self {
            conversation_id: conversation_id.clone(),
            component_id: format!("conversation_item_{}", conversation_id),
        }
    }
}

impl Renderable for ConversationItemComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, _dirty_rect: Option<Rect>) {
        let conversation = app.chat_state.conversations.iter()
            .find(|c| c.id == self.conversation_id);
        if conversation.is_some() {
            renderer.validate_component(&self.component_id, None, "ConversationItemComponent");
        }
    }
    
    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }
    
    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 40.0) // Standard item height
    }
}

/// Message list component that caches message item components
pub struct MessageListComponent {
    component_id: String,
    item_components: HashMap<usize, Box<dyn Renderable>>, // Map message index to component
    last_message_count: usize,
}

impl MessageListComponent {
    pub fn new() -> Self {
        Self {
            component_id: "message_list".to_string(),
            item_components: HashMap::new(),
            last_message_count: 0,
        }
    }
    
    /// Update the list based on current messages
    pub fn update(&mut self, message_count: usize) {
        // Remove components for messages that no longer exist
        if message_count < self.last_message_count {
            let indices_to_remove: Vec<usize> = self.item_components.keys()
                .filter(|&&idx| idx >= message_count)
                .cloned()
                .collect();
            for idx in indices_to_remove {
                self.item_components.remove(&idx);
            }
        }
        
        // Create components for new messages
        for idx in self.last_message_count..message_count {
            let item = MessageItemComponent::new(idx);
            self.item_components.insert(idx, Box::new(item));
        }
        
        self.last_message_count = message_count;
    }
    
    /// Update the component based on current app state
    pub fn update_from_app(&mut self, app: &App) {
        let message_count = if let Some(ref chat) = app.chat_window {
            chat.messages.len()
        } else {
            0
        };
        self.update(message_count);
    }
}

impl Renderable for MessageListComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("chat"), "MessageListComponent");
        renderer.push_parent(self.component_id.clone());
        for (idx, item) in &self.item_components {
            let item_id = format!("message_item_{}", idx);
            renderer.validate_component(&item_id, Some(&self.component_id), "MessageItemComponent");
            renderer.push_parent(item_id.clone());
            item.render(renderer, app, vertices, dirty_rect);
            renderer.pop_parent();
        }
        renderer.pop_parent();
    }
    
    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }
    
    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Individual message item component
struct MessageItemComponent {
    message_index: usize,
    component_id: String,
}

impl MessageItemComponent {
    fn new(message_index: usize) -> Self {
        Self {
            message_index,
            component_id: format!("message_item_{}", message_index),
        }
    }
}

impl Renderable for MessageItemComponent {
    fn render(&self, renderer: &mut Renderer, _app: &App, _vertices: &mut Vec<Vertex>, _dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, None, "MessageItemComponent");
    }
    
    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }
    
    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Document list component that caches document item components
pub struct DocumentListComponent {
    component_id: String,
    item_components: HashMap<String, Box<dyn Renderable>>, // Map document ID to component
    last_document_ids: Vec<String>,
}

impl DocumentListComponent {
    pub fn new() -> Self {
        Self {
            component_id: "document_list".to_string(),
            item_components: HashMap::new(),
            last_document_ids: Vec::new(),
        }
    }
    
    /// Update the list based on current documents
    pub fn update(&mut self, document_ids: &[String]) {
        let current_ids: Vec<String> = document_ids.to_vec();
        
        // Remove components for documents that no longer exist
        let ids_to_remove: Vec<String> = self.last_document_ids.iter()
            .filter(|id| !current_ids.contains(id))
            .cloned()
            .collect();
        for id in ids_to_remove {
            self.item_components.remove(&id);
        }
        
        // Create components for new documents
        for doc_id in document_ids {
            if !self.item_components.contains_key(doc_id) {
                let item = DocumentItemComponent::new(doc_id.clone());
                self.item_components.insert(doc_id.clone(), Box::new(item));
            }
        }
        
        self.last_document_ids = current_ids;
    }
    
    /// Update the component based on current app state
    pub fn update_from_app(&mut self, app: &App) {
        use crate::persistence::DocumentPersistence;
        let document_ids: Vec<String> = DocumentPersistence::list_documents()
            .unwrap_or_default();
        self.update(&document_ids);
    }
}

impl Renderable for DocumentListComponent {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, Some("sidebar_content"), "DocumentListComponent");
        renderer.push_parent(self.component_id.clone());
        for (id, item) in &self.item_components {
            let item_id = format!("document_item_{}", id);
            renderer.validate_component(&item_id, Some(&self.component_id), "DocumentItemComponent");
            renderer.push_parent(item_id.clone());
            item.render(renderer, app, vertices, dirty_rect);
            renderer.pop_parent();
        }
        renderer.pop_parent();
    }
    
    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }
    
    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

/// Individual document item component
struct DocumentItemComponent {
    document_id: String,
    component_id: String,
}

impl DocumentItemComponent {
    fn new(document_id: String) -> Self {
        Self {
            document_id: document_id.clone(),
            component_id: format!("document_item_{}", document_id),
        }
    }
}

impl Renderable for DocumentItemComponent {
    fn render(&self, renderer: &mut Renderer, _app: &App, _vertices: &mut Vec<Vertex>, _dirty_rect: Option<Rect>) {
        renderer.validate_component(&self.component_id, None, "DocumentItemComponent");
    }
    
    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }
    
    fn update_layout(&mut self, _available_rect: Rect, _dirty_rect: Option<Rect>, _app: Option<&App>) {
    }
    
    fn min_size(&self) -> Vec2 {
        Vec2::new(0.0, 40.0)
    }
}

