use crate::ui::chat_window::ChatMessage;
use crate::state::shard::Shard;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    /// Constellar graph ID for this conversation (one graph per conversation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<String>,
    /// Shards in this conversation (ordered by creation time); used when graph_id is None (legacy).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shards: Vec<Shard>,
    pub created_at: u64,  // Timestamp
}

impl Conversation {
    /// True if no messages have been sent (no shards). Used to avoid persisting empty conversations.
    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct ChatState {
    pub conversations: Vec<Conversation>,
    pub current_conversation_id: Option<String>,
}

impl ChatState {
    pub fn new() -> Self {
        Self {
            conversations: Vec::new(),
            current_conversation_id: None,
        }
    }
    
    /// Generate a title from the first user message
    /// Similar to UI version: removes markdown, extracts meaningful words
    pub fn generate_title(text: &str) -> String {
        // Remove markdown characters and condense whitespace
        let clean = text
            .chars()
            .filter(|c| !matches!(c, '#' | '*' | '_' | '`' | '[' | ']' | '(' | ')'))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
            .trim()
            .to_string();
        
        let words: Vec<&str> = clean.split_whitespace().collect();
        let stop_words: std::collections::HashSet<&str> = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "this", "that", "these", "those"
        ].iter().cloned().collect();
        
        let meaningful: Vec<String> = words
            .iter()
            .filter_map(|w| {
                let wc = w.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect::<String>();
                if wc.len() > 2 && !stop_words.contains(wc.as_str()) && !wc.chars().all(|c| c.is_ascii_digit()) {
                    // Capitalize first letter
                    let mut chars: Vec<char> = w.chars().collect();
                    if let Some(first) = chars.first_mut() {
                        *first = first.to_uppercase().next().unwrap_or(*first);
                    }
                    Some(chars.iter().collect())
                } else {
                    None
                }
            })
            .take(4)
            .collect();
        
        if !meaningful.is_empty() {
            meaningful.join(" ")
        } else {
            // Fallback to first 20 characters
            clean.chars().take(20).collect()
        }
    }

    pub fn create_conversation(&mut self) -> String {
        let id = format!("conv_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs());
        let conversation = Conversation {
            id: id.clone(),
            title: "New Conversation".to_string(),
            graph_id: None,  // Set by caller after create_graph()
            shards: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        self.conversations.push(conversation);
        self.current_conversation_id = Some(id.clone());
        id
    }
    
    pub fn get_current_conversation(&mut self) -> Option<&mut Conversation> {
        if let Some(ref id) = self.current_conversation_id {
            self.conversations.iter_mut().find(|c| c.id == *id)
        } else {
            None
        }
    }

    pub fn add_message_to_current(&mut self, message: ChatMessage) {
        // Convert message to shard and add
        let shard = message.to_shard();
        self.add_shard_to_current(shard);
    }
    
    /// Add a shard to the current conversation
    pub fn add_shard_to_current(&mut self, mut shard: Shard) {
        if let Some(conv) = self.get_current_conversation() {
            // Set parent relationship if there's a previous shard
            if let Some(prev_shard) = conv.shards.last() {
                shard.set_parent(prev_shard.id.clone());
                // Update previous shard to have this as child
                if let Some(prev) = conv.shards.last_mut() {
                    prev.add_child(shard.id.clone());
                }
            }
            
            conv.shards.push(shard);
            
            // Update title if it's still the default
            if conv.title == "New Conversation" && conv.shards.len() == 1 {
                if let Some(first_shard) = conv.shards.first() {
                    if matches!(first_shard.metadata.role, crate::ui::chat_window::MessageRole::User) {
                        conv.title = Self::generate_title(&first_shard.text);
                    }
                }
            }
        } else {
            // Create new conversation if none exists
            self.create_conversation();
            if let Some(conv) = self.get_current_conversation() {
                conv.shards.push(shard);
            }
        }
    }
    
    /// Get a shard by ID from the current conversation
    pub fn get_shard(&self, shard_id: &str) -> Option<&Shard> {
        if let Some(ref id) = self.current_conversation_id {
            if let Some(conv) = self.conversations.iter().find(|c| c.id == *id) {
                return conv.shards.iter().find(|s| s.id == shard_id);
            }
        }
        None
    }
    
    /// Get a mutable shard by ID from the current conversation
    pub fn get_shard_mut(&mut self, shard_id: &str) -> Option<&mut Shard> {
        if let Some(ref id) = self.current_conversation_id {
            if let Some(conv) = self.conversations.iter_mut().find(|c| c.id == *id) {
                return conv.shards.iter_mut().find(|s| s.id == shard_id);
            }
        }
        None
    }
    
    /// Link two shards as friends (additional context, not parent-child related)
    pub fn link_shards_as_friends(&mut self, shard_id1: &str, shard_id2: &str) {
        if let Some(shard1) = self.get_shard_mut(shard_id1) {
            shard1.add_friend(shard_id2.to_string());
        }
        if let Some(shard2) = self.get_shard_mut(shard_id2) {
            shard2.add_friend(shard_id1.to_string());
        }
    }
    
    /// Compute embedding for a shard (async, should be called from async context)
    /// This is a placeholder - actual implementation will call API client
    pub async fn compute_embedding_for_shard(
        &mut self,
        shard_id: &str,
        api_client: &crate::api::client::ApiClient,
    ) -> Result<(), String> {
        if let Some(shard) = self.get_shard_mut(shard_id) {
            if shard.has_embedding() {
                return Ok(()); // Already has embedding
            }
            
            // Compute embedding via API
            match api_client.compute_embedding(&shard.text).await {
                Ok(embedding) => {
                    shard.set_embedding(embedding);
                    Ok(())
                }
                Err(e) => Err(format!("Failed to compute embedding: {}", e)),
            }
        } else {
            Err(format!("Shard {} not found", shard_id))
        }
    }

    pub fn delete_conversation(&mut self, id: &str) {
        self.conversations.retain(|c| c.id != id);
        if self.current_conversation_id.as_ref() == Some(&id.to_string()) {
            self.current_conversation_id = self.conversations.first().map(|c| c.id.clone());
        }
    }

    pub fn switch_conversation(&mut self, id: &str) {
        if self.conversations.iter().any(|c| c.id == id) {
            self.current_conversation_id = Some(id.to_string());
        }
    }
    
    /// Update a message in the current conversation
    pub fn update_message(&mut self, idx: usize, updater: impl FnOnce(&mut ChatMessage)) {
        if let Some(conv) = self.get_current_conversation() {
            if idx < conv.shards.len() {
                // Convert shard to message, update, convert back
                let mut msg = ChatMessage::from_shard(&conv.shards[idx]);
                updater(&mut msg);
                conv.shards[idx] = msg.to_shard();
                // Preserve ID
                conv.shards[idx].id = conv.shards[idx].id.clone();
            }
        }
    }
    
    /// Delete a message from the current conversation
    pub fn delete_message(&mut self, idx: usize) {
        if let Some(conv) = self.get_current_conversation() {
            if idx < conv.shards.len() {
                let removed_shard = conv.shards.remove(idx);
                // Update parent/child relationships
                if let Some(parent_id) = &removed_shard.parent_id {
                    if let Some(parent) = conv.shards.iter_mut().find(|s| s.id == *parent_id) {
                        parent.children_ids.retain(|id| id != &removed_shard.id);
                    }
                }
                // Update children to remove parent reference
                for child_id in &removed_shard.children_ids {
                    if let Some(child) = conv.shards.iter_mut().find(|s| s.id == *child_id) {
                        child.parent_id = removed_shard.parent_id.clone();
                    }
                }
            }
        }
    }
    
    /// Get messages from current conversation (converted from shards)
    pub fn get_current_messages(&self) -> Vec<ChatMessage> {
        if let Some(ref id) = self.current_conversation_id {
            if let Some(conv) = self.conversations.iter().find(|c| c.id == *id) {
                // Convert shards to messages
                return conv.shards.iter().map(|s| ChatMessage::from_shard(s)).collect();
            }
        }
        Vec::new()
    }
    
    /// Set messages for current conversation (used to sync with ChatWindow)
    /// Converts messages to shards
    pub fn set_current_messages(&mut self, messages: Vec<ChatMessage>) {
        if let Some(conv) = self.get_current_conversation() {
            // Convert messages to shards
            let mut new_shards: Vec<Shard> = messages.iter().map(|m| m.to_shard()).collect();
            // Set up parent/child relationships
            for i in 1..new_shards.len() {
                let prev_id = new_shards[i - 1].id.clone();
                let current_id = new_shards[i].id.clone();
                new_shards[i].set_parent(prev_id.clone());
                new_shards[i - 1].add_child(current_id);
            }
            conv.shards = new_shards;
        }
    }
    
    /// Get shards from current conversation
    pub fn get_current_shards(&self) -> Vec<Shard> {
        if let Some(ref id) = self.current_conversation_id {
            if let Some(conv) = self.conversations.iter().find(|c| c.id == *id) {
                return conv.shards.clone();
            }
        }
        Vec::new()
    }
    
    /// Finalize a conversation - ensure it has a proper title
    /// If title is None, generates one from the first user message
    pub fn finalize_conversation(&mut self, title: Option<String>) -> Option<String> {
        if let Some(conv) = self.get_current_conversation() {
            // If title is still default or empty, generate/update it
            if conv.title == "New Conversation" || conv.title.is_empty() {
                if let Some(title) = title {
                    conv.title = title;
                } else {
                    // Generate title from first user shard
                    if let Some(first_shard) = conv.shards.iter()
                        .find(|s| matches!(s.metadata.role, crate::ui::chat_window::MessageRole::User)) {
                        conv.title = Self::generate_title(&first_shard.text);
                    } else {
                        conv.title = "New Conversation".to_string();
                    }
                }
            }
            return Some(conv.id.clone());
        } else {
            // No current conversation, create one
            let id = self.create_conversation();
            if let Some(conv) = self.get_current_conversation() {
                if let Some(title) = title {
                    conv.title = title;
                } else {
                    conv.title = "New Conversation".to_string();
                }
            }
            return Some(id);
        }
    }
}
