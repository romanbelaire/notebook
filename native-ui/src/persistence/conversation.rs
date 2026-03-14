use crate::state::chat::{Conversation, ChatState};
use crate::persistence::{get_data_subdir};
use std::path::PathBuf;
use std::fs;
use serde_json;

pub struct ConversationPersistence;

impl ConversationPersistence {
    /// Get the conversations directory path
    fn get_conversations_dir() -> Result<PathBuf, std::io::Error> {
        get_data_subdir("conversations")
    }

    /// Get the file path for a conversation by ID
    fn get_conversation_path(id: &str) -> Result<PathBuf, std::io::Error> {
        let dir = Self::get_conversations_dir()?;
        Ok(dir.join(format!("{}.json", id)))
    }

    /// Save a conversation to disk
    pub fn save_conversation(conversation: &Conversation) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::get_conversation_path(&conversation.id)?;
        let json = serde_json::to_string_pretty(conversation)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Load a conversation from disk
    pub fn load_conversation(id: &str) -> Result<Conversation, Box<dyn std::error::Error>> {
        let path = Self::get_conversation_path(id)?;
        if !path.exists() {
            return Err(format!("Conversation {} not found", id).into());
        }
        let json = fs::read_to_string(&path)?;
        let conversation: Conversation = serde_json::from_str(&json)?;
        Ok(conversation)
    }

    /// Delete a conversation from disk
    pub fn delete_conversation(id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::get_conversation_path(id)?;
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// Load all conversations from disk
    pub fn load_all_conversations() -> Result<Vec<Conversation>, Box<dyn std::error::Error>> {
        let dir = Self::get_conversations_dir()?;
        let mut conversations = Vec::new();
        
        if !dir.exists() {
            return Ok(conversations);
        }

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                match Self::load_conversation_from_path(&path) {
                    Ok(conv) => conversations.push(conv),
                    Err(e) => eprintln!("Failed to load conversation from {:?}: {}", path, e),
                }
            }
        }
        
        // Sort by created_at (newest first)
        conversations.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        Ok(conversations)
    }

    /// Load a conversation from a specific path
    fn load_conversation_from_path(path: &std::path::Path) -> Result<Conversation, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let conversation: Conversation = serde_json::from_str(&json)?;
        Ok(conversation)
    }

    /// Save the entire chat state (non-empty conversations only; empty ones are not persisted).
    /// Removes from disk any conversation that is now empty.
    pub fn save_chat_state(state: &ChatState) -> Result<(), Box<dyn std::error::Error>> {
        for conversation in &state.conversations {
            if conversation.is_empty() {
                let _ = Self::delete_conversation(&conversation.id);
                continue;
            }
            Self::save_conversation(conversation)?;
        }
        Ok(())
    }

    /// Load the entire chat state (all conversations)
    pub fn load_chat_state() -> Result<ChatState, Box<dyn std::error::Error>> {
        let conversations = Self::load_all_conversations()?;
        Ok(ChatState {
            conversations,
            current_conversation_id: None,
        })
    }
}

