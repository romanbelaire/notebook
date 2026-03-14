use serde::{Deserialize, Serialize};
use crate::ui::chat_window::{ChatMessage, MessageRole};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub query: String,
    pub history: Vec<ApiChatMessage>,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_model: Option<String>,
}

fn default_provider() -> String {
    "local".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contexts: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<Vec<String>>,
}

impl From<&ChatMessage> for ApiChatMessage {
    fn from(msg: &ChatMessage) -> Self {
        ApiChatMessage {
            role: match msg.role {
                MessageRole::User => "user".to_string(),
                MessageRole::Assistant => "assistant".to_string(),
            },
            content: msg.content.clone(),
            contexts: if msg.contexts.is_empty() {
                None
            } else {
                Some(msg.contexts.clone())
            },
            citations: if msg.citations.is_empty() {
                None
            } else {
                Some(msg.citations.iter().map(|c| {
                    serde_json::json!({
                        "text": c.text,
                        "source": c.source,
                    })
                }).collect())
            },
            notes: if msg.notes.is_empty() {
                None
            } else {
                Some(msg.notes.clone())
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub answer: String,
    #[serde(default)]
    pub contexts: Vec<String>,
    #[serde(default)]
    pub citations: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Collection {
    pub id: i32,
    pub name: String,
    pub paper_count: i32,
    #[serde(default)]
    pub papers: Vec<ApiPaper>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextPoolRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiPaper {
    pub id: i32,
    pub filename: String,
    pub title: Option<String>,
    pub authors: Option<String>,
    pub year: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateCollectionRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    pub id: String,
    pub title: String,
    pub text: String,
    pub contexts: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateShardRequest {
    pub id: String,
    pub text: String,
    pub contexts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateShardRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contexts: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Constellar graph API
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct CreateGraphResponse {
    pub graph_id: String,
    pub root_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphShardResponse {
    pub id: String,
    #[serde(default)]
    pub parent_ids: Vec<String>,
    #[serde(default)]
    pub visible: bool,
    #[serde(default)]
    pub user_content: Option<String>,
    #[serde(default)]
    pub assistant_content: Option<String>,
    #[serde(default)]
    pub contexts: Vec<String>,
    #[serde(default)]
    pub citations: Vec<serde_json::Value>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphCompileRequest {
    pub current_leaf_id: String,
    pub user_draft: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphCompileResponse {
    pub messages: Vec<serde_json::Value>,
    pub compiled_shard_ids: Vec<String>,
    pub token_count: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphShardPatchRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GetGraphResponse {
    pub shards: std::collections::HashMap<String, GraphShardResponse>,
    pub root_id: String,
    pub current_leaf_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphSendRequest {
    pub current_leaf_id: String,
    pub user_draft: String,
    #[serde(default = "graph_send_default_provider")]
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_token_limit: Option<u32>,
}

fn graph_send_default_provider() -> String {
    "local".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphSendResponse {
    pub response: String,
    pub new_leaf_id: String,
    pub token_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphListResponse {
    pub graph_ids: Vec<String>,
}

