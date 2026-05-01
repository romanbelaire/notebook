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
    #[serde(default)]
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
    #[serde(deserialize_with = "deserialize_optional_year")]
    pub year: Option<i32>,
    #[serde(default = "default_true")]
    pub exists: bool,
}

fn default_true() -> bool {
    true
}

fn deserialize_optional_year<'de, D>(deserializer: D) -> Result<Option<i32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let raw = Option::<Value>::deserialize(deserializer)?;
    match raw {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => {
            let year = number
                .as_i64()
                .ok_or_else(|| D::Error::custom("year number is not an integer"))?;
            let year = i32::try_from(year)
                .map_err(|_| D::Error::custom("year number is out of i32 range"))?;
            Ok(Some(year))
        }
        Some(Value::String(text)) => {
            let year = text
                .parse::<i32>()
                .map_err(|_| D::Error::custom(format!("invalid year string: {}", text)))?;
            Ok(Some(year))
        }
        Some(other) => Err(D::Error::custom(format!(
            "invalid year type: {}",
            other
        ))),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArxivImportResponse {
    pub task_id: String,
    pub ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateCollectionRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CollectionPaperIdsRequest {
    pub paper_ids: Vec<i32>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphMention {
    Paper { paper_id: i32 },
    Shard { graph_id: String, shard_id: String },
    Graph { graph_id: String },
    /// Local notepad document (`data/documents/{id}.json`).
    Notepad { document_id: String },
}

/// Parse `@paper:123`, `@shard:gid:sid`, `@graph:gid` tokens from draft text.
pub fn parse_graph_mentions_from_draft(text: &str) -> Vec<GraphMention> {
    use std::collections::HashSet;
    let mut out = Vec::new();
    let mut seen_paper: HashSet<i32> = HashSet::new();
    let mut seen_shard: HashSet<(String, String)> = HashSet::new();
    let mut seen_graph: HashSet<String> = HashSet::new();

    for rest in text.split("@paper:").skip(1) {
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(id) = digits.parse::<i32>() {
            if seen_paper.insert(id) {
                out.push(GraphMention::Paper { paper_id: id });
            }
        }
    }
    for rest in text.split("@shard:").skip(1) {
        let first = rest.split_whitespace().next().unwrap_or("");
        let mut parts = first.splitn(2, ':');
        let gid = parts.next().unwrap_or("");
        let sid = parts.next().unwrap_or("");
        if !gid.is_empty() && !sid.is_empty() {
            let g = gid.to_string();
            let s = sid.to_string();
            if seen_shard.insert((g.clone(), s.clone())) {
                out.push(GraphMention::Shard {
                    graph_id: g,
                    shard_id: s,
                });
            }
        }
    }
    for rest in text.split("@graph:").skip(1) {
        let gid = rest.split_whitespace().next().unwrap_or("");
        if !gid.is_empty() && seen_graph.insert(gid.to_string()) {
            out.push(GraphMention::Graph {
                graph_id: gid.to_string(),
            });
        }
    }
    let mut seen_notepad: HashSet<String> = HashSet::new();
    for rest in text.split("@notepad:").skip(1) {
        let tok = rest.split_whitespace().next().unwrap_or("");
        if !tok.is_empty() && seen_notepad.insert(tok.to_string()) {
            out.push(GraphMention::Notepad {
                document_id: tok.to_string(),
            });
        }
    }
    out
}

/// Remove mention tokens from draft; keeps user-visible text for the shard.
pub fn strip_graph_mention_tokens(text: &str) -> String {
    let mut s = text.to_string();
    for _ in 0..64 {
        let mut changed = false;
        if let Some(i) = s.find("@paper:") {
            let after_prefix = &s[i + 7..];
            let n = after_prefix.chars().take_while(|c| c.is_ascii_digit()).count();
            let end = i + 7 + n;
            s = format!("{}{}", &s[..i], s[end..].trim_start());
            changed = true;
        } else if let Some(i) = s.find("@shard:") {
            let after = &s[i + 7..];
            let tok = after.split_whitespace().next().unwrap_or("");
            let end = i + 7 + tok.len();
            s = format!("{}{}", &s[..i], s[end..].trim_start());
            changed = true;
        } else if let Some(i) = s.find("@graph:") {
            let after = &s[i + 7..];
            let tok = after.split_whitespace().next().unwrap_or("");
            let end = i + 7 + tok.len();
            s = format!("{}{}", &s[..i], s[end..].trim_start());
            changed = true;
        } else if let Some(i) = s.find("@notepad:") {
            let after = &s[i + 9..];
            let tok = after.split_whitespace().next().unwrap_or("");
            let end = i + 9 + tok.len();
            s = format!("{}{}", &s[..i], s[end..].trim_start());
            changed = true;
        }
        if !changed {
            break;
        }
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ").trim().to_string()
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<Vec<String>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mentions: Vec<GraphMention>,
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

