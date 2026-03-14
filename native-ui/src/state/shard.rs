use serde::{Serialize, Deserialize};
use crate::ui::chat_window::{MessageRole, Citation};

/// Core data structure representing a blob of context
/// This is the first-class citizen in the data ecosystem
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Shard {
    /// Unique identifier for this shard
    pub id: String,
    
    /// The text content of this shard
    pub text: String,
    
    /// Vector embedding (384 dimensions for all-MiniLM-L6-v2)
    /// None if embedding hasn't been computed yet
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    
    /// Sources referenced by this shard (PDFs, links, friend shards)
    pub sources: Vec<ShardSource>,
    
    /// Parent shard ID (the message that preceded this one)
    /// None for root shards
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    
    /// Direct children shard IDs (messages spawned from this one)
    pub children_ids: Vec<String>,
    
    /// Friend shard IDs (other shards linked as additional context, not parent-child related)
    /// Supports both "friends_ids" and legacy "sisters_ids" field names
    #[serde(alias = "sisters_ids", default)]
    pub friends_ids: Vec<String>,
    
    /// Timestamp when this shard was created
    pub created_at: u64,
    
    /// Additional metadata for rendering and API communication
    pub metadata: ShardMetadata,
}

impl<'de> Deserialize<'de> for Shard {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;
        
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Id,
            Text,
            Embedding,
            Sources,
            ParentId,
            ChildrenIds,
            FriendsIds,
            #[serde(rename = "sisters_ids")]
            SistersIds,
            CreatedAt,
            Metadata,
        }
        
        struct ShardVisitor;
        
        impl<'de> Visitor<'de> for ShardVisitor {
            type Value = Shard;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Shard")
            }
            
            fn visit_map<V>(self, mut map: V) -> Result<Shard, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut id = None;
                let mut text = None;
                let mut embedding = None;
                let mut sources = None;
                let mut parent_id = None;
                let mut children_ids = None;
                let mut friends_ids = None;
                let mut created_at = None;
                let mut metadata = None;
                
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Id => {
                            if id.is_some() {
                                return Err(de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value()?);
                        }
                        Field::Text => {
                            if text.is_some() {
                                return Err(de::Error::duplicate_field("text"));
                            }
                            text = Some(map.next_value()?);
                        }
                        Field::Embedding => {
                            if embedding.is_some() {
                                return Err(de::Error::duplicate_field("embedding"));
                            }
                            embedding = Some(map.next_value()?);
                        }
                        Field::Sources => {
                            if sources.is_some() {
                                return Err(de::Error::duplicate_field("sources"));
                            }
                            sources = Some(map.next_value()?);
                        }
                        Field::ParentId => {
                            if parent_id.is_some() {
                                return Err(de::Error::duplicate_field("parent_id"));
                            }
                            parent_id = Some(map.next_value()?);
                        }
                        Field::ChildrenIds => {
                            if children_ids.is_some() {
                                return Err(de::Error::duplicate_field("children_ids"));
                            }
                            children_ids = Some(map.next_value()?);
                        }
                        Field::FriendsIds => {
                            if friends_ids.is_some() {
                                return Err(de::Error::duplicate_field("friends_ids"));
                            }
                            friends_ids = Some(map.next_value()?);
                        }
                        Field::SistersIds => {
                            // Handle legacy sisters_ids field - use it if friends_ids not already set
                            if friends_ids.is_none() {
                                friends_ids = Some(map.next_value()?);
                            } else {
                                let _: Vec<String> = map.next_value()?; // Skip duplicate
                            }
                        }
                        Field::CreatedAt => {
                            if created_at.is_some() {
                                return Err(de::Error::duplicate_field("created_at"));
                            }
                            created_at = Some(map.next_value()?);
                        }
                        Field::Metadata => {
                            if metadata.is_some() {
                                return Err(de::Error::duplicate_field("metadata"));
                            }
                            metadata = Some(map.next_value()?);
                        }
                    }
                }
                
                let id = id.ok_or_else(|| de::Error::missing_field("id"))?;
                let text = text.ok_or_else(|| de::Error::missing_field("text"))?;
                let sources = sources.unwrap_or_default();
                let parent_id = parent_id;
                let children_ids = children_ids.unwrap_or_default();
                let friends_ids = friends_ids.unwrap_or_default();
                let created_at = created_at.ok_or_else(|| de::Error::missing_field("created_at"))?;
                let metadata = metadata.unwrap_or_default();
                
                Ok(Shard {
                    id,
                    text,
                    embedding,
                    sources,
                    parent_id,
                    children_ids,
                    friends_ids,
                    created_at,
                    metadata,
                })
            }
        }
        
        deserializer.deserialize_map(ShardVisitor)
    }
}

/// Metadata associated with a shard
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardMetadata {
    /// Role of the message (User or Assistant)
    pub role: MessageRole,
    
    /// Context strings from retrieval
    pub contexts: Vec<String>,
    
    /// Citations from sources
    pub citations: Vec<Citation>,
    
    /// User-attached notes (comments) on this message; included in prompt when shard is used for generation
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Source types that can be referenced by a shard
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShardSource {
    /// Reference to a PDF paper
    Pdf {
        paper_id: i32,
        page: Option<u32>,
    },
    /// Reference to an external link
    Link {
        url: String,
    },
    /// Reference to another shard (friend shard)
    /// Supports both "friend_shard" and legacy "sister_shard" variant names
    #[serde(rename = "friend_shard", alias = "sister_shard")]
    FriendShard {
        shard_id: String,
    },
}

impl Shard {
    /// Create a new shard with the given text and role
    pub fn new(text: String, role: MessageRole) -> Self {
        let id = format!("shard_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos());
        
        Self {
            id,
            text: text.clone(),
            embedding: None,
            sources: Vec::new(),
            parent_id: None,
            children_ids: Vec::new(),
            friends_ids: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: ShardMetadata {
                role,
                contexts: Vec::new(),
                citations: Vec::new(),
                notes: Vec::new(),
            },
        }
    }
    
    /// Set the embedding for this shard
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }
    
    /// Add a source to this shard
    pub fn add_source(&mut self, source: ShardSource) {
        self.sources.push(source);
    }
    
    /// Set the parent shard
    pub fn set_parent(&mut self, parent_id: String) {
        self.parent_id = Some(parent_id);
    }
    
    /// Add a child shard
    pub fn add_child(&mut self, child_id: String) {
        if !self.children_ids.contains(&child_id) {
            self.children_ids.push(child_id);
        }
    }
    
    /// Add a friend shard (linked context, not parent-child related)
    pub fn add_friend(&mut self, friend_id: String) {
        if !self.friends_ids.contains(&friend_id) {
            self.friends_ids.push(friend_id);
        }
    }
    
    /// Check if this shard has an embedding
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }
    
    /// Get the embedding dimension (384 for all-MiniLM-L6-v2)
    pub fn embedding_dimension() -> usize {
        384
    }
}

impl Default for ShardMetadata {
    fn default() -> Self {
        Self {
            role: MessageRole::User,
            contexts: Vec::new(),
            citations: Vec::new(),
            notes: Vec::new(),
        }
    }
}

