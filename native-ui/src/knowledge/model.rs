//! Compact enums for SQLite rows and validation of allowed endpoint pairs.

/// Stored in `refs.ref_kind` / edge payloads.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefKind {
    Document = 1,
    /// Block-level anchor: `external_id` = `document_segment_external_id(doc, block)`.
    DocumentSegment = 2,
    KnowledgeNode = 3,
    /// Constellation shard; `external_id` = `shard_external_id(graph_id, shard_id)`.
    Shard = 4,
    SourceArtifact = 5,
    UsageEvent = 6,
    SlateItem = 7,
    Graph = 8,
    Paper = 9,
}

impl RefKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(RefKind::Document),
            2 => Some(RefKind::DocumentSegment),
            3 => Some(RefKind::KnowledgeNode),
            4 => Some(RefKind::Shard),
            5 => Some(RefKind::SourceArtifact),
            6 => Some(RefKind::UsageEvent),
            7 => Some(RefKind::SlateItem),
            8 => Some(RefKind::Graph),
            9 => Some(RefKind::Paper),
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Contains = 1,
    Mentions = 2,
    /// `from` = knowledge_node, `to` = shard (provenance: node derived from shard).
    GeneratedFrom = 3,
    /// `from` = knowledge_node | document | shard | paper | graph, `to` = usage_event.
    UsedIn = 4,
    ReferencesSource = 5,
    ContentTransfer = 6,
}

impl EdgeKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(EdgeKind::Contains),
            2 => Some(EdgeKind::Mentions),
            3 => Some(EdgeKind::GeneratedFrom),
            4 => Some(EdgeKind::UsedIn),
            5 => Some(EdgeKind::ReferencesSource),
            6 => Some(EdgeKind::ContentTransfer),
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Generic = 0,
    MentionInsert = 1,
    ClipboardPaste = 2,
    ClipboardCopy = 3,
    ContentTransfer = 4,
    SlatePush = 5,
    SlatePaste = 6,
    SlateReorder = 7,
    SlateRemove = 8,
    CompileContextAttach = 9,
    AttachShardsToKnowledgeNode = 10,
    ContainsDocumentStructure = 11,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Actor {
    User = 1,
    SystemExplicitPolicy = 2,
}

impl Actor {
    pub fn from_u8(v: u8) -> Self {
        match v {
            2 => Actor::SystemExplicitPolicy,
            _ => Actor::User,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRef {
    pub kind: RefKind,
    pub external_id: String,
}

impl GraphRef {
    pub fn new(kind: RefKind, external_id: impl Into<String>) -> Self {
        Self {
            kind,
            external_id: external_id.into(),
        }
    }
}

pub fn shard_external_id(graph_id: &str, shard_id: &str) -> String {
    format!("{}::{}", graph_id, shard_id)
}

pub fn document_segment_external_id(document_id: &str, block_id: &str) -> String {
    format!("{}::{}", document_id, block_id)
}

/// Fail-fast validation: returns `Err` if this edge kind is not allowed between these ref kinds.
pub fn assert_edge_allowed(kind: EdgeKind, from: RefKind, to: RefKind) -> Result<(), String> {
    let ok = match kind {
        EdgeKind::Contains => matches!(
            (from, to),
            (RefKind::Document, RefKind::DocumentSegment)
                | (RefKind::DocumentSegment, RefKind::KnowledgeNode)
                | (RefKind::Document, RefKind::KnowledgeNode)
        ),
        EdgeKind::Mentions => {
            matches!(
                from,
                RefKind::KnowledgeNode | RefKind::Document | RefKind::DocumentSegment
            ) && matches!(
                to,
                RefKind::KnowledgeNode
                    | RefKind::Document
                    | RefKind::Shard
                    | RefKind::Paper
                    | RefKind::Graph
                    | RefKind::SourceArtifact
            )
        }
        EdgeKind::GeneratedFrom => {
            matches!((from, to), (RefKind::KnowledgeNode, RefKind::Shard))
        }
        EdgeKind::UsedIn => {
            matches!(
                from,
                RefKind::KnowledgeNode
                    | RefKind::Document
                    | RefKind::Shard
                    | RefKind::Paper
                    | RefKind::Graph
            ) && matches!(to, RefKind::UsageEvent)
        }
        EdgeKind::ReferencesSource => {
            matches!((from, to), (RefKind::KnowledgeNode, RefKind::SourceArtifact))
        }
        EdgeKind::ContentTransfer => {
            // Any typed ref to any typed ref — provenance is in the event, not semantics.
            true
        }
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "edge {:?} not allowed for {:?} -> {:?}",
            kind, from, to
        ))
    }
}
