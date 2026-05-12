//! Knowledge graph types and edge validation (event-backed, explicit links only).
pub mod model;

pub use model::{
    assert_edge_allowed, document_segment_external_id, shard_external_id, Actor, EdgeKind,
    EventKind, GraphRef, RefKind,
};
