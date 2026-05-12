//! Local SQLite store for event-backed knowledge graph edges (compact refs, interned strings).
use crate::api::models::GraphMention;
use crate::knowledge::model::{
    assert_edge_allowed, document_segment_external_id, shard_external_id, Actor, EdgeKind,
    EventKind, GraphRef, RefKind,
};
use crate::persistence::get_data_dir;
use rusqlite::{params, Connection};
use std::sync::Mutex;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_kind INTEGER NOT NULL,
    external_id TEXT NOT NULL,
    UNIQUE(ref_kind, external_id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind INTEGER NOT NULL,
    actor INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    payload BLOB
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind INTEGER NOT NULL,
    from_ref_id INTEGER NOT NULL REFERENCES refs(id),
    to_ref_id INTEGER NOT NULL REFERENCES refs(id),
    created_by_event_id INTEGER NOT NULL REFERENCES events(id),
    meta BLOB
);

CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_ref_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_ref_id);
CREATE INDEX IF NOT EXISTS idx_edges_kind_from ON edges(kind, from_ref_id);
CREATE INDEX IF NOT EXISTS idx_edges_event ON edges(created_by_event_id);
"#;

pub struct KnowledgeGraphStore {
    conn: Mutex<Connection>,
}

impl KnowledgeGraphStore {
    pub fn open() -> Result<Self, Box<dyn std::error::Error>> {
        let dir = get_data_dir()?;
        let path = dir.join("knowledge_graph.db");
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn upsert_ref_tx(tx: &rusqlite::Transaction<'_>, g: &GraphRef) -> Result<i64, rusqlite::Error> {
        tx.execute(
            "INSERT INTO refs (ref_kind, external_id) VALUES (?1, ?2)
             ON CONFLICT(ref_kind, external_id) DO NOTHING",
            params![g.kind as u8, g.external_id],
        )?;
        let id: i64 = tx.query_row(
            "SELECT id FROM refs WHERE ref_kind = ?1 AND external_id = ?2",
            params![g.kind as u8, g.external_id],
            |r| r.get(0),
        )?;
        Ok(id)
    }

    /// Insert one event and its edges in a single transaction. Fails on invalid edge pairs.
    pub fn append_event_with_edges(
        &self,
        kind: EventKind,
        actor: Actor,
        timestamp_ms: i64,
        payload: Option<Vec<u8>>,
        edges: &[(EdgeKind, GraphRef, GraphRef, Option<Vec<u8>>)],
    ) -> Result<i64, Box<dyn std::error::Error>> {
        for (ek, a, b, _) in edges {
            assert_edge_allowed(*ek, a.kind, b.kind)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        }
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut tx = conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO events (kind, actor, timestamp_ms, payload) VALUES (?1, ?2, ?3, ?4)",
            params![
                kind as u8,
                actor as u8,
                timestamp_ms,
                payload,
            ],
        )?;
        let event_id: i64 = tx.last_insert_rowid();
        for (ek, from, to, meta) in edges {
            let from_id = Self::upsert_ref_tx(&tx, from)?;
            let to_id = Self::upsert_ref_tx(&tx, to)?;
            tx.execute(
                "INSERT INTO edges (kind, from_ref_id, to_ref_id, created_by_event_id, meta)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![*ek as u8, from_id, to_id, event_id, meta.as_deref()],
            )?;
        }
        tx.commit()?;
        Ok(event_id)
    }

    /// Many shards may link to one knowledge node; one shard may link to many nodes (multiple events).
    pub fn attach_shards_to_knowledge_node(
        &self,
        knowledge_node_id: &str,
        graph_id: &str,
        shard_ids: &[String],
    ) -> Result<i64, Box<dyn std::error::Error>> {
        let ts = now_ms();
        let kn = GraphRef::new(RefKind::KnowledgeNode, knowledge_node_id);
        let mut edges: Vec<(EdgeKind, GraphRef, GraphRef, Option<Vec<u8>>)> = Vec::new();
        for sid in shard_ids {
            let shard_ref = GraphRef::new(
                RefKind::Shard,
                shard_external_id(graph_id, sid),
            );
            edges.push((
                EdgeKind::GeneratedFrom,
                kn.clone(),
                shard_ref,
                None,
            ));
        }
        if edges.is_empty() {
            return Err("attach_shards_to_knowledge_node: no shards".into());
        }
        self.append_event_with_edges(
            EventKind::AttachShardsToKnowledgeNode,
            Actor::User,
            ts,
            None,
            &edges,
        )
    }

    /// `contains` document → segment and segment → knowledge node (optional second hop).
    pub fn record_contains_knowledge_under_block(
        &self,
        document_id: &str,
        block_id: &str,
        knowledge_node_id: &str,
    ) -> Result<i64, Box<dyn std::error::Error>> {
        let seg_key = document_segment_external_id(document_id, block_id);
        let doc_ref = GraphRef::new(RefKind::Document, document_id);
        let seg_ref = GraphRef::new(RefKind::DocumentSegment, seg_key);
        let kn_ref = GraphRef::new(RefKind::KnowledgeNode, knowledge_node_id);
        let edges = vec![
            (
                EdgeKind::Contains,
                doc_ref,
                seg_ref.clone(),
                None,
            ),
            (
                EdgeKind::Contains,
                seg_ref,
                kn_ref,
                None,
            ),
        ];
        let ts = now_ms();
        self.append_event_with_edges(
            EventKind::ContainsDocumentStructure,
            Actor::User,
            ts,
            None,
            &edges,
        )
    }

    /// Record compile/send context: one usage_event ref + `used_in` from each mention target.
    pub fn record_compile_context_attach(
        &self,
        graph_id: &str,
        mentions: &[GraphMention],
    ) -> Result<Option<i64>, Box<dyn std::error::Error>> {
        if mentions.is_empty() {
            return Ok(None);
        }
        let ts = now_ms();
        let usage_external = format!("compile_{}_{}", ts, uuid::Uuid::new_v4());
        let usage_ref = GraphRef::new(RefKind::UsageEvent, usage_external.clone());
        let mut edges: Vec<(EdgeKind, GraphRef, GraphRef, Option<Vec<u8>>)> = Vec::new();
        for m in mentions {
            let from = match m {
                GraphMention::Paper { paper_id } => {
                    GraphRef::new(RefKind::Paper, paper_id.to_string())
                }
                GraphMention::Shard {
                    graph_id: gid,
                    shard_id,
                } => GraphRef::new(RefKind::Shard, shard_external_id(gid, shard_id)),
                GraphMention::Graph { graph_id: gid } => GraphRef::new(RefKind::Graph, gid.clone()),
                GraphMention::Notepad { document_id } => {
                    GraphRef::new(RefKind::Document, document_id.clone())
                }
            };
            edges.push((EdgeKind::UsedIn, from, usage_ref.clone(), None));
        }
        let payload = serde_json::to_vec(&serde_json::json!({ "graph_id": graph_id }))
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let event_id = self.append_event_with_edges(
            EventKind::CompileContextAttach,
            Actor::User,
            ts,
            Some(payload),
            &edges,
        )?;
        Ok(Some(event_id))
    }

    pub fn record_slate_push_event(&self, slate_item_id: &str, preview_len: u32) -> Result<i64, Box<dyn std::error::Error>> {
        let ts = now_ms();
        let payload = serde_json::to_vec(&serde_json::json!({
            "slate_item_id": slate_item_id,
            "preview_len": preview_len,
        }))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let item_ref = GraphRef::new(RefKind::SlateItem, slate_item_id);
        let mut conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut tx = conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO events (kind, actor, timestamp_ms, payload) VALUES (?1, ?2, ?3, ?4)",
            params![
                EventKind::SlatePush as u8,
                Actor::User as u8,
                ts,
                payload,
            ],
        )?;
        let event_id = tx.last_insert_rowid();
        Self::upsert_ref_tx(&tx, &item_ref)?;
        tx.commit()?;
        Ok(event_id)
    }

    pub fn record_content_transfer(
        &self,
        from: GraphRef,
        to: GraphRef,
        meta: Option<Vec<u8>>,
    ) -> Result<i64, Box<dyn std::error::Error>> {
        let ts = now_ms();
        self.append_event_with_edges(
            EventKind::ContentTransfer,
            Actor::User,
            ts,
            None,
            &[(EdgeKind::ContentTransfer, from, to, meta)],
        )
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
