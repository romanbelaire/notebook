//! Graph state for Constellar: one graph per "current conversation", keyed by graph_id.
//! Mirrors backend ConversationGraph + ActiveState; nodes have world position/velocity for constellation UI.

use crate::api::models::GraphShardResponse;
use crate::gfx::text_layout::ParagraphWrappedFlow;
use crate::persistence::GraphLayoutPersistence;
use crate::ui::style;
use glam::Vec2;
use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use std::collections::{HashMap, HashSet, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Map a node id to a stable position in [0, 1) on the ribbon color wheel.
/// Uses DefaultHasher for determinism within a single process run.
fn hue_from_id(id: &str) -> f32 {
    let mut h = DefaultHasher::new();
    id.hash(&mut h);
    let raw = h.finish();
    (raw & 0xFFFF) as f32 / 0xFFFF as f32
}

/// One turn shard (message pair) or special shard. Mirrors API GraphShardResponse.
/// Per-message visibility: when compiling context, include user/assistant only if the corresponding flag is true.
#[derive(Clone, Debug)]
pub struct GraphShard {
    pub id: String,
    pub parent_ids: Vec<String>,
    /// Shard-level visibility (API); when patching we send user_visible && assistant_visible.
    pub visible: bool,
    /// Include user message in context when compiling.
    pub user_visible: bool,
    /// Include assistant message in context when compiling.
    pub assistant_visible: bool,
    pub user_content: Option<String>,
    pub assistant_content: Option<String>,
    pub contexts: Vec<String>,
    pub citations: Vec<serde_json::Value>,
    pub notes: Vec<String>,
    pub content: Option<String>,
    pub role: Option<String>,
}

impl GraphShard {
    pub fn is_turn(&self) -> bool {
        self.user_content.is_some()
    }

    /// True if this shard has no user or assistant content (empty root placeholder).
    pub fn is_empty_content(&self) -> bool {
        let u = self.user_content.as_deref().unwrap_or("");
        let a = self.assistant_content.as_deref().unwrap_or("");
        u.trim().is_empty() && a.trim().is_empty()
    }

    /// User + assistant markdown/plain text for system clipboard.
    pub fn clipboard_plain_text(&self) -> String {
        let mut out = String::new();
        if let Some(ref u) = self.user_content {
            if !u.is_empty() {
                out.push_str("User:\n");
                out.push_str(u);
            }
        }
        if let Some(ref a) = self.assistant_content {
            if !a.is_empty() {
                if !out.is_empty() {
                    out.push_str("\n\n");
                }
                out.push_str("Assistant:\n");
                out.push_str(a);
            }
        }
        out
    }
}

impl From<&GraphShardResponse> for GraphShard {
    fn from(r: &GraphShardResponse) -> Self {
        Self {
            id: r.id.clone(),
            parent_ids: r.parent_ids.clone(),
            visible: r.visible,
            user_visible: r.visible,
            assistant_visible: r.visible,
            user_content: r.user_content.clone(),
            assistant_content: r.assistant_content.clone(),
            contexts: r.contexts.clone(),
            citations: r.citations.clone(),
            notes: r.notes.clone(),
            content: r.content.clone(),
            role: r.role.clone(),
        }
    }
}

/// Node in the constellation: shard data + world position for deterministic tree layout.
#[derive(Clone, Debug)]
pub struct ConstellationNode {
    pub shard: GraphShard,
    pub position: Vec2,
    /// Cached size (width, height) from last layout; used for hit-test and rendering.
    pub size: Vec2,
    /// Unclamped content height computed from text + chrome. If this exceeds `size.y`,
    /// the shard has vertical overflow and its inner content can be scrolled.
    pub content_height: f32,
    /// Measured user bubble text height (world units) for per-bubble scroll overflow.
    pub user_text_height: f32,
    /// Measured assistant bubble text height (world units) for per-bubble scroll overflow.
    pub assistant_text_height: f32,
    /// Measured user bubble text width (world units); used by render to avoid re-measure.
    pub user_text_width: f32,
    /// Measured assistant bubble text width (world units); used by render to avoid re-measure.
    pub assistant_text_width: f32,
    /// Cached text-layer key for user bubble: (content_hash, scale_bucket).
    pub text_layer_key_user: Option<(u64, u32)>,
    /// Cached text-layer key for assistant bubble: (content_hash, scale_bucket).
    pub text_layer_key_assistant: Option<(u64, u32)>,
    /// Logical user text area size in world units for text layer composition.
    pub text_layer_size_user: Vec2,
    /// Logical assistant text area size in world units for text layer composition.
    pub text_layer_size_assistant: Vec2,
    /// Depth in the graph hierarchy (root = 0, children = parent_depth + 1).
    pub depth: u32,
    /// Number of direct children (nodes for which this node is a parent).
    pub child_count: u32,
    /// Position on the ribbon color wheel [0, 1).
    /// Roots get a hash-derived hue; first children inherit the parent's hue;
    /// subsequent children get a fresh hash-derived hue.
    pub ribbon_hue_t: f32,
}

/// Active graph state: current conversation = one graph.
#[derive(Clone, Debug, Default)]
pub struct GraphState {
    pub graph_id: Option<String>,
    pub root_id: Option<String>,
    pub current_leaf_id: Option<String>,
    /// Monotonic version bumped whenever graph content changes (nodes added/removed or content/notes updated).
    /// Used to invalidate constellation layout and markdown caches.
    pub content_version: u64,
    /// id -> node (turn shards only for v1; special shards can be skipped or rendered as single block).
    pub nodes: HashMap<String, ConstellationNode>,
    /// User-resized node sizes (from layout persistence or drag-resize). When set, update_node_sizes uses these instead of content-derived size.
    pub manual_sizes: HashMap<String, Vec2>,
    /// Node ids that have been manually dragged; tree layout preserves their position.
    pub manual_positions: HashSet<String>,
    /// True when node sizes have changed and a layout pass is needed before the next render.
    pub layout_dirty: bool,
    /// Most recently visited child for each parent, used for intelligent down-arrow navigation.
    pub last_visited_child: HashMap<String, String>,
}

/// Fixed shard width in world units. All cards default to this width so columns are uniform.
const STANDARD_SHARD_WIDTH: f32 = 360.0;
/// Placeholder width used before `update_node_sizes` runs.
const PLACEHOLDER_SHARD_WIDTH: f32 = STANDARD_SHARD_WIDTH;
/// Placeholder height for nodes before layout measures content.
const PLACEHOLDER_NODE_HEIGHT: f32 = 120.0;
const MIN_SHARD_MANUAL_WIDTH: f32 = 200.0;
const MIN_SHARD_MANUAL_HEIGHT: f32 = 60.0;
/// Horizontal gap between sibling subtrees in world units.
const LAYOUT_SIBLING_GAP: f32 = 80.0;
/// Vertical gap between a parent row and its children row in world units.
const LAYOUT_ROW_GAP: f32 = 80.0;

impl GraphState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Min/max width for constellation shard drag-resize (world units).
    pub fn shard_manual_width_bounds(_viewport_size_x: f32) -> (f32, f32) {
        (MIN_SHARD_MANUAL_WIDTH, STANDARD_SHARD_WIDTH * 2.0)
    }

    /// Minimum manual height (world units) for constellation shard drag-resize.
    /// Maximum is per-shard (the shard's own `content_height`) and lives in the resize handler.
    pub fn shard_manual_min_height() -> f32 {
        MIN_SHARD_MANUAL_HEIGHT
    }

    pub fn is_empty(&self) -> bool {
        self.graph_id.is_none() && self.nodes.is_empty()
    }

    /// Constellation rendering (shards, full-bleed viewport) — requires a loaded graph with drawable nodes.
    #[inline]
    pub fn constellation_view_active(&self) -> bool {
        self.graph_id.is_some() && !self.nodes.is_empty()
    }

    pub fn set_graph(
        &mut self,
        graph_id: String,
        root_id: String,
        current_leaf_id: String,
        shards: HashMap<String, GraphShard>,
    ) {
        self.content_version = self.content_version.wrapping_add(1);
        self.graph_id = Some(graph_id.clone());
        self.root_id = Some(root_id.clone());
        self.current_leaf_id = Some(current_leaf_id);
        let (stored_manual_positions, stored_sizes) = GraphLayoutPersistence::load_positions(&graph_id);
        self.manual_sizes = stored_sizes;
        self.manual_positions.clear();
        // Restore manually-dragged positions; these override the tree layout.
        let manual_positions_data = stored_manual_positions;
        // Do not add the empty root to nodes (backend creates root with empty strings); first real message attaches to root.
        self.nodes = shards
            .into_iter()
            .filter(|(id, s)| s.is_turn() && !(id == &root_id && s.is_empty_content()))
            .map(|(id, shard)| {
                let size = Vec2::new(PLACEHOLDER_SHARD_WIDTH, PLACEHOLDER_NODE_HEIGHT); // Placeholder; layout will measure
                (
                    id,
                    ConstellationNode {
                        shard,
                        position: Vec2::ZERO,
                        size,
                        content_height: 0.0,
                        user_text_height: 0.0,
                        assistant_text_height: 0.0,
                        user_text_width: 0.0,
                        assistant_text_width: 0.0,
                        text_layer_key_user: None,
                        text_layer_key_assistant: None,
                        text_layer_size_user: Vec2::ZERO,
                        text_layer_size_assistant: Vec2::ZERO,
                        depth: 0,
                        child_count: 0,
                        ribbon_hue_t: 0.0,
                    },
                )
            })
            .collect();
        self.promote_empty_root_to_first_child();
        // Apply stored manual positions after node map is built.
        for (id, pos) in &manual_positions_data {
            if let Some(node) = self.nodes.get_mut(id) {
                node.position = *pos;
                self.manual_positions.insert(id.clone());
            }
        }
        self.layout_dirty = true;
        self.recompute_hierarchy_metadata();
        self.compute_ribbon_hues();
    }

    /// If the root node is empty and has exactly one child (the first chat shard), make that child the root
    /// and remove the empty node so we don't show a separate empty root in the constellation.
    fn promote_empty_root_to_first_child(&mut self) {
        let root_id_val = match &self.root_id {
            Some(r) => r.clone(),
            None => return,
        };
        let root_is_empty = self
            .nodes
            .get(&root_id_val)
            .map(|n| n.shard.is_empty_content())
            .unwrap_or(false);
        if !root_is_empty {
            return;
        }
        let children = self.children_ids(&root_id_val);
        if children.len() == 1 {
            let new_root = children[0].clone();
            self.nodes.remove(&root_id_val);
            self.root_id = Some(new_root);
        }
    }

    /// Deterministic tree layout: positions all nodes using a two-pass centering algorithm.
    ///
    /// Each node's primary parent is `parent_ids[0]` (if present in nodes); secondary parents
    /// draw cross-edges but do not affect placement. Siblings are horizontally centered under their parent.
    ///
    /// Nodes in `manual_positions` keep their current position; their children are still placed
    /// relative to the node's actual position.
    fn apply_tree_layout(&mut self) {
        if self.nodes.is_empty() {
            self.layout_dirty = false;
            return;
        }

        let root_id = match &self.root_id {
            Some(r) => r.clone(),
            None => {
                self.layout_dirty = false;
                return;
            }
        };

        // Build primary_children map: primary parent -> sorted list of children.
        // A node's primary parent is parent_ids[0] if that node exists in self.nodes.
        // Nodes without a primary parent in nodes become layout roots.
        let mut primary_children: HashMap<String, Vec<String>> = HashMap::new();
        let mut layout_roots: Vec<String> = Vec::new();

        for (id, node) in &self.nodes {
            match node.shard.parent_ids.first() {
                Some(pid) if self.nodes.contains_key(pid) => {
                    primary_children.entry(pid.clone()).or_default().push(id.clone());
                }
                _ => layout_roots.push(id.clone()),
            }
        }
        layout_roots.sort();
        for children in primary_children.values_mut() {
            children.sort();
        }

        // Snapshot node sizes for read-only use during computation.
        let heights: HashMap<String, f32> = self.nodes.iter()
            .map(|(id, n)| (id.clone(), n.size.y))
            .collect();
        let widths: HashMap<String, f32> = self.nodes.iter()
            .map(|(id, n)| (id.clone(), n.size.x))
            .collect();

        // BFS traversal order (used to get post-order by reversing).
        let mut bfs_order: Vec<String> = Vec::new();
        {
            let mut queue: VecDeque<String> = layout_roots.iter().cloned().collect();
            let mut visited: HashSet<String> = layout_roots.iter().cloned().collect();
            while let Some(id) = queue.pop_front() {
                bfs_order.push(id.clone());
                if let Some(children) = primary_children.get(&id) {
                    for cid in children {
                        if visited.insert(cid.clone()) {
                            queue.push_back(cid.clone());
                        }
                    }
                }
            }
        }

        // Pass 1 (post-order): compute subtree_width for each node.
        // A subtree's width is the horizontal space it requires including all sibling gaps.
        let mut subtree_widths: HashMap<String, f32> = HashMap::new();
        for id in bfs_order.iter().rev() {
            let node_w = widths.get(id).copied().unwrap_or(STANDARD_SHARD_WIDTH);
            let children = primary_children.get(id).map(|v| v.as_slice()).unwrap_or(&[]);
            if children.is_empty() {
                subtree_widths.insert(id.clone(), node_w);
            } else {
                let sum: f32 = children.iter()
                    .map(|cid| subtree_widths.get(cid).copied().unwrap_or(STANDARD_SHARD_WIDTH))
                    .sum();
                let gaps = (children.len() - 1) as f32 * LAYOUT_SIBLING_GAP;
                subtree_widths.insert(id.clone(), (sum + gaps).max(node_w));
            }
        }

        // Pass 2 (pre-order): assign positions top-down.
        // Roots are centered around x=0; each row is offset vertically by the parent height + row gap.

        let total_roots_w: f32 = {
            let sum: f32 = layout_roots.iter()
                .map(|id| subtree_widths.get(id).copied().unwrap_or(0.0))
                .sum();
            let gaps = (layout_roots.len().saturating_sub(1)) as f32 * LAYOUT_SIBLING_GAP;
            sum + gaps
        };

        // Queue entries: (node_id, computed_center_x, computed_y)
        let mut assign_queue: VecDeque<(String, f32, f32)> = VecDeque::new();
        let mut cursor_x = -total_roots_w / 2.0;
        for id in &layout_roots {
            let subtree_w = subtree_widths.get(id).copied().unwrap_or(0.0);
            let center_x = cursor_x + subtree_w / 2.0;
            assign_queue.push_back((id.clone(), center_x, 0.0));
            cursor_x += subtree_w + LAYOUT_SIBLING_GAP;
        }

        let mut new_positions: HashMap<String, Vec2> = HashMap::new();
        while let Some((id, center_x, y)) = assign_queue.pop_front() {
            let node_h = heights.get(&id).copied().unwrap_or(PLACEHOLDER_NODE_HEIGHT);
            let node_w = widths.get(&id).copied().unwrap_or(STANDARD_SHARD_WIDTH);
            // For manually positioned nodes, use their actual position as the anchor
            // so children still flow from the correct location.
            let (actual_center_x, actual_y) = if self.manual_positions.contains(&id) {
                let pos = self.nodes[&id].position;
                (pos.x + node_w / 2.0, pos.y)
            } else {
                new_positions.insert(id.clone(), Vec2::new(center_x - node_w / 2.0, y));
                (center_x, y)
            };

            let children = primary_children.get(&id).cloned().unwrap_or_default();
            if children.is_empty() {
                continue;
            }
            let child_y = actual_y + node_h + LAYOUT_ROW_GAP;
            let total_children_w: f32 = {
                let sum: f32 = children.iter()
                    .map(|cid| subtree_widths.get(cid).copied().unwrap_or(STANDARD_SHARD_WIDTH))
                    .sum();
                let gaps = (children.len() - 1) as f32 * LAYOUT_SIBLING_GAP;
                sum + gaps
            };
            let mut child_cursor_x = actual_center_x - total_children_w / 2.0;
            for cid in &children {
                let child_subtree_w = subtree_widths.get(cid).copied().unwrap_or(STANDARD_SHARD_WIDTH);
                let child_center_x = child_cursor_x + child_subtree_w / 2.0;
                assign_queue.push_back((cid.clone(), child_center_x, child_y));
                child_cursor_x += child_subtree_w + LAYOUT_SIBLING_GAP;
            }
        }

        // Apply computed positions to non-manual nodes.
        for (id, pos) in new_positions {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.position = pos;
            }
        }

        self.layout_dirty = false;
    }

    pub fn add_node(&mut self, shard: GraphShard, _position: Vec2) {
        self.content_version = self.content_version.wrapping_add(1);
        let id = shard.id.clone();
        let size = Vec2::new(PLACEHOLDER_SHARD_WIDTH, PLACEHOLDER_NODE_HEIGHT);
        self.nodes.insert(
            id.clone(),
            ConstellationNode {
                shard,
                position: Vec2::ZERO,
                size,
                content_height: 0.0,
                user_text_height: 0.0,
                assistant_text_height: 0.0,
                user_text_width: 0.0,
                assistant_text_width: 0.0,
                text_layer_key_user: None,
                text_layer_key_assistant: None,
                text_layer_size_user: Vec2::ZERO,
                text_layer_size_assistant: Vec2::ZERO,
                depth: 0,
                child_count: 0,
                ribbon_hue_t: 0.0,
            },
        );
        // When the first shard is created via chat (child of empty root), promote it to root.
        self.promote_empty_root_to_first_child();
        self.layout_dirty = true;
        self.recompute_hierarchy_metadata();
        self.compute_ribbon_hues();
    }

    /// Recompute depth (BFS from root) and child_count for all nodes.
    fn recompute_hierarchy_metadata(&mut self) {

        let root_id = match &self.root_id {
            Some(r) => r.clone(),
            None => return,
        };

        let mut depth_map: HashMap<String, u32> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        if self.nodes.contains_key(&root_id) {
            depth_map.insert(root_id.clone(), 0);
            queue.push_back(root_id.clone());
        } else {
            for cid in self.children_ids(&root_id) {
                depth_map.insert(cid.clone(), 0);
                queue.push_back(cid);
            }
        }

        while let Some(id) = queue.pop_front() {
            let d = *depth_map.get(&id).unwrap();
            for (cid, node) in &self.nodes {
                if node.shard.parent_ids.contains(&id) {
                    let next_depth = d + 1;
                    let entry = depth_map.entry(cid.clone()).or_insert(next_depth);
                    if next_depth < *entry {
                        *entry = next_depth;
                    }
                    queue.push_back(cid.clone());
                }
            }
        }

        let mut child_counts: HashMap<String, u32> = HashMap::new();
        for (id, node) in &self.nodes {
            for pid in &node.shard.parent_ids {
                let entry = child_counts.entry(pid.clone()).or_insert(0);
                *entry += 1;
            }
            // Ensure every node appears at least with 0 children.
            child_counts.entry(id.clone()).or_insert(0);
        }

        for (id, node) in self.nodes.iter_mut() {
            node.depth = *depth_map.get(id).unwrap_or(&0);
            node.child_count = *child_counts.get(id).unwrap_or(&0);
        }
    }

    /// Assign `ribbon_hue_t` to every node:
    /// - Layout roots (no primary parent in graph) get `hue_from_id(node_id)`.
    /// - The first child (sorted index 0) of any node inherits the parent's hue.
    /// - Every subsequent child gets its own `hue_from_id(child_id)`.
    ///
    /// Processed BFS top-down so parent hues are always available when children are visited.
    fn compute_ribbon_hues(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        // Build primary_children: parent -> sorted child ids (same logic as apply_tree_layout).
        let mut primary_children: HashMap<String, Vec<String>> = HashMap::new();
        let mut layout_roots: Vec<String> = Vec::new();
        for (id, node) in &self.nodes {
            match node.shard.parent_ids.first() {
                Some(pid) if self.nodes.contains_key(pid) => {
                    primary_children.entry(pid.clone()).or_default().push(id.clone());
                }
                _ => layout_roots.push(id.clone()),
            }
        }
        layout_roots.sort();
        for v in primary_children.values_mut() {
            v.sort();
        }

        // BFS: assign hues top-down.
        let mut hue_map: HashMap<String, f32> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        for id in &layout_roots {
            hue_map.insert(id.clone(), hue_from_id(id));
            queue.push_back(id.clone());
        }

        while let Some(id) = queue.pop_front() {
            let parent_hue = *hue_map.get(&id).unwrap_or(&0.0);
            if let Some(children) = primary_children.get(&id) {
                for (idx, cid) in children.iter().enumerate() {
                    let hue = if idx == 0 {
                        parent_hue
                    } else {
                        hue_from_id(cid)
                    };
                    hue_map.insert(cid.clone(), hue);
                    queue.push_back(cid.clone());
                }
            }
        }

        for (id, node) in self.nodes.iter_mut() {
            node.ribbon_hue_t = *hue_map.get(id).unwrap_or(&0.0);
        }
    }

    pub fn get_node(&self, id: &str) -> Option<&ConstellationNode> {
        self.nodes.get(id)
    }

    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut ConstellationNode> {
        self.nodes.get_mut(id)
    }

    /// Node ids in BFS order from root; used to build linear message list from graph.
    /// When root has no shard (not in nodes), order starts from children of root_id.
    pub fn node_ids_bfs_order(&self) -> Vec<String> {
        let root_id = match &self.root_id {
            Some(r) => r.clone(),
            None => return Vec::new(),
        };
        let mut queue = VecDeque::new();
        if self.nodes.contains_key(&root_id) {
            queue.push_back(root_id);
        } else {
            for cid in self.children_ids(&root_id) {
                queue.push_back(cid);
            }
        }
        let mut out = Vec::new();
        let mut visited = HashSet::new();
        while let Some(id) = queue.pop_front() {
            if !visited.insert(id.clone()) {
                continue;
            }
            out.push(id.clone());
            for (cid, node) in &self.nodes {
                if node.shard.parent_ids.contains(&id) {
                    if visited.insert(cid.clone()) {
                        queue.push_back(cid.clone());
                    }
                }
            }
        }
        out
    }

    /// Axis-aligned bounding box of all nodes in world space. Returns (min, max) or None if empty.
    pub fn compute_bbox(&self) -> Option<(Vec2, Vec2)> {
        let mut min = Vec2::splat(f32::MAX);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for node in self.nodes.values() {
            let r0 = node.position;
            let r1 = node.position + node.size;
            min = min.min(r0);
            max = max.max(r1);
        }
        if min.x == f32::MAX {
            None
        } else {
            Some((min, max))
        }
    }

    pub fn clear(&mut self) {
        self.graph_id = None;
        self.root_id = None;
        self.current_leaf_id = None;
        self.nodes.clear();
    }

    pub fn bump_content_version(&mut self) {
        self.content_version = self.content_version.wrapping_add(1);
    }

    /// Children of a node (nodes whose parent_ids contains id). Order: by creation/insertion.
    pub fn children_ids(&self, id: &str) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.shard.parent_ids.iter().any(|p| p == id))
            .map(|(cid, _)| cid.clone())
            .collect()
    }

    /// Set the active leaf, recording it as the last-visited child of its primary parent.
    pub fn set_current_leaf(&mut self, id: String) {
        let primary_parent = self.nodes
            .get(&id)
            .and_then(|n| n.shard.parent_ids.first().cloned())
            .filter(|pid| self.nodes.contains_key(pid));
        if let Some(parent_id) = primary_parent {
            self.last_visited_child.insert(parent_id, id.clone());
        }
        self.current_leaf_id = Some(id);
    }

    /// Parent IDs of a node (for edges and arrow navigation).
    pub fn parent_ids(&self, id: &str) -> Vec<String> {
        self.nodes
            .get(id)
            .map(|n| n.shard.parent_ids.clone())
            .unwrap_or_default()
    }

    /// Siblings of a node: other children of the same primary parent, sorted by ID (matches layout order).
    pub fn sibling_ids(&self, id: &str) -> Vec<String> {
        let primary_parent = self.nodes
            .get(id)
            .and_then(|n| n.shard.parent_ids.first().cloned())
            .filter(|pid| self.nodes.contains_key(pid));
        let Some(parent_id) = primary_parent else { return vec![]; };
        let mut siblings: Vec<String> = self.nodes
            .iter()
            .filter(|(sid, n)| *sid != id && n.shard.parent_ids.first().map(|p| p == &parent_id).unwrap_or(false))
            .map(|(sid, _)| sid.clone())
            .collect();
        siblings.sort();
        siblings
    }

    /// Previous sibling in sorted order, wrapping around.
    pub fn prev_sibling_id(&self, id: &str) -> Option<String> {
        let parent_id = self.nodes
            .get(id)
            .and_then(|n| n.shard.parent_ids.first().cloned())
            .filter(|pid| self.nodes.contains_key(pid))?;
        let mut all_children: Vec<String> = self.nodes
            .iter()
            .filter(|(_, n)| n.shard.parent_ids.first().map(|p| p == &parent_id).unwrap_or(false))
            .map(|(cid, _)| cid.clone())
            .collect();
        all_children.sort();
        let pos = all_children.iter().position(|s| s == id)?;
        if pos == 0 { None } else { Some(all_children[pos - 1].clone()) }
    }

    /// Next sibling in sorted order.
    pub fn next_sibling_id(&self, id: &str) -> Option<String> {
        let parent_id = self.nodes
            .get(id)
            .and_then(|n| n.shard.parent_ids.first().cloned())
            .filter(|pid| self.nodes.contains_key(pid))?;
        let mut all_children: Vec<String> = self.nodes
            .iter()
            .filter(|(_, n)| n.shard.parent_ids.first().map(|p| p == &parent_id).unwrap_or(false))
            .map(|(cid, _)| cid.clone())
            .collect();
        all_children.sort();
        let pos = all_children.iter().position(|s| s == id)?;
        all_children.get(pos + 1).cloned()
    }

    /// Measure markdown block with wrap; returns (width, height) aligned with
    /// [`crate::gfx::renderer::Renderer::build_markdown_scene`].
    fn measure_markdown_block<M: FnMut(&str, f32, f32, bool, bool) -> ParagraphWrappedFlow>(
        measure: &mut M,
        markdown: &str,
        max_width: f32,
        font_size: f32,
    ) -> Vec2 {
        fn effective_font_size(font_size: f32, _bold: bool, _italic: bool, _code: bool) -> f32 {
            font_size
        }

        fn flush_inline<M: FnMut(&str, f32, f32, bool, bool) -> ParagraphWrappedFlow>(
            measure: &mut M,
            text: &mut String,
            font_size: f32,
            bold: bool,
            italic: bool,
            code: bool,
            current_x: &mut f32,
            total_h: &mut f32,
            max_w: &mut f32,
            max_width: f32,
            line_max_font: &mut f32,
            line_has_content: &mut bool,
            trailing_segment: bool,
        ) {
            if text.is_empty() {
                return;
            }
            let size = effective_font_size(font_size, bold, italic, code);
            let seg_x = *current_x;
            let rem = (max_width - seg_x).max(1.0);
            let flow = measure(text, size, rem, bold, italic);
            *current_x = seg_x + flow.last_line_advance;
            if trailing_segment {
                *total_h += flow.content_height;
            } else {
                *total_h += flow.content_height - flow.last_line_height;
            }
            *max_w = (*max_w).max(seg_x + flow.layout_width);
            *line_max_font = (*line_max_font).max(size);
            *line_has_content = true;
            text.clear();
        }

        let line_ratio = style::font_size::LINE_HEIGHT_RATIO;
        let mut current_x = 0.0f32;
        let mut max_w = 0.0f32;
        let mut total_h = 0.0f32;
        let mut line_max_font = font_size;
        let mut line_has_content = false;
        let mut current_text = String::new();
        let mut is_bold = false;
        let mut is_italic = false;
        let mut is_code = false;
        #[derive(Clone, Copy)]
        struct ListFrame {
            ordered: bool,
            next_n: u64,
        }
        let mut list_stack: Vec<ListFrame> = Vec::new();

        for event in Parser::new(markdown) {
            match event {
                Event::Start(Tag::List(first)) => {
                    list_stack.push(ListFrame {
                        ordered: first.is_some(),
                        next_n: first.unwrap_or(1),
                    });
                }
                Event::End(TagEnd::List(_)) => {
                    list_stack.pop();
                }
                Event::Start(Tag::Item) => {
                    let prefix = list_stack.last().map(|f| {
                        if f.ordered {
                            format!("{}. ", f.next_n)
                        } else {
                            "• ".to_string()
                        }
                    });
                    if let Some(p) = prefix {
                        current_text.push_str(&p);
                    }
                }
                Event::End(TagEnd::Item) => {
                    if let Some(f) = list_stack.last_mut() {
                        if f.ordered {
                            f.next_n += 1;
                        }
                    }
                    if !current_text.is_empty() {
                        let sz = effective_font_size(font_size, is_bold, is_italic, is_code);
                        let rem = (max_width - current_x).max(1.0);
                        let flow = measure(&current_text, sz, rem, is_bold, is_italic);
                        total_h += flow.content_height;
                        max_w = max_w.max(current_x + flow.layout_width);
                        line_max_font = line_max_font.max(sz);
                        line_has_content = true;
                        current_text.clear();
                    }
                    current_x = 0.0;
                    line_max_font = font_size;
                    line_has_content = false;
                }
                Event::End(TagEnd::Paragraph) => {
                    if !current_text.is_empty() {
                        let sz = effective_font_size(font_size, is_bold, is_italic, is_code);
                        let rem = (max_width - current_x).max(1.0);
                        let flow = measure(&current_text, sz, rem, is_bold, is_italic);
                        total_h += flow.content_height;
                        max_w = max_w.max(current_x + flow.layout_width);
                        line_max_font = line_max_font.max(sz);
                        line_has_content = true;
                        current_text.clear();
                    } else {
                        total_h += font_size * line_ratio;
                    }
                    current_x = 0.0;
                    line_max_font = font_size;
                    line_has_content = false;
                }
                Event::Start(Tag::Strong) => {
                    flush_inline(
                        measure,
                        &mut current_text,
                        font_size,
                        is_bold,
                        is_italic,
                        is_code,
                        &mut current_x,
                        &mut total_h,
                        &mut max_w,
                        max_width,
                        &mut line_max_font,
                        &mut line_has_content,
                        false,
                    );
                    is_bold = true;
                }
                Event::End(pulldown_cmark::TagEnd::Strong) => {
                    flush_inline(
                        measure,
                        &mut current_text,
                        font_size,
                        is_bold,
                        is_italic,
                        is_code,
                        &mut current_x,
                        &mut total_h,
                        &mut max_w,
                        max_width,
                        &mut line_max_font,
                        &mut line_has_content,
                        false,
                    );
                    is_bold = false;
                }
                Event::Start(Tag::Emphasis) => {
                    flush_inline(
                        measure,
                        &mut current_text,
                        font_size,
                        is_bold,
                        is_italic,
                        is_code,
                        &mut current_x,
                        &mut total_h,
                        &mut max_w,
                        max_width,
                        &mut line_max_font,
                        &mut line_has_content,
                        false,
                    );
                    is_italic = true;
                }
                Event::End(pulldown_cmark::TagEnd::Emphasis) => {
                    flush_inline(
                        measure,
                        &mut current_text,
                        font_size,
                        is_bold,
                        is_italic,
                        is_code,
                        &mut current_x,
                        &mut total_h,
                        &mut max_w,
                        max_width,
                        &mut line_max_font,
                        &mut line_has_content,
                        false,
                    );
                    is_italic = false;
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    flush_inline(
                        measure,
                        &mut current_text,
                        font_size,
                        is_bold,
                        is_italic,
                        is_code,
                        &mut current_x,
                        &mut total_h,
                        &mut max_w,
                        max_width,
                        &mut line_max_font,
                        &mut line_has_content,
                        false,
                    );
                    is_code = true;
                }
                Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                    flush_inline(
                        measure,
                        &mut current_text,
                        font_size,
                        is_bold,
                        is_italic,
                        is_code,
                        &mut current_x,
                        &mut total_h,
                        &mut max_w,
                        max_width,
                        &mut line_max_font,
                        &mut line_has_content,
                        false,
                    );
                    is_code = false;
                }
                Event::Text(text) => {
                    let size = effective_font_size(font_size, is_bold, is_italic, is_code);
                    for word in text.split_whitespace() {
                        let candidate = if current_text.is_empty() {
                            word.to_string()
                        } else {
                            format!(" {}", word)
                        };
                        let test = format!("{}{}", current_text, candidate);
                        let rem_try = (max_width - current_x).max(1.0);
                        let try_flow = measure(&test, size, rem_try, is_bold, is_italic);
                        if current_x + try_flow.layout_width > max_width && !current_text.is_empty()
                        {
                            let rem_flush = (max_width - current_x).max(1.0);
                            let flow = measure(
                                &current_text,
                                size,
                                rem_flush,
                                is_bold,
                                is_italic,
                            );
                            total_h += flow.content_height;
                            max_w = max_w.max(current_x + flow.layout_width);
                            line_max_font = line_max_font.max(size);
                            line_has_content = true;
                            current_text.clear();
                            current_x = 0.0;
                            current_text = word.to_string();
                        } else {
                            current_text = test;
                        }
                    }
                }
                Event::SoftBreak | Event::HardBreak => {
                    if !current_text.is_empty() {
                        let sz = effective_font_size(font_size, is_bold, is_italic, is_code);
                        let rem = (max_width - current_x).max(1.0);
                        let flow = measure(
                            &current_text,
                            sz,
                            rem,
                            is_bold,
                            is_italic,
                        );
                        total_h += flow.content_height;
                        max_w = max_w.max(current_x + flow.layout_width);
                        line_max_font = line_max_font.max(sz);
                        line_has_content = true;
                        current_text.clear();
                    } else {
                        total_h += font_size * line_ratio;
                    }
                    current_x = 0.0;
                    line_max_font = font_size;
                    line_has_content = false;
                }
                _ => {}
            }
        }

        flush_inline(
            measure,
            &mut current_text,
            font_size,
            is_bold,
            is_italic,
            is_code,
            &mut current_x,
            &mut total_h,
            &mut max_w,
            max_width,
            &mut line_max_font,
            &mut line_has_content,
            true,
        );
        if total_h == 0.0 {
            total_h = font_size * line_ratio;
        }
        max_w = max_w.max(current_x);

        Vec2::new(max_w.max(1.0).min(max_width), total_h)
    }

    /// World rect (min, max) for viewport culling. When set, only nodes whose AABB intersects this rect are measured.
    fn node_in_visible_rect(
        position: Vec2,
        size: Vec2,
        visible_min: Vec2,
        visible_max: Vec2,
    ) -> bool {
        !(position.x + size.x < visible_min.x
            || position.x > visible_max.x
            || position.y + size.y < visible_min.y
            || position.y > visible_max.y)
    }

    /// Update each node's size from its content (user + assistant bubbles). Single source of truth for `node.size`;
    /// call each frame before constellation render (see renderer) so the shard background and hit-test stay in sync.
    /// When a node is being edited in place, pass its id and the current edit textarea size in `editing_override`
    /// so the shard background resizes to fit the message box.
    /// When `visible_rect` is `Some((min, max))` in world space, only out-of-view nodes are skipped — unless
    /// `layout_dirty` is true, in which case all nodes are measured so the tree layout has accurate heights.
    pub fn update_node_sizes<M: FnMut(&str, f32, f32, bool, bool) -> ParagraphWrappedFlow>(
        &mut self,
        measure: M,
        editing_override: Option<(&str, Vec2)>,
        visible_rect: Option<(Vec2, Vec2)>,
        viewport_width: f32,
    ) {
        const PADDING: f32 = style::padding::SMALL;
        const BUBBLE_SPACING: f32 = 6.0;
        const ACTION_ROW_HEIGHT: f32 = 28.0;
        const MSG_BUTTON_ROW_RESERVE: f32 = 22.0;
        const FONT_SIZE: f32 = style::font_size::MESSAGE_BODY;

        // Default card width: 80% of viewport, floored at STANDARD_SHARD_WIDTH.
        let card_width = (viewport_width * 0.8).max(STANDARD_SHARD_WIDTH);
        let min_node_height = MIN_SHARD_MANUAL_HEIGHT;
        let lateral = (style::padding::SMALL + style::padding::SHARD_MESSAGE_INSET) * 2.0;
        // Initial wrap width derived from standard card width (matches constellation bubble geometry).
        let text_wrap_width = (card_width * style::constellation::BUBBLE_MAX_WIDTH_RATIO - lateral)
            .max(style::constellation::BUBBLE_MIN_CONTENT_WIDTH);

        // When layout is dirty we must measure all nodes (even off-screen) so the tree layout
        // has accurate heights. Otherwise apply the visible_rect cull for performance.
        let skip_cull = self.layout_dirty;

        let mut measure = measure;
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        for node_id in &node_ids {
            {
                let node = self.nodes.get(node_id).unwrap();
                if !skip_cull {
                    if let Some((visible_min, visible_max)) = visible_rect {
                        if !Self::node_in_visible_rect(node.position, node.size, visible_min, visible_max) {
                            continue;
                        }
                    }
                }
            }

            // Determine the actual card width for this node (manual override or default).
            let shard_w = self
                .manual_sizes
                .get(node_id)
                .map(|m| m.x.clamp(MIN_SHARD_MANUAL_WIDTH, card_width * 2.0))
                .unwrap_or(card_width);
            let bubble_inner_floor = (shard_w * style::constellation::BUBBLE_MAX_WIDTH_RATIO - lateral)
                .max(style::constellation::BUBBLE_MIN_CONTENT_WIDTH);
            let wrap_w = text_wrap_width.max(bubble_inner_floor);

            let node = self.nodes.get(node_id).unwrap();
            let assistant_size = editing_override.as_ref().and_then(|(id, size)| {
                if *id == node_id.as_str() { Some(*size) } else { None }
            });

            let mut user_text_h = 0.0f32;
            let mut user_text_w = 0.0f32;
            let mut assistant_text_h = 0.0f32;
            let mut assistant_text_w = 0.0f32;

            if let Some(ref u) = node.shard.user_content.clone() {
                if !u.is_empty() {
                    let sz = Self::measure_markdown_block(&mut measure, &u, wrap_w, FONT_SIZE);
                    user_text_h = sz.y;
                    user_text_w = sz.x.max(bubble_inner_floor);
                }
            }
            if let Some(edit_size) = assistant_size {
                assistant_text_h = edit_size.y;
                assistant_text_w = edit_size.x;
            } else if let Some(ref a) = node.shard.assistant_content.clone() {
                if !a.is_empty() {
                    let sz = Self::measure_markdown_block(&mut measure, &a, wrap_w, FONT_SIZE);
                    assistant_text_h = sz.y;
                    assistant_text_w = sz.x.max(bubble_inner_floor);
                }
            }

            let node = self.nodes.get(node_id).unwrap();
            let mut content_h = PADDING;
            if user_text_h > 0.0 {
                content_h += user_text_h + PADDING * 2.0 + MSG_BUTTON_ROW_RESERVE + BUBBLE_SPACING;
            }
            if let Some(edit_size) = assistant_size {
                content_h += edit_size.y + PADDING * 2.0 + MSG_BUTTON_ROW_RESERVE;
            } else if assistant_text_h > 0.0 {
                content_h += assistant_text_h + PADDING * 2.0 + MSG_BUTTON_ROW_RESERVE;
            }

            const CITATION_LINE_HEIGHT: f32 =
                style::font_size::SMALL * style::font_size::LINE_HEIGHT_RATIO;
            const CITATION_GAP: f32 = 4.0;
            if !node.shard.citations.is_empty() {
                content_h += CITATION_GAP + node.shard.citations.len() as f32 * CITATION_LINE_HEIGHT;
            }
            const NOTE_LINE_H: f32 = style::font_size::LARGE;
            const NOTES_GAP: f32 = 4.0;
            if !node.shard.notes.is_empty() {
                content_h += NOTES_GAP + node.shard.notes.len() as f32 * NOTE_LINE_H;
            }
            content_h += PADDING + ACTION_ROW_HEIGHT;

            let raw_h = content_h.max(min_node_height);
            let node = self.nodes.get_mut(node_id).unwrap();
            if let Some(manual) = self.manual_sizes.get(node_id) {
                node.size = Vec2::new(
                    manual.x.clamp(MIN_SHARD_MANUAL_WIDTH, card_width * 2.0),
                    manual.y.clamp(min_node_height, raw_h),
                );
            } else {
                node.size = Vec2::new(shard_w, raw_h);
            }
            node.content_height = raw_h;
            node.user_text_height = user_text_h;
            node.assistant_text_height = assistant_text_h;
            node.user_text_width = user_text_w;
            node.assistant_text_width = assistant_text_w;
        }

        // Run tree layout after measuring all nodes when sizes have changed.
        if self.layout_dirty {
            self.apply_tree_layout();
        }
    }
}
