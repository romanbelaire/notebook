//! Graph state for Constellar: one graph per "current conversation", keyed by graph_id.
//! Mirrors backend ConversationGraph + ActiveState; nodes have world position/velocity for constellation UI.

use glam::Vec2;
use std::collections::HashMap;
use crate::api::models::GraphShardResponse;
use crate::persistence::GraphLayoutPersistence;

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

/// Node in the constellation: shard data + world position and velocity for layout/physics.
#[derive(Clone, Debug)]
pub struct ConstellationNode {
    pub shard: GraphShard,
    pub position: Vec2,
    pub velocity: Vec2,
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
}

impl GraphState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.graph_id.is_none() && self.nodes.is_empty()
    }

    pub fn set_graph(&mut self, graph_id: String, root_id: String, current_leaf_id: String, shards: HashMap<String, GraphShard>) {
        self.content_version = self.content_version.wrapping_add(1);
        self.graph_id = Some(graph_id.clone());
        self.root_id = Some(root_id.clone());
        self.current_leaf_id = Some(current_leaf_id);
        let stored = GraphLayoutPersistence::load_positions(&graph_id);
        // Do not add the empty root to nodes (backend creates root with empty strings); first real message attaches to root.
        self.nodes = shards
            .into_iter()
            .filter(|(id, s)| s.is_turn() && !(id == &root_id && s.is_empty_content()))
            .map(|(id, shard)| {
                let position = stored.get(&id).copied().unwrap_or(Vec2::ZERO);
                let velocity = Vec2::ZERO;
                let size = Vec2::new(280.0, 120.0); // Placeholder; layout will measure
                (
                    id,
                    ConstellationNode {
                        shard,
                        position,
                        velocity,
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
                    },
                )
            })
            .collect();
        self.promote_empty_root_to_first_child();
        // Only run BFS for nodes that have no stored position
        self.apply_initial_layout_merge(&stored);
        self.recompute_hierarchy_metadata();
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

    /// Initial layout: use stored positions where available; BFS for nodes without stored position.
    /// When root has no shard (not in nodes), layout starts from nodes whose parent is root_id.
    fn apply_initial_layout_merge(&mut self, stored: &HashMap<String, Vec2>) {
        let root_id = match &self.root_id {
            Some(r) => r.clone(),
            None => return,
        };
        const DX: f32 = 220.0;
        const GAP: f32 = 24.0;
        const PLACEHOLDER_CHILD_HEIGHT: f32 = 120.0;
        // Anchor logical root at origin when present; otherwise treat its children as first layer
        // and place them in a vertical column rooted at the origin before BFS.
        let mut queue: Vec<String> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        if let Some(n) = self.nodes.get_mut(&root_id) {
            if !stored.contains_key(&root_id) {
                n.position = Vec2::ZERO;
            }
            queue.push(root_id.clone());
            visited.insert(root_id.clone());
        } else {
            let children = self.children_ids(&root_id);
            let mut cursor_y = 0.0f32;
            for cid in children {
                if let Some(child) = self.nodes.get_mut(&cid) {
                    if !stored.contains_key(&cid) {
                        let cx = 0.0;
                        let cy = cursor_y;
                        child.position = Vec2::new(cx, cy);
                    }
                }
                visited.insert(cid.clone());
                queue.push(cid.clone());
                cursor_y += PLACEHOLDER_CHILD_HEIGHT + GAP;
            }
        }
        while let Some(id) = queue.pop() {
            let parent_pos = self.nodes.get(&id).map(|n| n.position).unwrap_or(Vec2::ZERO);
            let parent_size = self.nodes.get(&id).map(|n| n.size).unwrap_or(Vec2::new(280.0, 120.0));
            let parent_bottom = parent_pos.y + parent_size.y;
            let children: Vec<String> = self
                .nodes
                .iter()
                .filter(|(_, n)| n.shard.parent_ids.contains(&id))
                .map(|(cid, _)| cid.clone())
                .collect();
            for (i, cid) in children.into_iter().enumerate() {
                if visited.insert(cid.clone()) {
                    if let Some(n) = self.nodes.get_mut(&cid) {
                        if !stored.contains_key(&cid) {
                            let cx = parent_pos.x + DX;
                            let cy = parent_bottom + GAP + (i as f32) * (PLACEHOLDER_CHILD_HEIGHT + GAP);
                            n.position = Vec2::new(cx, cy);
                        }
                    }
                    queue.push(cid);
                }
            }
        }
    }

    pub fn add_node(&mut self, shard: GraphShard, position: Vec2) {
        self.content_version = self.content_version.wrapping_add(1);
        let id = shard.id.clone();
        let velocity = Vec2::ZERO;
        let size = Vec2::new(280.0, 120.0);
        self.nodes.insert(
            id.clone(),
            ConstellationNode {
                shard,
                position,
                velocity,
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
            },
        );
        // When the first shard is created via chat (child of empty root), promote it to root.
        self.promote_empty_root_to_first_child();
        // If this node became the root, place it at origin.
        if self.root_id.as_ref() == Some(&id) {
            if let Some(n) = self.nodes.get_mut(&id) {
                n.position = Vec2::ZERO;
            }
        }
        self.recompute_hierarchy_metadata();
    }

    /// Recompute depth (BFS from root) and child_count for all nodes.
    fn recompute_hierarchy_metadata(&mut self) {
        use std::collections::{HashMap, VecDeque};

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

    /// ID of the node used as the physics anchor when present (logical root).
    fn anchored_root_id(&self) -> Option<String> {
        self.root_id
            .as_ref()
            .and_then(|rid| {
                if self.nodes.contains_key(rid) {
                    Some(rid.clone())
                } else {
                    None
                }
            })
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
        let mut queue = std::collections::VecDeque::new();
        if self.nodes.contains_key(&root_id) {
            queue.push_back(root_id);
        } else {
            for cid in self.children_ids(&root_id) {
                queue.push_back(cid);
            }
        }
        let mut out = Vec::new();
        let mut visited = std::collections::HashSet::new();
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

    /// Physics constants for BOIDS-like layout
    const TETHER_GAP: f32 = 24.0; // Rest length: parent bottom to child top
    const TETHER_STIFFNESS: f32 = 300.0; // Bouncy preset
    const TETHER_DAMPING: f32 = 12.0;
    const REPULSION_MIN_SEP: f32 = 90.0;
    const REPULSION_STRENGTH: f32 = 800.0;
    const VELOCITY_DAMPING: f32 = 4.0;
    const ROOT_CENTER_STRENGTH: f32 = 0.5;
    /// Nodes with speed below this are treated as frozen for repulsion (no repulsion force added).
    const PER_NODE_FREEZE_EPS: f32 = 0.005;

    /// Step physics: tether to parents, repulsion between nodes, damping. Call each frame when graph is active.
    pub fn step_physics(&mut self, dt: f32) {
        if self.nodes.is_empty() {
            return;
        }
        let root_id = match self.anchored_root_id() {
            Some(r) => r,
            None => return,
        };

        // Collect positions and sizes for force computation (read-only)
        let positions: HashMap<String, Vec2> = self
            .nodes
            .iter()
            .map(|(id, n)| (id.clone(), n.position))
            .collect();
        let sizes: HashMap<String, Vec2> = self
            .nodes
            .iter()
            .map(|(id, n)| (id.clone(), n.size))
            .collect();
        let speeds: HashMap<String, f32> = self
            .nodes
            .iter()
            .map(|(id, n)| (id.clone(), n.velocity.length()))
            .collect();

        // Effective radius per node: base card radius plus extra spacing based on child_count.
        let mut radii: HashMap<String, f32> = HashMap::new();
        for (id, node) in &self.nodes {
            let base = 0.5 * node.size.x.max(node.size.y);
            let extra = 12.0 * (node.child_count as f32).sqrt();
            radii.insert(id.clone(), base + extra);
        }

        // Spatial hash grid for repulsion: group nodes into cells so we only consider nearby neighbors.
        let mut max_dim = 0.0f32;
        for node in self.nodes.values() {
            max_dim = max_dim.max(node.size.x.max(node.size.y));
        }
        let max_dim = if max_dim > 0.0 { max_dim } else { 1.0 };
        let cell_size = max_dim * 2.0;
        let inv_cell_size = 1.0 / cell_size;
        let mut grid: HashMap<(i32, i32), Vec<String>> = HashMap::new();
        for (id, pos) in &positions {
            let size = sizes
                .get(id)
                .copied()
                .unwrap_or(Vec2::new(280.0, 120.0));
            let center = *pos + size * 0.5;
            let cx = (center.x * inv_cell_size).floor() as i32;
            let cy = (center.y * inv_cell_size).floor() as i32;
            grid.entry((cx, cy)).or_insert_with(Vec::new).push(id.clone());
        }

        for (id, node) in self.nodes.iter_mut() {
            if id == &root_id {
                // Keep the root pinned at origin so the rest of the graph stabilises around a fixed anchor.
                node.position = Vec2::ZERO;
                node.velocity = Vec2::ZERO;
                continue;
            }
            let mut force = Vec2::ZERO;

            // Tether to each parent: parent bottom-center → child top-center, rest length = TETHER_GAP.
            // When parent is root and root has no shard (not in nodes), tether to origin.
            for parent_id in &node.shard.parent_ids {
                let parent_attach = if parent_id == &root_id && !positions.contains_key(parent_id) {
                    Vec2::ZERO
                } else if let Some(&parent_pos) = positions.get(parent_id) {
                    let parent_size = sizes.get(parent_id).copied().unwrap_or(Vec2::new(280.0, 120.0));
                    parent_pos + Vec2::new(parent_size.x * 0.5, parent_size.y)
                } else {
                    continue;
                };
                let child_attach = node.position + Vec2::new(node.size.x * 0.5, 0.0);
                let delta = parent_attach - child_attach;
                let dist = delta.length().max(1.0);
                let stretch = dist - Self::TETHER_GAP;
                let spring_force = delta.normalize() * stretch * Self::TETHER_STIFFNESS * 0.01;
                force += spring_force;
            }

            // Repulsion from nearby nodes only (spatial hash grid). Skip for nearly-static nodes.
            let speed = node.velocity.length();
            if speed >= Self::PER_NODE_FREEZE_EPS {
                let my_center = node.position + node.size * 0.5;
                let my_radius = *radii.get(id).unwrap();
                let cell_x = (my_center.x * inv_cell_size).floor() as i32;
                let cell_y = (my_center.y * inv_cell_size).floor() as i32;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let key = (cell_x + dx, cell_y + dy);
                        if let Some(neighbors) = grid.get(&key) {
                            for other_id in neighbors {
                                if other_id == id {
                                    continue;
                                }
                                let other_pos = positions
                                    .get(other_id)
                                    .copied()
                                    .unwrap_or(Vec2::ZERO);
                                let other_size = sizes
                                    .get(other_id)
                                    .copied()
                                    .unwrap_or(Vec2::new(280.0, 120.0));
                                let other_center = other_pos + other_size * 0.5;
                                let other_radius = *radii.get(other_id).unwrap();
                                let delta = my_center - other_center;
                                let dist = delta.length().max(1.0);
                                let min_sep = my_radius + other_radius + Self::REPULSION_MIN_SEP;
                                if dist < min_sep {
                                    let overlap = min_sep - dist;
                                    let repulsion =
                                        delta.normalize() * overlap * Self::REPULSION_STRENGTH / dist;
                                    force += repulsion;
                                }
                            }
                        }
                    }
                }
            }

            // Integrate with depth- and parent-aware damping so parents settle before children.
            node.velocity += force * dt;

            let parent_max_speed = node
                .shard
                .parent_ids
                .iter()
                .map(|pid| *speeds.get(pid).unwrap_or(&0.0))
                .fold(0.0, f32::max);
            let depth = node.depth as f32;
            let parent_busy = parent_max_speed > 0.02;
            let extra_damping = if parent_busy && depth > 0.0 {
                (depth * 0.5).min(4.0)
            } else {
                0.0
            };
            let damping = Self::VELOCITY_DAMPING + extra_damping;

            node.velocity *= 1.0 - (damping * dt).min(0.9);
            node.position += node.velocity * dt;
        }
    }

    /// Sum of velocity magnitudes across all nodes. Used to skip physics when settled.
    pub fn total_velocity_magnitude(&self) -> f32 {
        self.nodes
            .values()
            .map(|n| n.velocity.length())
            .sum()
    }

    /// When physics is not running and total velocity is below threshold, zero all velocities
    /// so we reach a clean "resting" state and avoid continuous redraws from floating-point drift.
    pub fn zero_velocities_if_settled(&mut self, threshold: f32) {
        if self.total_velocity_magnitude() <= threshold {
            for node in self.nodes.values_mut() {
                node.velocity = Vec2::ZERO;
            }
        }
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

    /// Parent IDs of a node (for edges and Alt+arrow).
    pub fn parent_ids(&self, id: &str) -> Vec<String> {
        self.nodes
            .get(id)
            .map(|n| n.shard.parent_ids.clone())
            .unwrap_or_default()
    }

    /// Measure text block with word wrap; returns (width, height). Used for adaptive node sizing.
    fn measure_block<M: FnMut(&str, f32) -> Vec2>(measure: &mut M, text: &str, max_width: f32, font_size: f32) -> Vec2 {
        let line_height = font_size * 1.2;
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_line = String::new();
        let mut max_w = 0.0f32;
        let mut line_count = 0u32;
        for word in words {
            let test_line = if current_line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", current_line, word)
            };
            let test_size = measure(&test_line, font_size);
            if test_size.x > max_width && !current_line.is_empty() {
                max_w = max_w.max(measure(&current_line, font_size).x);
                line_count += 1;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }
        if !current_line.is_empty() {
            line_count += 1;
            max_w = max_w.max(measure(&current_line, font_size).x);
        }
        let h = (line_count.max(1) as f32) * line_height;
        Vec2::new(max_w, h)
    }

    /// World rect (min, max) for viewport culling. When set, only nodes whose AABB intersects this rect are measured.
    fn node_in_visible_rect(position: Vec2, size: Vec2, visible_min: Vec2, visible_max: Vec2) -> bool {
        !(position.x + size.x < visible_min.x
            || position.x > visible_max.x
            || position.y + size.y < visible_min.y
            || position.y > visible_max.y)
    }

    /// Update each node's size from its content (user + assistant bubbles). Single source of truth for `node.size`;
    /// call each frame before constellation render (see renderer) so the shard background and hit-test stay in sync.
    /// When a node is being edited in place, pass its id and the current edit textarea size in `editing_override`
    /// so the shard background resizes to fit the message box. Future manual resize can plug in by constraining
    /// or overriding the computed size here.
    /// When `visible_rect` is `Some((min, max))` in world space, only nodes whose AABB intersects that rect are updated.
    pub fn update_node_sizes<M: FnMut(&str, f32) -> Vec2>(
        &mut self,
        viewport_size: Vec2,
        measure: M,
        editing_override: Option<(&str, Vec2)>,
        visible_rect: Option<(Vec2, Vec2)>,
    ) {
        const PADDING: f32 = 8.0;
        const BUBBLE_SPACING: f32 = 6.0;
        const ACTION_ROW_HEIGHT: f32 = 28.0;  // action buttons + pin area
        const MSG_BUTTON_ROW_RESERVE: f32 = 22.0;  // space at bottom of each bubble for edit/hide buttons
        const FONT_SIZE: f32 = 16.0;
        const MIN_NODE_WIDTH: f32 = 200.0;
        const MIN_NODE_HEIGHT: f32 = 60.0;
        const MAX_CARD_HEIGHT: f32 = 360.0; // Fixed cap so long messages truncate and scroll within card

        // Allow slightly wider cards so default zoom text isn't overly wrapped.
        let max_width = (viewport_size.x * 0.8).max(MIN_NODE_WIDTH);
        let max_height = viewport_size.y * 0.75;

        let mut measure = measure;
        for (node_id, node) in self.nodes.iter_mut() {
            if let Some((visible_min, visible_max)) = visible_rect {
                if !Self::node_in_visible_rect(node.position, node.size, visible_min, visible_max) {
                    continue;
                }
            }
            let mut content_w = 0.0f32;
            let mut content_h = PADDING; // top margin
            let mut user_text_h = 0.0f32;
            let mut user_text_w = 0.0f32;
            let mut assistant_text_h = 0.0f32;
            let mut assistant_text_w = 0.0f32;

            if let Some(ref u) = node.shard.user_content {
                if !u.is_empty() {
                    let sz = Self::measure_block(&mut measure, u, max_width - PADDING * 2.0, FONT_SIZE);
                    user_text_h = sz.y;
                    user_text_w = sz.x;
                    content_w = content_w.max(sz.x);
                    content_h += sz.y + PADDING * 2.0 + MSG_BUTTON_ROW_RESERVE + BUBBLE_SPACING; // user bubble (text + pad + button row) + spacing
                }
            }
            let assistant_size = editing_override.as_ref().and_then(|(id, size)| {
                if *id == node_id.as_str() { Some(*size) } else { None }
            });
            if let Some(edit_size) = assistant_size {
                assistant_text_h = edit_size.y;
                assistant_text_w = edit_size.x;
                content_w = content_w.max(edit_size.x + PADDING * 2.0);
                content_h += edit_size.y + PADDING * 2.0 + MSG_BUTTON_ROW_RESERVE; // assistant bubble when editing
            } else if let Some(ref a) = node.shard.assistant_content {
                if !a.is_empty() {
                    let sz = Self::measure_block(&mut measure, a, max_width - PADDING * 2.0, FONT_SIZE);
                    assistant_text_h = sz.y;
                    assistant_text_w = sz.x;
                    content_w = content_w.max(sz.x);
                    content_h += sz.y + PADDING * 2.0 + MSG_BUTTON_ROW_RESERVE; // assistant bubble (text + pad + button row)
                }
            }
            const CITATION_LINE_HEIGHT: f32 = 14.4;
            const CITATION_GAP: f32 = 4.0;
            if !node.shard.citations.is_empty() {
                content_h += CITATION_GAP + node.shard.citations.len() as f32 * CITATION_LINE_HEIGHT;
            }
            const NOTE_LINE_H: f32 = 18.0;
            const NOTES_GAP: f32 = 4.0;
            if !node.shard.notes.is_empty() {
                content_h += NOTES_GAP + node.shard.notes.len() as f32 * NOTE_LINE_H;
            }

            content_h += PADDING; // bottom margin
            content_h += ACTION_ROW_HEIGHT; // pin + action buttons
            let raw_w = (content_w + PADDING * 2.0).max(MIN_NODE_WIDTH);
            let raw_h = content_h.max(MIN_NODE_HEIGHT);

            // Size to content; cap at max for very long messages (fixed MAX_CARD_HEIGHT ensures truncation)
            let w = raw_w.min(max_width);
            let h = raw_h.min(max_height).min(MAX_CARD_HEIGHT).max(MIN_NODE_HEIGHT);
            node.size = Vec2::new(w, h);
            node.content_height = raw_h;
            node.user_text_height = user_text_h;
            node.assistant_text_height = assistant_text_h;
            node.user_text_width = user_text_w;
            node.assistant_text_width = assistant_text_w;
        }
    }
}
