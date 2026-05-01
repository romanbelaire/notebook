use crate::app::App;
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::gfx::types::{elbow_to_vertices, Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::{text_input_render, Rect};
use crate::ui::style;
use crate::ui::{Text, TextAlignment};
use glam::{Vec2, Vec4};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

/// Format a single citation from GraphShard's serde_json::Value for display (title, source, year, section, page).
fn format_constellation_citation(idx: usize, v: &serde_json::Value) -> String {
    let mut s = format!("[{}] ", idx + 1);
    if let Some(t) = v.get("title").and_then(|t| t.as_str()) {
        s.push_str(t);
    }
    s.push_str(" (");
    if let Some(src) = v.get("source").and_then(|x| x.as_str()) {
        s.push_str(src);
    }
    if let Some(yr) = v.get("year").and_then(|x| x.as_str()) {
        s.push_str(", ");
        s.push_str(yr);
    }
    s.push(')');
    if let Some(sec) = v.get("section").and_then(|x| x.as_str()) {
        s.push_str(" – ");
        s.push_str(sec);
    }
    if let Some(p) = v.get("page").and_then(|x| x.as_u64()) {
        s.push_str(&format!(", p.{}", p));
    }
    s
}

/// Measure text block with word wrap; returns (width, height). Used for bubble layout.
fn measure_wrapped_block(
    measure: &mut impl FnMut(&str, f32) -> Vec2,
    text: &str,
    max_width: f32,
    font_size: f32,
) -> Vec2 {
    let line_height = font_size * style::font_size::LINE_HEIGHT_RATIO;
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
        let test_size = measure(test_line.as_str(), font_size);
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

/// Extra screen-space slack so [`ConstellationView::camera_position_animated`] / [`ConstellationView::scale_animated`]
/// lerping does not move shard rects across the cull boundary frame-to-frame.
const SHARD_GEOMETRY_CULL_STABILITY_PAD_PX: f32 = 32.0;

fn macro_jiggle(id: &str, t_seconds: f32) -> Vec2 {
    let mut hx = DefaultHasher::new();
    id.hash(&mut hx);
    let p1 = (hx.finish() & 0xFFFF) as f32 / 65535.0;
    let mut hy = DefaultHasher::new();
    format!("{}:y", id).hash(&mut hy);
    let p2 = (hy.finish() & 0xFFFF) as f32 / 65535.0;
    let amp = style::constellation::MACRO_JIGGLE_AMPLITUDE_PX;
    let w = std::f32::consts::TAU * style::constellation::MACRO_JIGGLE_RATE_HZ;
    Vec2::new(
        ((t_seconds * w) + p1 * std::f32::consts::TAU).sin() * amp,
        ((t_seconds * w * 0.87) + p2 * std::f32::consts::TAU).cos() * amp,
    )
}

fn render_constellation_macro(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let chat = app.chat_window.as_ref().unwrap();
    let view = &chat.constellation_view;
    let graph = &app.graph_state;
    let scale = view.scale_animated;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f32();

    let edge_thickness = (style::stroke::GRAPH_EDGE_PX * 0.5 * scale).max(1.0);
    for node in graph.nodes.values() {
        let child_center = node.position + node.size * 0.5;
        for parent_id in &node.shard.parent_ids {
            if let Some(parent) = graph.get_node(parent_id) {
                let parent_center = parent.position + parent.size * 0.5;
                let mut a = view.world_to_screen(parent_center) + macro_jiggle(parent_id, now);
                let mut b = view.world_to_screen(child_center) + macro_jiggle(&node.shard.id, now);
                if !view.contains_screen(a) && !view.contains_screen(b) {
                    continue;
                }
                a.y += 1.0;
                b.y -= 1.0;
                elbow_to_vertices(
                    a,
                    b,
                    style::constellation::EDGE_CORNER_RADIUS * 0.35 * scale,
                    style::stroke::GRAPH_EDGE_STEPS / 2,
                    edge_thickness,
                    style::edge_palette::sample(node.ribbon_hue_t).truncate().extend(0.45),
                    vertices,
                );
            }
        }
    }

    let dot_r = style::constellation::MACRO_NODE_RADIUS_PX * scale;
    for (id, node) in &graph.nodes {
        let center = node.position + node.size * 0.5;
        let center_screen = view.world_to_screen(center) + macro_jiggle(id, now);
        if !view.contains_screen(center_screen) {
            continue;
        }
        let is_hovered = chat.hovered_node_id.as_deref() == Some(id.as_str());
        let is_focused = graph.current_leaf_id.as_deref() == Some(id.as_str());
        let is_selected = chat.macro_selected_node_ids.contains(id);
        let base = style::edge_palette::sample(node.ribbon_hue_t);
        let fill_color = if is_focused {
            style::accent::PHOSPHOR_GLOW()
        } else {
            base.truncate().extend(0.9)
        };
        let dot = Quad {
            position: center_screen - Vec2::splat(dot_r),
            size: Vec2::splat(dot_r * 2.0),
            color: fill_color,
            corner_radius: dot_r,
            bubble_effect: false,
            slider_effect: false,
        };
        renderer.add_quad(&dot, None);

        if is_selected || is_hovered {
            let ring_w = style::constellation::MACRO_SELECTED_RING_PX * scale;
            let ring_r = dot_r + ring_w;
            let ring = Quad {
                position: center_screen - Vec2::splat(ring_r),
                size: Vec2::splat(ring_r * 2.0),
                color: Vec4::new(
                    style::accent::PHOSPHOR().x,
                    style::accent::PHOSPHOR().y,
                    style::accent::PHOSPHOR().z,
                    ring_w,
                ),
                corner_radius: -ring_r,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&ring.to_vertices());
        }
    }
}

/// Cull in **screen space** against the constellation widget rect — matches [`ConstellationView::world_to_screen`].
/// `pad_px` expands the visible region so near-edge cards are not dropped while still on screen.
fn screen_rect_intersects_view(pos: Vec2, size: Vec2, view_pos: Vec2, view_size: Vec2, pad_px: f32) -> bool {
    let vx0 = view_pos.x - pad_px;
    let vy0 = view_pos.y - pad_px;
    let vx1 = view_pos.x + view_size.x + pad_px;
    let vy1 = view_pos.y + view_size.y + pad_px;
    !(pos.x + size.x < vx0 || pos.x > vx1 || pos.y + size.y < vy0 || pos.y > vy1)
}

pub(crate) fn render_constellation(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    #[inline]
    fn snap_rect_to_pixels(pos: Vec2, size: Vec2) -> (Vec2, Vec2) {
        let snapped_pos = Vec2::new(pos.x.round(), pos.y.round());
        let snapped_br = Vec2::new((pos.x + size.x).round(), (pos.y + size.y).round());
        let snapped_size = snapped_br - snapped_pos;
        (snapped_pos, snapped_size)
    }
    let chat = app.chat_window.as_ref().unwrap();
    let view = &chat.constellation_view;
    // Full-bleed graph: no outer scissor. Sidebar/header/composer draw in later composite passes
    // (see COMPOSITE_DRAW_ORDER in renderer). Per-shard content_clip scissors still apply.
    let graph = &app.graph_state;
    let scale = view.scale_animated;
    if chat.constellation_macro_active() {
        render_constellation_macro(renderer, app, vertices);
        return;
    }
    // Allow text to continue shrinking with zoom so bubbles don't grow taller due to extra wrapping.
    let min_font = style::constellation::MIN_FONT;

    // World-space view center / span for far-node simplification (see `is_far`).
    let view_tl = view.screen_to_world(view.position);
    let view_br = view.screen_to_world(view.position + view.size);
    let view_center = (view_tl + view_br) * 0.5;
    let view_diagonal = (view_br - view_tl).length();
    // Nodes farther than this (in world units) get simplified rendering: backplate + border only, no text layers.
    let far_threshold = view_diagonal * style::constellation::FAR_SIMPLIFY_THRESHOLD;
    // 10.0 was formerly style::elevation::SHARD_PAD; kept as a geometry-only stability margin.
    let geometry_visible_pad = 10.0 * scale + 64.0 + SHARD_GEOMETRY_CULL_STABILITY_PAD_PX;

    let mut node_ids: Vec<String> = graph.nodes.keys().cloned().collect();
    node_ids.sort();

    // 1. Edges (behind nodes): parent bottom-center -> child top-center.
    // Primary edge: first parent, solid full-weight line (PHOSPHOR color).
    // Secondary edges: additional parents on DAG nodes, thinner muted line.
    let edge_thickness = style::stroke::GRAPH_EDGE_PX * scale;
    let secondary_edge_thickness = (style::stroke::GRAPH_EDGE_PX * 0.5).max(1.5) * scale;
    let edge_inset =
        (style::constellation::EDGE_INSET * scale).max(style::constellation::EDGE_INSET_MIN);

    // Build children_of: parent_id -> sorted list of child ids, for bundle x-offset computation.
    // Sorted so the child ordering matches node_ids iteration (deterministic left→right spread).
    let mut children_of: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for id in &node_ids {
        let node = &graph.nodes[id];
        for parent_id in &node.shard.parent_ids {
            children_of.entry(parent_id.clone()).or_default().push(id.clone());
        }
    }

    // Stride between adjacent lines in the bundle: line width + gap so they are tight, never touching.
    let bundle_gap = style::constellation::EDGE_BUNDLE_GAP_PX * scale;
    let line_stride = edge_thickness + bundle_gap;

    for id in &node_ids {
        let node = &graph.nodes[id];
        let child_top_center = node.position + Vec2::new(node.size.x * 0.5, 0.0);
        let c_sp = view.world_to_screen(node.position);
        let c_ss = view.world_size_to_screen(node.size);
        let (c_pos, c_size) = snap_rect_to_pixels(c_sp, c_ss);
        let child_in_view =
            screen_rect_intersects_view(c_pos, c_size, view.position, view.size, geometry_visible_pad);
        let n_parents = node.shard.parent_ids.len();
        for (parent_idx, parent_id) in node.shard.parent_ids.iter().enumerate() {
            if let Some(parent) = graph.get_node(parent_id) {
                let parent_bottom_center =
                    parent.position + Vec2::new(parent.size.x * 0.5, parent.size.y);
                let p_sp = view.world_to_screen(parent.position);
                let p_ss = view.world_size_to_screen(parent.size);
                let (p_pos, p_size) = snap_rect_to_pixels(p_sp, p_ss);
                let parent_in_view = screen_rect_intersects_view(
                    p_pos,
                    p_size,
                    view.position,
                    view.size,
                    geometry_visible_pad,
                );
                if !child_in_view && !parent_in_view {
                    continue;
                }
                let is_primary = parent_idx == 0;
                // Edge color = child's ribbon hue (propagated from the branch root).
                let base_color = style::edge_palette::sample(node.ribbon_hue_t);
                let (thickness, color) = if is_primary {
                    (edge_thickness, base_color)
                } else {
                    // Secondary (cross) edges: thinner and semi-transparent.
                    (secondary_edge_thickness, base_color.truncate().extend(0.55))
                };

                // Bundle x-offsets: spread lines symmetrically around the node center so
                // they travel in parallel near each card, then fan out via the elbow to reach
                // the corresponding anchor on the opposite card.
                //
                // child side: one slot per parent of this child
                let child_offset_x = (parent_idx as f32 - (n_parents as f32 - 1.0) * 0.5)
                    * line_stride;
                // parent side: one slot per child of that parent
                let siblings = children_of.get(parent_id.as_str());
                let n_children = siblings.map(|v| v.len()).unwrap_or(1);
                let child_slot = siblings
                    .and_then(|v| v.iter().position(|c| c == id))
                    .unwrap_or(0);
                let parent_offset_x =
                    (child_slot as f32 - (n_children as f32 - 1.0) * 0.5) * line_stride;

                let mut a = view.world_to_screen(parent_bottom_center);
                let mut b = view.world_to_screen(child_top_center);
                // Apply bundle spread in screen space (world→screen is uniform scale + translate).
                a.x += parent_offset_x;
                b.x += child_offset_x;
                // Nudge both endpoints into the inter-card gap so the connector
                // lands on the border ring rather than visually entering the card.
                a.y += edge_inset;
                b.y -= edge_inset;
                elbow_to_vertices(
                    a, b,
                    style::constellation::EDGE_CORNER_RADIUS * scale,
                    style::stroke::GRAPH_EDGE_STEPS,
                    thickness, color,
                    vertices,
                );
            }
        }
    }

    // 2. Nodes: bbox (fill + border) then content. Skip nodes outside view.
    let mut layout_cache = std::collections::HashMap::new();
    let mut visible_node_count = 0usize;
    let scale_bucket = (scale * style::constellation::SCALE_BUCKET_MULTIPLIER).round() as u32;
    for id in &node_ids {
        let node = &graph.nodes[id];
        let sp = view.world_to_screen(node.position);
        let ss = view.world_size_to_screen(node.size);
        let (screen_pos, screen_size) = snap_rect_to_pixels(sp, ss);
        let corner_radius = style::corner_radius::MEDIUM * scale;

        let in_view = screen_rect_intersects_view(
            screen_pos,
            screen_size,
            view.position,
            view.size,
            geometry_visible_pad,
        );

        let node_center = node.position + node.size * 0.5;
        let dist_from_center = (node_center - view_center).length();
        let is_far = dist_from_center > far_threshold;

        if !in_view {
            continue;
        }
        visible_node_count += 1;

        // Per-bubble 2D scroll in screen space; only text inside bubbles scrolls.
        let scroll_offsets = chat.constellation_scroll_offsets.borrow();
        let bubble_scroll = scroll_offsets
            .get(id)
            .copied()
            .unwrap_or(crate::ui::chat_window::BubbleScroll::default());
        let user_scroll = bubble_scroll.user;
        let assistant_scroll = bubble_scroll.assistant;
        drop(scroll_offsets);

        let padding = style::padding::SMALL * scale;
        let shard_msg_inset = style::padding::SHARD_MESSAGE_INSET * scale;
        let border_inset =
            (style::constellation::EDGE_INSET * scale).max(style::constellation::EDGE_INSET_MIN);

        // Elevation: shards float above the constellation field. Queue one shadow per shard before
        // its backplate fill so every card reads as lifted. Shadow verts go into the renderer's
        // current batch; fills accumulate in the scratch `vertices` that the caller flushes after
        // the loop, so the shadows always land behind the fills in draw order.
        let shard_rect =
            crate::ui::core::Rect::new(screen_pos.x, screen_pos.y, screen_size.x, screen_size.y);
        renderer.queue_shadow(&shard_rect, corner_radius, &style::elevation::LOW());

        // Shard chassis fill (`bg::SHARD_BACKPLATE`); ring via shader border mode (negative corner_radius).
        let fill = Quad {
            position: screen_pos,
            size: screen_size,
            color: style::bg::SHARD_BACKPLATE(),
            corner_radius,
            bubble_effect: false,
            slider_effect: false,
        };
        renderer.add_quad(&fill, None);

        let outer_radius = corner_radius + border_inset;
        let pb = style::border::PHOSPHOR();
        let shard_border = Quad {
            position: screen_pos - Vec2::splat(border_inset),
            size: screen_size + Vec2::splat(border_inset * 2.0),
            color: Vec4::new(pb.x, pb.y, pb.z, border_inset),
            corner_radius: -outer_radius,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&shard_border.to_vertices());

        // Hover highlight: soft tint when node is hovered but not focused
        let is_hovered = chat.hovered_node_id.as_deref() == Some(id.as_str());
        let is_focused = graph.current_leaf_id.as_deref() == Some(id.as_str());
        if is_hovered && !is_focused {
            let ph = style::accent::PHOSPHOR();
            let hover_tint = Vec4::new(ph.x, ph.y, ph.z, 0.08);
            let hover_fill = Quad {
                position: screen_pos,
                size: screen_size,
                color: hover_tint,
                corner_radius,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&hover_fill.to_vertices());
        }

        // Move/resize handles: subtle strip on hover (top = move, bottom-right = resize)
        let handle_tint = Vec4::new(
            style::accent::PHOSPHOR().x,
            style::accent::PHOSPHOR().y,
            style::accent::PHOSPHOR().z,
            0.14,
        );
        const MOVE_HANDLE_H: f32 = 14.0;
        const RESIZE_HANDLE_SZ: f32 = 14.0;
        if is_hovered || is_focused {
            let move_bar = Quad {
                position: screen_pos,
                size: Vec2::new(screen_size.x, MOVE_HANDLE_H * scale),
                color: handle_tint,
                corner_radius: 0.0,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&move_bar.to_vertices());
            let resize_x = screen_pos.x + screen_size.x - RESIZE_HANDLE_SZ * scale;
            let resize_y = screen_pos.y + screen_size.y - RESIZE_HANDLE_SZ * scale;
            let resize_corner = Quad {
                position: Vec2::new(resize_x, resize_y),
                size: Vec2::new(RESIZE_HANDLE_SZ * scale, RESIZE_HANDLE_SZ * scale),
                color: handle_tint,
                corner_radius: 0.0,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&resize_corner.to_vertices());
        }

        // Two-message chat inside node: user bubble (blue, right), assistant bubble (grey, left).
        // Base sizes from update_node_sizes (world); scaled here. Inner width is floored to
        // max_content_width so bubbles match the shard backplate budget even when measure is tight.
        let bubble_spacing = style::constellation::BUBBLE_SPACING * scale;
        // Match the wider measurement used in update_node_sizes so text has more horizontal room at default zoom.
        let max_content_width = (screen_size.x * style::constellation::BUBBLE_MAX_WIDTH_RATIO
            - (padding + shard_msg_inset) * 2.0)
            .max(style::constellation::BUBBLE_MIN_CONTENT_WIDTH * scale);
        let font_size = (style::font_size::MESSAGE_BODY * scale).max(min_font);
        let font_small = (style::font_size::SMALL * scale).max(min_font);
        let user_size = Vec2::new(node.user_text_width * scale, node.user_text_height * scale);
        let assistant_size = Vec2::new(
            node.assistant_text_width * scale,
            node.assistant_text_height * scale,
        );
        // Expand bubble inner width to the shard backplate budget even when graph-side measure
        // is tight (short longest line) or stale; keeps wrap target and grey/blue quads aligned.
        let user_inner_w = user_size.x.max(max_content_width);
        let assistant_inner_w = assistant_size.x.max(max_content_width);
        layout_cache.insert(
            id.clone(),
            (
                Vec2::new(user_inner_w, user_size.y),
                Vec2::new(assistant_inner_w, assistant_size.y),
            ),
        );

        if is_far {
            // Far node: backplate + border only (already drawn above). Skip bubble content.
            continue;
        }

        // Clip inner content to the card rect so overflowing text is visually truncated.
        let content_clip =
            crate::ui::core::Rect::new(screen_pos.x, screen_pos.y, screen_size.x, screen_size.y);
        renderer.push_scissor(&content_clip);

        // Fixed layout: bubbles do not scroll; only text inside each bubble scrolls.
        let mut y = screen_pos.y + padding + shard_msg_inset;
        use crate::ui::core::layout;
        use crate::ui::icons::icon_names;
        let btn_size = style::constellation::ACTION_BUTTON_SIZE * scale;
        let btn_spacing = style::constellation::ACTION_BUTTON_SPACING * scale;
        let row_padding = style::constellation::ACTION_ROW_PADDING * scale;
        let msg_btn = style::constellation::MESSAGE_ACTION_BUTTON_SIZE * scale;
        let icon_size = style::constellation::MESSAGE_ACTION_ICON_SIZE * scale;
        let icon_size_14 = style::constellation::ACTION_ICON_SIZE * scale;
        let button_color = style::text::SECONDARY();

        let button_reserve = style::constellation::BUTTON_ROW_RESERVE * scale;

        // Shard bottom icon row (eye / + / folder / more): match layout used below so text clips above it.
        let action_area_w = screen_size.x - padding * 2.0;
        let (shard_btn_row, shard_row_space) =
            style::constellation::fit_shard_action_row(action_area_w, btn_size, btn_spacing);
        let shard_actions_top = screen_pos.y + screen_size.y - row_padding - shard_btn_row;
        let message_text_clip_bottom =
            shard_actions_top - style::constellation::SHARD_ACTION_TEXT_CLEARANCE * scale;

        // Bottom of visible content area (above shard actions); per-bubble controls stay within this when overflowing
        let content_bottom = message_text_clip_bottom - shard_msg_inset;

        // User bubble: blue, right-aligned; show placeholder when user_visible is false
        if user_size.x > 0.0 && user_size.y > 0.0 {
            let bubble_w = user_inner_w + padding * 2.0;
            let bubble_x = screen_pos.x + screen_size.x - padding - shard_msg_inset - bubble_w;
            let bubble_y = y;

            // Use the text-layer raster height as the single source of truth for text height.
            let text_height = if node.shard.user_visible {
                let component_id = format!("constellation_node_{}_user", id);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(
                    &component_id,
                    Some("chat"),
                    "ConstellationMessageUser",
                );
                let content = node.shard.user_content.as_deref().unwrap();
                let mut hasher = DefaultHasher::new();
                content.hash(&mut hasher);
                let content_hash = hasher.finish();
                let inner_width = bubble_w - padding * 2.0;
                let width_bucket = inner_width.round() as u32;
                let key = (id.clone(), true, content_hash, scale_bucket, width_bucket);
                renderer.set_composite_layer(CompositeLayer::ConstellationText);
                let (layer_w, layer_h) = renderer.get_or_create_text_layer(
                    key.clone(),
                    content,
                    inner_width,
                    font_size,
                    crate::ui::style::text::PRIMARY(),
                );
                // 1:1 blit so font size matches layout; wide bubble only adds wrap width, not texture stretch.
                let dest_w = layer_w as f32;
                let dest_h = layer_h as f32;
                // Right-align when text fits; when overflowing horizontally, anchor left and scroll by user_scroll.x.
                let dest_x =
                    bubble_x + padding + (inner_width - dest_w).max(0.0) - user_scroll.x;
                let bubble_h_pre = dest_h + padding * 2.0 + button_reserve;
                let dest_y = bubble_y + padding - user_scroll.y;
                let (bubble_clip_pos, bubble_clip_size) = snap_rect_to_pixels(
                    Vec2::new(bubble_x, bubble_y),
                    Vec2::new(bubble_w, bubble_h_pre),
                );
                let bubble_clip = Rect::new(
                    bubble_clip_pos.x,
                    bubble_clip_pos.y,
                    bubble_clip_size.x,
                    bubble_clip_size.y,
                );
                let mut text_layer_clip = bubble_clip;
                if text_layer_clip.bottom() > message_text_clip_bottom {
                    text_layer_clip.height = (message_text_clip_bottom - text_layer_clip.y).max(0.0);
                }
                renderer.draw_text_layer(
                    key,
                    (dest_x, dest_y, dest_w, dest_h),
                    0.0,
                    Some(&text_layer_clip),
                );
                renderer.set_composite_layer(CompositeLayer::MainContent);
                renderer.pop_parent();
                dest_h
            } else {
                style::constellation::HIDDEN_PLACEHOLDER_HEIGHT * scale
            };

            let bubble_h = text_height + padding * 2.0 + button_reserve;
            let (bubble_pos, bubble_size) =
                snap_rect_to_pixels(Vec2::new(bubble_x, bubble_y), Vec2::new(bubble_w, bubble_h));
            let user_bg = if node.shard.user_visible && !chat.is_shard_muted(id) {
                style::bg::USER_MESSAGE()
            } else {
                style::bg::MUTED_MESSAGE()
            };
            let user_bubble = Quad {
                position: bubble_pos,
                size: bubble_size,
                color: user_bg,
                corner_radius,
                bubble_effect: true,
                slider_effect: false,
            };
            renderer.add_quad(&user_bubble, Some(&content_clip));
            let user_bubble_rect = crate::ui::core::Rect::new(
                bubble_pos.x,
                bubble_pos.y,
                bubble_size.x,
                bubble_size.y,
            );
            renderer.push_scissor(&user_bubble_rect);

            if !node.shard.user_visible {
                let user_text_start = Vec2::new(bubble_x + padding, bubble_y + padding);
                let mut placeholder = Text::new_for_render("(Message hidden)")
                    .with_font_size(font_small)
                    .with_color(style::text::TERTIARY())
                    .with_alignment(TextAlignment::Left);
                placeholder.update_layout(
                    Rect::new(
                        user_text_start.x,
                        user_text_start.y,
                        max_content_width,
                        text_height,
                    ),
                    None,
                    None,
                );
                renderer.push_parent(format!("constellation_node_{}_user_hidden", id));
                renderer.validate_component(
                    &format!("constellation_node_{}_user_hidden", id),
                    Some("chat"),
                    "ConstellationMessageHidden",
                );
                renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
                placeholder.render(renderer, app, vertices, None);
                renderer.pop_parent();
                renderer.set_composite_layer(CompositeLayer::MainContent);
            }

            renderer.pop_scissor();
            let visible_bottom = (bubble_y + bubble_h).min(content_bottom);
            let edit_pos = layout::center(
                &Rect::new(
                    bubble_x + bubble_w
                        - msg_btn * 2.0
                        - style::constellation::MESSAGE_ACTION_INSET * scale,
                    visible_bottom - msg_btn - style::constellation::MESSAGE_ACTION_INSET * scale,
                    msg_btn,
                    msg_btn,
                ),
                Vec2::splat(icon_size),
            );
            let hide_pos = layout::center(
                &Rect::new(
                    bubble_x + bubble_w
                        - msg_btn
                        - style::constellation::MESSAGE_ACTION_INSET * scale,
                    visible_bottom - msg_btn - style::constellation::MESSAGE_ACTION_INSET * scale,
                    msg_btn,
                    msg_btn,
                ),
                Vec2::splat(icon_size),
            );
            renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
            renderer.queue_icon(icon_names::PENCIL, edit_pos, icon_size, button_color);
            renderer.queue_icon(
                if node.shard.user_visible {
                    icon_names::EYE_CLOSED
                } else {
                    icon_names::EYE_OPEN
                },
                hide_pos,
                icon_size,
                button_color,
            );
            renderer.set_composite_layer(CompositeLayer::MainContent);
            y += bubble_h + bubble_spacing;
        }

        // Assistant bubble: grey, left-aligned
        let is_editing = chat
            .messages
            .iter()
            .position(|m| m.shard_id.as_deref() == Some(id.as_str()))
            .map(|idx| chat.editing_message_idx == Some(idx))
            .unwrap_or(false);
        if is_editing {
            // In-place edit: render edit_textarea within the assistant bubble area
            let bubble_w = (chat.edit_textarea.size.x + padding * 2.0)
                .max(style::constellation::BUBBLE_MIN_CONTENT_WIDTH * scale);
            let bubble_h = chat.edit_textarea.size.y + padding * 2.0;
            let bubble_x = screen_pos.x + padding + shard_msg_inset;
            let bubble_y = y;
            let assistant_bubble = Quad {
                position: Vec2::new(bubble_x, bubble_y),
                size: Vec2::new(bubble_w, bubble_h),
                color: style::bg::ASSISTANT_MESSAGE(),
                corner_radius,
                bubble_effect: true,
                slider_effect: false,
            };
            renderer.add_quad(&assistant_bubble, Some(&content_clip));
            let mut edit_textarea = chat.edit_textarea.clone();
            edit_textarea.position = Vec2::new(bubble_x + padding, bubble_y + padding);
            edit_textarea.size = chat.edit_textarea.size;
            edit_textarea.cursor_visible = app.cursor_visible;
            edit_textarea.cursor_animation_value = app.cursor_position_animation.value;
            renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
            text_input_render::render_text_input(
                renderer,
                &edit_textarea,
                app,
                vertices,
                Some(font_size),
                Some(padding),
                Some(corner_radius),
                false,
            );
            let visible_bottom = (bubble_y + bubble_h).min(content_bottom);
            let edit_pos = layout::center(
                &Rect::new(
                    bubble_x + bubble_w
                        - msg_btn * 2.0
                        - style::constellation::MESSAGE_ACTION_INSET * scale,
                    visible_bottom - msg_btn - style::constellation::MESSAGE_ACTION_INSET * scale,
                    msg_btn,
                    msg_btn,
                ),
                Vec2::splat(icon_size),
            );
            let hide_pos = layout::center(
                &Rect::new(
                    bubble_x + bubble_w
                        - msg_btn
                        - style::constellation::MESSAGE_ACTION_INSET * scale,
                    visible_bottom - msg_btn - style::constellation::MESSAGE_ACTION_INSET * scale,
                    msg_btn,
                    msg_btn,
                ),
                Vec2::splat(icon_size),
            );
            renderer.queue_icon(icon_names::PENCIL, edit_pos, icon_size, button_color);
            renderer.queue_icon(
                if node.shard.assistant_visible {
                    icon_names::EYE_CLOSED
                } else {
                    icon_names::EYE_OPEN
                },
                hide_pos,
                icon_size,
                button_color,
            );
            renderer.set_composite_layer(CompositeLayer::MainContent);
            y += bubble_h + bubble_spacing;
        } else if assistant_size.x > 0.0 && assistant_size.y > 0.0 {
            let citation_line_height =
                (style::font_size::SMALL * style::font_size::LINE_HEIGHT_RATIO) * scale;
            let citation_gap = style::constellation::CITATION_GAP * scale;
            let citations_expanded = chat.constellation_citations_expanded(id);
            let citation_line_count = if node.shard.citations.is_empty() {
                0usize
            } else if citations_expanded {
                node.shard.citations.len()
            } else {
                1usize
            };
            let bubble_w = assistant_inner_w + padding * 2.0;
            let bubble_x = screen_pos.x + padding;
            let bubble_y = y;

            let citations_height = if citation_line_count == 0 {
                0.0
            } else {
                citation_gap + citation_line_count as f32 * citation_line_height
            };

            // Use text-layer raster height for assistant content as source of truth.
            let text_height = if node.shard.assistant_visible {
                let component_id = format!("constellation_node_{}_assistant", id);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(
                    &component_id,
                    Some("chat"),
                    "ConstellationMessageAssistant",
                );
                let content = node.shard.assistant_content.as_deref().unwrap();
                let mut hasher = DefaultHasher::new();
                content.hash(&mut hasher);
                let content_hash = hasher.finish();
                let inner_width = bubble_w - padding * 2.0;
                let width_bucket = inner_width.round() as u32;
                let key = (id.clone(), false, content_hash, scale_bucket, width_bucket);
                renderer.set_composite_layer(CompositeLayer::ConstellationText);
                let (layer_w, layer_h) = renderer.get_or_create_text_layer(
                    key.clone(),
                    content,
                    inner_width,
                    font_size,
                    style::text::PRIMARY(),
                );
                let dest_w = layer_w as f32;
                let dest_h = layer_h as f32;
                let dest_x = bubble_x + padding - assistant_scroll.x;
                let bubble_h_pre = dest_h + padding * 2.0 + citations_height + button_reserve;
                let dest_y = bubble_y + padding - assistant_scroll.y;
                let (bubble_clip_pos, bubble_clip_size) = snap_rect_to_pixels(
                    Vec2::new(bubble_x, bubble_y),
                    Vec2::new(bubble_w, bubble_h_pre),
                );
                let bubble_clip = Rect::new(
                    bubble_clip_pos.x,
                    bubble_clip_pos.y,
                    bubble_clip_size.x,
                    bubble_clip_size.y,
                );
                let mut text_layer_clip = bubble_clip;
                if text_layer_clip.bottom() > message_text_clip_bottom {
                    text_layer_clip.height = (message_text_clip_bottom - text_layer_clip.y).max(0.0);
                }
                renderer.draw_text_layer(
                    key,
                    (dest_x, dest_y, dest_w, dest_h),
                    0.0,
                    Some(&text_layer_clip),
                );
                renderer.set_composite_layer(CompositeLayer::MainContent);
                renderer.pop_parent();
                dest_h
            } else {
                style::constellation::HIDDEN_PLACEHOLDER_HEIGHT * scale
            };

            let bubble_h = text_height + padding * 2.0 + citations_height + button_reserve;
            let (bubble_pos, bubble_size) =
                snap_rect_to_pixels(Vec2::new(bubble_x, bubble_y), Vec2::new(bubble_w, bubble_h));
            let ast_bg = if node.shard.assistant_visible && !chat.is_shard_muted(id) {
                style::bg::ASSISTANT_MESSAGE()
            } else {
                style::bg::MUTED_MESSAGE()
            };
            let assistant_bubble = Quad {
                position: bubble_pos,
                size: bubble_size,
                color: ast_bg,
                corner_radius,
                bubble_effect: true,
                slider_effect: false,
            };
            renderer.add_quad(&assistant_bubble, Some(&content_clip));
            let assistant_bubble_rect = crate::ui::core::Rect::new(
                bubble_pos.x,
                bubble_pos.y,
                bubble_size.x,
                bubble_size.y,
            );
            renderer.push_scissor(&assistant_bubble_rect);
            let assistant_text_start = Vec2::new(bubble_x + padding, bubble_y + padding);
            if !node.shard.assistant_visible {
                let placeholder_height = style::constellation::HIDDEN_PLACEHOLDER_HEIGHT * scale;
                let mut placeholder = Text::new_for_render("(Message hidden)")
                    .with_font_size(font_small)
                    .with_color(style::text::TERTIARY())
                    .with_alignment(TextAlignment::Left);
                placeholder.update_layout(
                    Rect::new(
                        assistant_text_start.x,
                        assistant_text_start.y,
                        max_content_width,
                        placeholder_height,
                    ),
                    None,
                    None,
                );
                renderer.push_parent(format!("constellation_node_{}_assistant_hidden", id));
                renderer.validate_component(
                    &format!("constellation_node_{}_assistant_hidden", id),
                    Some("chat"),
                    "ConstellationMessageHidden",
                );
                renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
                placeholder.render(renderer, app, vertices, None);
                renderer.pop_parent();
                renderer.set_composite_layer(CompositeLayer::MainContent);
            }
            // Citations below assistant content (same style as linear chat expanded citations).
            // Anchored left (no x-scroll): citations stay in place even when text scrolls horizontally.
            let text_start_for_citations =
                Vec2::new(bubble_x + padding, bubble_y + padding - assistant_scroll.y);
            renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
            for citation_idx in 0..citation_line_count {
                let citation_text = if citations_expanded {
                    format_constellation_citation(citation_idx, &node.shard.citations[citation_idx])
                } else {
                    format!("Sources ({})", node.shard.citations.len())
                };
                let citation_y = text_start_for_citations.y
                    + text_height
                    + citation_gap
                    + citation_idx as f32 * citation_line_height;
                let citation_rect = Rect::new(
                    bubble_x + padding,
                    citation_y,
                    bubble_w - padding * 2.0,
                    citation_line_height,
                );
                let mut citation_text_component = Text::new_for_render(&citation_text)
                    .with_font_size(font_small)
                    .with_color(style::text::SECONDARY())
                    .with_alignment(TextAlignment::Left);
                citation_text_component.update_layout(citation_rect, None, None);
                let component_id = format!("constellation_citation_{}_{}", id, citation_idx);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(&component_id, Some("chat"), "ConstellationCitation");
                citation_text_component.render(renderer, app, vertices, None);
                renderer.pop_parent();
            }
            renderer.set_composite_layer(CompositeLayer::MainContent);
            renderer.pop_scissor();
            let visible_bottom = (bubble_y + bubble_h).min(content_bottom);
            let edit_pos = layout::center(
                &Rect::new(
                    bubble_x + bubble_w
                        - msg_btn * 2.0
                        - style::constellation::MESSAGE_ACTION_INSET * scale,
                    visible_bottom - msg_btn - style::constellation::MESSAGE_ACTION_INSET * scale,
                    msg_btn,
                    msg_btn,
                ),
                Vec2::splat(icon_size),
            );
            let hide_pos = layout::center(
                &Rect::new(
                    bubble_x + bubble_w
                        - msg_btn
                        - style::constellation::MESSAGE_ACTION_INSET * scale,
                    visible_bottom - msg_btn - style::constellation::MESSAGE_ACTION_INSET * scale,
                    msg_btn,
                    msg_btn,
                ),
                Vec2::splat(icon_size),
            );
            renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
            renderer.queue_icon(icon_names::PENCIL, edit_pos, icon_size, button_color);
            renderer.queue_icon(
                if node.shard.assistant_visible {
                    icon_names::EYE_CLOSED
                } else {
                    icon_names::EYE_OPEN
                },
                hide_pos,
                icon_size,
                button_color,
            );
            renderer.set_composite_layer(CompositeLayer::MainContent);
            y += bubble_h + bubble_spacing;
        }

        // Notes list (mirroring linear chat)
        let note_line_h = style::constellation::NOTE_LINE_HEIGHT * scale;
        let notes_gap = style::constellation::CITATION_GAP * scale;
        if !node.shard.notes.is_empty() {
            let notes_start_y = y + notes_gap;
            for (note_idx, note) in node.shard.notes.iter().enumerate() {
                let line_y = notes_start_y + note_idx as f32 * note_line_h;
                let note_text = if note.len() > 60 {
                    format!("• {}...", &note[..60])
                } else {
                    format!("• {}", note)
                };
                let mut note_text_component = Text::new_for_render(&note_text)
                    .with_font_size(font_small)
                    .with_color(style::text::SECONDARY())
                    .with_alignment(TextAlignment::Left);
                let note_rect = Rect::new(
                    screen_pos.x + padding + shard_msg_inset,
                    line_y,
                    screen_size.x
                        - (padding + shard_msg_inset) * 2.0
                        - style::constellation::NOTE_TEXT_EXTRA_RIGHT_PAD * scale,
                    note_line_h,
                );
                note_text_component.update_layout(note_rect, None, None);
                let component_id = format!("constellation_note_{}_{}", id, note_idx);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(&component_id, Some("chat"), "ConstellationNote");
                renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
                note_text_component.render(renderer, app, vertices, None);
                renderer.pop_parent();
                let edit_w = style::constellation::NOTE_ICON_WIDTH * scale;
                let edit_rect = Rect::new(
                    screen_pos.x + screen_size.x
                        - padding
                        - style::constellation::NOTE_EDIT_RIGHT_OFFSET * scale,
                    line_y,
                    edit_w,
                    note_line_h,
                );
                let remove_rect = Rect::new(
                    screen_pos.x + screen_size.x
                        - padding
                        - style::constellation::NOTE_REMOVE_RIGHT_OFFSET * scale,
                    line_y,
                    edit_w,
                    note_line_h,
                );
                let edit_icon_pos = layout::center(&edit_rect, Vec2::splat(icon_size));
                let remove_icon_pos = layout::center(&remove_rect, Vec2::splat(icon_size));
                renderer.queue_icon(icon_names::PENCIL, edit_icon_pos, icon_size, button_color);
                renderer.queue_icon(icon_names::TRASH, remove_icon_pos, icon_size, button_color);
                renderer.set_composite_layer(CompositeLayer::MainContent);
            }
            y = notes_start_y + node.shard.notes.len() as f32 * note_line_h;
        }

        // Action row: pin (top-right of content area), hide/note/add context/more (bottom)

        // Pin button: top-right of node (assistant messages only)
        let pin_pos = Vec2::new(
            screen_pos.x + screen_size.x - row_padding - btn_size,
            screen_pos.y + row_padding,
        );
        let is_pinned = app.insights_state.insights.iter().any(|i| i.id == *id);
        let pin_icon = if is_pinned {
            icon_names::PIN_RED
        } else {
            icon_names::PIN
        };
        let pin_rect = Rect::from_pos_size(pin_pos, Vec2::splat(btn_size));
        let icon_pos = layout::center(&pin_rect, Vec2::splat(icon_size_14));
        renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
        renderer.queue_icon(
            pin_icon,
            icon_pos,
            icon_size_14,
            if is_pinned {
                crate::ui::style::graph::PIN_ICON_TINT()
            } else {
                button_color
            },
        );

        // Action buttons at bottom: hide, note, add context, more
        let action_y = shard_actions_top;
        let action_area = Rect::new(
            screen_pos.x + padding,
            action_y,
            action_area_w,
            shard_btn_row,
        );
        let widths = [shard_btn_row, shard_btn_row, shard_btn_row, shard_btn_row];
        let rects = layout::stack_horizontal(&action_area, &widths, shard_row_space, 0.0);
        let icons = [
            icon_names::EYE_CLOSED,
            icon_names::PLUS,
            icon_names::FOLDER,
            icon_names::DOTS_6_VERTICAL,
        ];
        for (i, rect) in rects.iter().enumerate() {
            let icon_pos = layout::center(rect, Vec2::splat(icon_size_14));
            renderer.queue_icon(icons[i], icon_pos, icon_size_14, button_color);
        }
        renderer.set_composite_layer(CompositeLayer::MainContent);

        renderer.pop_scissor();
    }

    if let Some(ref chat) = app.chat_window {
        *chat.constellation_layout_cache.borrow_mut() = Some(layout_cache);
    }
    if app.debug_text_stats {
        renderer.set_debug_constellation_visible_nodes(visible_node_count);
    }

    // Note input popup: below (or above if would overflow) the shard when adding/editing note
    let note_msg_idx = chat
        .adding_note_msg_idx
        .or(chat.editing_note.map(|(m, _)| m));
    if let Some(msg_idx) = note_msg_idx {
        if let Some(shard_id) = chat.messages.get(msg_idx).and_then(|m| m.shard_id.as_ref()) {
            if graph.get_node(shard_id).is_some() {
                let mut note_input = chat.note_input.clone();
                note_input.cursor_visible = app.cursor_visible;
                note_input.cursor_animation_value = app.cursor_position_animation.value;
                let note_font = (style::font_size::MESSAGE_BODY * scale).max(min_font);
                let note_padding = style::padding::SMALL * scale;
                let note_corner = style::corner_radius::MEDIUM * scale;
                renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
                text_input_render::render_text_input(
                    renderer,
                    &note_input,
                    app,
                    vertices,
                    Some(note_font),
                    Some(note_padding),
                    Some(note_corner),
                    false,
                );
                let send_rect = chat.note_send_button_rect;
                let send_bg = Quad {
                    position: send_rect.position(),
                    size: send_rect.size(),
                    color: style::button::PRIMARY(),
                    corner_radius: note_corner,
                    bubble_effect: false,
                    slider_effect: false,
                };
                renderer.set_composite_layer(CompositeLayer::MainContent);
                vertices.extend_from_slice(&send_bg.to_vertices());
                renderer.set_composite_layer(CompositeLayer::ConstellationOverlay);
                let mut send_text = Text::new_for_render("Send")
                    .with_font_size(note_font)
                    .with_color(style::text::PRIMARY())
                    .with_alignment(TextAlignment::Center);
                send_text.update_layout(send_rect, None, None);
                renderer.push_parent("constellation_note_send".to_string());
                renderer.validate_component(
                    "constellation_note_send",
                    Some("chat"),
                    "ConstellationNoteSend",
                );
                send_text.render(renderer, app, vertices, None);
                renderer.pop_parent();
                renderer.set_composite_layer(CompositeLayer::MainContent);
            }
        }
    }
}
