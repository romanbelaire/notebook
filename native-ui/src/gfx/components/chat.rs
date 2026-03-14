use crate::gfx::renderer::Renderer;
use crate::gfx::types::{segment_to_vertices, Quad, Vertex};
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::tab_bar::Tab;
use crate::ui::style;
use crate::ui::core::{Rect, text_input_render};
use crate::ui::{Text, TextAlignment};
use crate::ui::components::Renderable;
use pulldown_cmark::{Event, Parser, Tag};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Render markdown text with basic formatting (bold, italic, code)
/// Returns the final Y position after rendering
fn render_markdown_text(
    renderer: &mut Renderer,
    app: &App,
    markdown: &str,
    start_pos: Vec2,
    max_width: f32,
    font_size: f32,
    text_color: Vec4,
    vertices: &mut Vec<Vertex>,
    component_prefix: &str,
    mut record: Option<&mut Vec<(String, Vec2, f32, Vec4)>>,
) -> f32 {
    let parser = Parser::new(markdown);
    let mut current_x = start_pos.x;
    let mut current_y = start_pos.y;
    let line_height = font_size * 1.2;
    let mut current_text = String::new();
    let mut is_code = false;
    let mut is_bold = false;
    let mut is_italic = false;

    let effective_font_size = |bold: bool, italic: bool, code: bool| -> f32 {
        if code {
            font_size * 0.9
        } else if bold && italic {
            font_size * 1.05
        } else if bold {
            font_size * 1.1
        } else if italic {
            font_size * 0.95
        } else {
            font_size
        }
    };
    
    for event in parser {
        match event {
            Event::Start(Tag::Strong) => {
                if !current_text.is_empty() {
                    let size = effective_font_size(is_bold, is_italic, is_code);
                    render_text_segment(
                        renderer,
                        app,
                        &current_text,
                        Vec2::new(current_x, current_y),
                        size,
                        text_color,
                        is_code,
                        is_bold,
                        is_italic,
                        vertices,
                        component_prefix,
                        start_pos,
                        record.as_deref_mut(),
                    );
                    let text_width = renderer.measure_text(&current_text, size).x;
                    current_x += text_width;
                    current_text.clear();
                }
                is_bold = true;
            }
            Event::End(pulldown_cmark::TagEnd::Strong) => {
                if !current_text.is_empty() {
                    let size = effective_font_size(is_bold, is_italic, is_code);
                    render_text_segment(
                        renderer,
                        app,
                        &current_text,
                        Vec2::new(current_x, current_y),
                        size,
                        text_color,
                        is_code,
                        is_bold,
                        is_italic,
                        vertices,
                        component_prefix,
                        start_pos,
                        record.as_deref_mut(),
                    );
                    let text_width = renderer.measure_text(&current_text, size).x;
                    current_x += text_width;
                    current_text.clear();
                }
                is_bold = false;
            }
            Event::Start(Tag::Emphasis) => {
                if !current_text.is_empty() {
                    let size = effective_font_size(is_bold, is_italic, is_code);
                    render_text_segment(
                        renderer,
                        app,
                        &current_text,
                        Vec2::new(current_x, current_y),
                        size,
                        text_color,
                        is_code,
                        is_bold,
                        is_italic,
                        vertices,
                        component_prefix,
                        start_pos,
                        record.as_deref_mut(),
                    );
                    let text_width = renderer.measure_text(&current_text, size).x;
                    current_x += text_width;
                    current_text.clear();
                }
                is_italic = true;
            }
            Event::End(pulldown_cmark::TagEnd::Emphasis) => {
                if !current_text.is_empty() {
                    let size = effective_font_size(is_bold, is_italic, is_code);
                    render_text_segment(
                        renderer,
                        app,
                        &current_text,
                        Vec2::new(current_x, current_y),
                        size,
                        text_color,
                        is_code,
                        is_bold,
                        is_italic,
                        vertices,
                        component_prefix,
                        start_pos,
                        record.as_deref_mut(),
                    );
                    let text_width = renderer.measure_text(&current_text, size).x;
                    current_x += text_width;
                    current_text.clear();
                }
                is_italic = false;
            }
            Event::Start(Tag::CodeBlock(_)) => {
                if !current_text.is_empty() {
                    let size = effective_font_size(is_bold, is_italic, is_code);
                    render_text_segment(
                        renderer,
                        app,
                        &current_text,
                        Vec2::new(current_x, current_y),
                        size,
                        text_color,
                        is_code,
                        is_bold,
                        is_italic,
                        vertices,
                        component_prefix,
                        start_pos,
                        record.as_deref_mut(),
                    );
                    let text_width = renderer.measure_text(&current_text, size).x;
                    current_x += text_width;
                    current_text.clear();
                }
                is_code = true;
            }
            Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                if !current_text.is_empty() {
                    let size = effective_font_size(is_bold, is_italic, is_code);
                    render_text_segment(
                        renderer,
                        app,
                        &current_text,
                        Vec2::new(current_x, current_y),
                        size,
                        text_color,
                        is_code,
                        is_bold,
                        is_italic,
                        vertices,
                        component_prefix,
                        start_pos,
                        record.as_deref_mut(),
                    );
                    let text_width = renderer.measure_text(&current_text, size).x;
                    current_x += text_width;
                    current_text.clear();
                }
                is_code = false;
            }
            Event::Text(text) => {
                let size = effective_font_size(is_bold, is_italic, is_code);
                let words: Vec<&str> = text.split_whitespace().collect();
                for word in words {
                    let word_with_space = if current_text.is_empty() {
                        word.to_string()
                    } else {
                        format!(" {}", word)
                    };
                    let test_text = format!("{}{}", current_text, word_with_space);
                    let test_width = renderer.measure_text(&test_text, size).x;
                    if test_width > max_width && !current_text.is_empty() {
                        render_text_segment(
                            renderer,
                            app,
                            &current_text,
                            Vec2::new(start_pos.x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            vertices,
                            component_prefix,
                            start_pos,
                            record.as_deref_mut(),
                        );
                        current_y += line_height;
                        current_x = start_pos.x;
                        current_text = word.to_string();
                    } else {
                        current_text = test_text;
                    }
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if !current_text.is_empty() {
                    let size = effective_font_size(is_bold, is_italic, is_code);
                    render_text_segment(
                        renderer,
                        app,
                        &current_text,
                        Vec2::new(start_pos.x, current_y),
                        size,
                        text_color,
                        is_code,
                        is_bold,
                        is_italic,
                        vertices,
                        component_prefix,
                        start_pos,
                        record.as_deref_mut(),
                    );
                    current_text.clear();
                }
                current_y += line_height;
                current_x = start_pos.x;
            }
            _ => {}
        }
    }
    if !current_text.is_empty() {
        let size = effective_font_size(is_bold, is_italic, is_code);
        render_text_segment(
            renderer,
            app,
            &current_text,
            Vec2::new(current_x, current_y),
            size,
            text_color,
            is_code,
            is_bold,
            is_italic,
            vertices,
            component_prefix,
            start_pos,
            record.as_deref_mut(),
        );
    }
    
    current_y + line_height
}

/// Render a text segment with optional formatting (code, bold, italic).
/// font_size is already the effective size (scaled for bold/italic by the caller).
fn render_text_segment(
    renderer: &mut Renderer,
    app: &App,
    text: &str,
    position: Vec2,
    font_size: f32,
    base_color: Vec4,
    is_code: bool,
    _is_bold: bool,
    _is_italic: bool,
    vertices: &mut Vec<Vertex>,
    component_prefix: &str,
    start_pos: Vec2,
    record: Option<&mut Vec<(String, Vec2, f32, Vec4)>>,
) {
    let color = if is_code {
        // Slightly warmer code text to match beige body text
        Vec4::new(0.86, 0.84, 0.86, 1.0)
    } else {
        base_color
    };
    let text_hash =
        format!("{:x}", text.chars().take(20).map(|c| c as u32).sum::<u32>());
    let component_id = format!(
        "{}_text_{}_{}_{}",
        component_prefix,
        text_hash,
        position.x as u32,
        position.y as u32
    );
    if !renderer.validate_component(&component_id, None, "MarkdownText") {
        return;
    }
    if let Some(rec) = record {
        rec.push((
            text.to_string(),
            position - start_pos,
            font_size,
            color,
        ));
    }
    renderer.queue_text(text, position, color, font_size);
}

const NODE_BORDER: Vec4 = Vec4::new(1.0, 1.0, 1.0, 0.9);
const EDGE_THICKNESS: f32 = 2.0;

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
fn measure_wrapped_block(measure: &mut impl FnMut(&str, f32) -> Vec2, text: &str, max_width: f32, font_size: f32) -> Vec2 {
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

/// World-space AABB (min, max) intersects rect (min, max).
fn world_rect_intersects(
    pos: Vec2,
    size: Vec2,
    view_min: Vec2,
    view_max: Vec2,
) -> bool {
    !(pos.x + size.x < view_min.x
        || pos.x > view_max.x
        || pos.y + size.y < view_min.y
        || pos.y > view_max.y)
}

fn render_constellation(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    #[inline]
    fn snap_rect_to_pixels(pos: Vec2, size: Vec2) -> (Vec2, Vec2) {
        let snapped_pos = Vec2::new(pos.x.round(), pos.y.round());
        let snapped_br = Vec2::new((pos.x + size.x).round(), (pos.y + size.y).round());
        let snapped_size = snapped_br - snapped_pos;
        (snapped_pos, snapped_size)
    }
    let chat = app.chat_window.as_ref().unwrap();
    let view = &chat.constellation_view;
    let graph = &app.graph_state;
    let scale = view.scale_animated;
    // Allow text to continue shrinking with zoom so bubbles don't grow taller due to extra wrapping.
    const MIN_FONT: f32 = 4.0;

    // Viewport culling: world rect (with margin) so we only draw nodes and edges in or near view.
    let view_tl = view.screen_to_world(view.position);
    let view_br = view.screen_to_world(view.position + view.size);
    let margin = view.size / scale;
    let view_min = view_tl - margin;
    let view_max = view_br + margin;
    let view_center = (view_tl + view_br) * 0.5;
    let view_diagonal = (view_br - view_tl).length();
    /// Nodes farther than this (in world units) get simplified rendering: backplate + border only, no text layers.
    const FAR_SIMPLIFY_THRESHOLD: f32 = 2.5;
    let far_threshold = view_diagonal * FAR_SIMPLIFY_THRESHOLD;

    // 1. Edges (behind nodes): child -> each parent. Only draw if at least one endpoint is in view.
    let edge_thickness = EDGE_THICKNESS * scale;
    for (id, node) in &graph.nodes {
        let child_center = node.position + node.size * 0.5;
        let child_in_view = world_rect_intersects(node.position, node.size, view_min, view_max);
        for parent_id in &node.shard.parent_ids {
            if let Some(parent) = graph.get_node(parent_id) {
                let parent_center = parent.position + parent.size * 0.5;
                let parent_in_view = world_rect_intersects(parent.position, parent.size, view_min, view_max);
                if !child_in_view && !parent_in_view {
                    continue;
                }
                let a = view.world_to_screen(child_center);
                let b = view.world_to_screen(parent_center);
                segment_to_vertices(a, b, edge_thickness, NODE_BORDER, vertices);
            }
        }
    }

    // 2. Nodes: bbox (fill + border) then content. Skip nodes outside view.
    let mut layout_cache = std::collections::HashMap::new();
    let mut visible_node_count = 0usize;
    let scale_bucket = (scale * 4.0).round() as u32;
    for (id, node) in &graph.nodes {
        if !world_rect_intersects(node.position, node.size, view_min, view_max) {
            continue;
        }
        visible_node_count += 1;
        let (screen_pos, screen_size) = {
            let sp = view.world_to_screen(node.position);
            let ss = view.world_size_to_screen(node.size);
            snap_rect_to_pixels(sp, ss)
        };

        // Phase 3: Distance-based simplification: far nodes get backplate + border only, no text layers.
        let node_center = node.position + node.size * 0.5;
        let dist_from_center = (node_center - view_center).length();
        let is_far = dist_from_center > far_threshold;

        // Per-bubble scroll (user, assistant) in screen space; only text inside bubbles scrolls.
        let scroll_offsets = chat.constellation_scroll_offsets.borrow();
        let (user_scroll, assistant_scroll) = scroll_offsets.get(id).copied().unwrap_or((0.0, 0.0));
        drop(scroll_offsets);

        let padding = style::padding::SMALL * scale;
        let corner_radius = style::corner_radius::MEDIUM * scale;
        let border_inset = (1.0f32 * scale).max(0.5);

        // Transparent gray backplate for shard (message pair)
        let fill = Quad {
            position: screen_pos,
            size: screen_size,
            color: style::bg::SHARD_BACKPLATE,
            corner_radius,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&fill.to_vertices());

        // Border: outline (draw larger quad first, then fill on top)
        let border_quad = Quad {
            position: screen_pos - Vec2::splat(border_inset),
            size: screen_size + Vec2::splat(border_inset * 2.0),
            color: NODE_BORDER,
            corner_radius: corner_radius + border_inset,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&border_quad.to_vertices());
        let fill_on_top = Quad {
            position: screen_pos,
            size: screen_size,
            color: style::bg::SHARD_BACKPLATE,
            corner_radius,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&fill_on_top.to_vertices());

        // Two-message chat inside node: user bubble (blue, right), assistant bubble (grey, left).
        // Sizes from update_node_sizes (single source of truth); scale to screen space to avoid re-measure.
        let bubble_spacing = 6.0f32 * scale;
        // Match the wider measurement used in update_node_sizes so text has more horizontal room at default zoom.
        let max_content_width = (screen_size.x * 0.8 - padding * 2.0).max(80.0 * scale);
        let font_size = (style::font_size::NORMAL * scale).max(MIN_FONT);
        let font_small = (style::font_size::SMALL * scale).max(MIN_FONT);
        let user_size = Vec2::new(node.user_text_width * scale, node.user_text_height * scale);
        let assistant_size = Vec2::new(node.assistant_text_width * scale, node.assistant_text_height * scale);
        layout_cache.insert(id.clone(), (user_size, assistant_size));

        if is_far {
            // Far node: backplate + border only (already drawn above). Skip bubble content.
            continue;
        }

        // Clip inner content to the card rect so overflowing text is visually truncated.
        let content_clip = crate::ui::core::Rect::new(
            screen_pos.x,
            screen_pos.y,
            screen_size.x,
            screen_size.y,
        );
        renderer.push_scissor(&content_clip);

        // Fixed layout: bubbles do not scroll; only text inside each bubble scrolls.
        let mut y = screen_pos.y + padding;
        use crate::ui::icons::icon_names;
        use crate::ui::core::layout;
        let btn_size = 18.0f32 * scale;
        let btn_spacing = 4.0f32 * scale;
        let row_padding = 6.0f32 * scale;
        let msg_btn = 14.0f32 * scale;
        let icon_size = 12.0f32 * scale;
        let icon_size_14 = 14.0f32 * scale;
        let button_color = style::text::SECONDARY;

        const BUTTON_ROW_RESERVE: f32 = 22.0;
        let button_reserve = BUTTON_ROW_RESERVE * scale;

        // User bubble: blue, right-aligned; show placeholder when user_visible is false
        if user_size.x > 0.0 && user_size.y > 0.0 {
            let bubble_w = user_size.x + padding * 2.0;
            let bubble_x = screen_pos.x + screen_size.x - padding - bubble_w;
            let bubble_y = y;

            // Use the text-layer raster height as the single source of truth for text height.
            let text_height = if node.shard.user_visible {
                let component_id = format!("constellation_node_{}_user", id);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(&component_id, Some("chat"), "ConstellationMessageUser");
                let content = node.shard.user_content.as_deref().unwrap();
                let mut hasher = DefaultHasher::new();
                content.hash(&mut hasher);
                let content_hash = hasher.finish();
                let inner_width = bubble_w - padding * 2.0;
                let width_bucket = inner_width.round() as u32;
                let key = (id.clone(), true, content_hash, scale_bucket, width_bucket);
                let (layer_w, layer_h) = renderer.get_or_create_text_layer(
                    key.clone(),
                    content,
                    inner_width,
                    font_size,
                    Vec4::new(1.0, 1.0, 1.0, 1.0),
                );
                let dest_x = bubble_x + padding;
                let dest_y = bubble_y + padding - user_scroll;
                let dest_w = inner_width;
                let scale_y = dest_w / layer_w as f32;
                let dest_h = layer_h as f32 * scale_y;
                renderer.draw_text_layer(
                    key,
                    (dest_x, dest_y, dest_w, dest_h),
                    0.0,
                );
                renderer.pop_parent();
                dest_h
            } else {
                20.0f32 * scale
            };

            let bubble_h = text_height + padding * 2.0 + button_reserve;
            let (bubble_pos, bubble_size) =
                snap_rect_to_pixels(Vec2::new(bubble_x, bubble_y), Vec2::new(bubble_w, bubble_h));
            let user_bg = if node.shard.user_visible {
                style::bg::USER_MESSAGE
            } else {
                style::bg::MUTED_MESSAGE
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
            let user_bubble_rect =
                crate::ui::core::Rect::new(bubble_pos.x, bubble_pos.y, bubble_size.x, bubble_size.y);
            renderer.push_scissor(&user_bubble_rect);

            if !node.shard.user_visible {
                let user_text_start = Vec2::new(bubble_x + padding, bubble_y + padding);
                let mut placeholder = Text::new_for_render("(Message hidden)")
                    .with_font_size(font_small)
                    .with_color(style::text::TERTIARY)
                    .with_alignment(TextAlignment::Left);
                placeholder.update_layout(
                    Rect::new(user_text_start.x, user_text_start.y, max_content_width, text_height),
                    None,
                    None,
                );
                renderer.push_parent(format!("constellation_node_{}_user_hidden", id));
                renderer.validate_component(
                    &format!("constellation_node_{}_user_hidden", id),
                    Some("chat"),
                    "ConstellationMessageHidden",
                );
                placeholder.render(renderer, app, vertices, None);
                renderer.pop_parent();
            }

            renderer.pop_scissor();
            let edit_pos = layout::center(
                &Rect::new(
                    bubble_x + bubble_w - msg_btn * 2.0 - 4.0 * scale,
                    bubble_y + bubble_h - msg_btn - 4.0 * scale,
                    msg_btn,
                    msg_btn,
                ),
                Vec2::splat(icon_size),
            );
            let hide_pos = layout::center(
                &Rect::new(
                    bubble_x + bubble_w - msg_btn - 4.0 * scale,
                    bubble_y + bubble_h - msg_btn - 4.0 * scale,
                    msg_btn,
                    msg_btn,
                ),
                Vec2::splat(icon_size),
            );
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
            y += bubble_h + bubble_spacing;
        }

        // Assistant bubble: grey, left-aligned
        let is_editing = chat.messages.iter().position(|m| m.shard_id.as_deref() == Some(id.as_str()))
            .map(|idx| chat.editing_message_idx == Some(idx))
            .unwrap_or(false);
        if is_editing {
            // In-place edit: render edit_textarea within the assistant bubble area
            let bubble_w = (chat.edit_textarea.size.x + padding * 2.0).max(80.0 * scale);
            let bubble_h = chat.edit_textarea.size.y + padding * 2.0;
            let bubble_x = screen_pos.x + padding;
            let bubble_y = y;
            let assistant_bubble = Quad {
                position: Vec2::new(bubble_x, bubble_y),
                size: Vec2::new(bubble_w, bubble_h),
                color: style::bg::ASSISTANT_MESSAGE,
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
            let edit_pos = layout::center(&Rect::new(bubble_x + bubble_w - msg_btn * 2.0 - 4.0 * scale, bubble_y + bubble_h - msg_btn - 4.0 * scale, msg_btn, msg_btn), Vec2::splat(icon_size));
            let hide_pos = layout::center(&Rect::new(bubble_x + bubble_w - msg_btn - 4.0 * scale, bubble_y + bubble_h - msg_btn - 4.0 * scale, msg_btn, msg_btn), Vec2::splat(icon_size));
            renderer.queue_icon(icon_names::PENCIL, edit_pos, icon_size, button_color);
            renderer.queue_icon(if node.shard.assistant_visible { icon_names::EYE_CLOSED } else { icon_names::EYE_OPEN }, hide_pos, icon_size, button_color);
            y += bubble_h + bubble_spacing;
        } else if assistant_size.x > 0.0 && assistant_size.y > 0.0 {
            let citation_line_height = (style::font_size::SMALL * 1.2) * scale;
            let citation_gap = 4.0f32 * scale;
            let bubble_w = assistant_size.x + padding * 2.0;
            let bubble_x = screen_pos.x + padding;
            let bubble_y = y;

            // Use text-layer raster height for assistant content as source of truth.
            let text_height = if node.shard.assistant_visible {
                let component_id = format!("constellation_node_{}_assistant", id);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(&component_id, Some("chat"), "ConstellationMessageAssistant");
                let content = node.shard.assistant_content.as_deref().unwrap();
                let mut hasher = DefaultHasher::new();
                content.hash(&mut hasher);
                let content_hash = hasher.finish();
                let inner_width = bubble_w - padding * 2.0;
                let width_bucket = inner_width.round() as u32;
                let key = (id.clone(), false, content_hash, scale_bucket, width_bucket);
                let (layer_w, layer_h) = renderer.get_or_create_text_layer(
                    key.clone(),
                    content,
                    inner_width,
                    font_size,
                    style::text::PRIMARY,
                );
                let dest_x = bubble_x + padding;
                let dest_y = bubble_y + padding - assistant_scroll;
                let dest_w = inner_width;
                let scale_y = dest_w / layer_w as f32;
                let dest_h = layer_h as f32 * scale_y;
                renderer.draw_text_layer(
                    key,
                    (dest_x, dest_y, dest_w, dest_h),
                    0.0,
                );
                renderer.pop_parent();
                dest_h
            } else {
                20.0f32 * scale
            };

            let citations_height = if node.shard.citations.is_empty() {
                0.0
            } else {
                citation_gap + node.shard.citations.len() as f32 * citation_line_height
            };
            let bubble_h = text_height + padding * 2.0 + citations_height + button_reserve;
            let (bubble_pos, bubble_size) =
                snap_rect_to_pixels(Vec2::new(bubble_x, bubble_y), Vec2::new(bubble_w, bubble_h));
            let ast_bg = if node.shard.assistant_visible {
                style::bg::ASSISTANT_MESSAGE
            } else {
                style::bg::MUTED_MESSAGE
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
            let assistant_bubble_rect =
                crate::ui::core::Rect::new(bubble_pos.x, bubble_pos.y, bubble_size.x, bubble_size.y);
            renderer.push_scissor(&assistant_bubble_rect);
            let assistant_text_start = Vec2::new(bubble_x + padding, bubble_y + padding);
            if !node.shard.assistant_visible {
                let placeholder_height = 20.0f32 * scale;
                let mut placeholder = Text::new_for_render("(Message hidden)")
                    .with_font_size(font_small)
                    .with_color(style::text::TERTIARY)
                    .with_alignment(TextAlignment::Left);
                placeholder.update_layout(Rect::new(assistant_text_start.x, assistant_text_start.y, max_content_width, placeholder_height), None, None);
                renderer.push_parent(format!("constellation_node_{}_assistant_hidden", id));
                renderer.validate_component(&format!("constellation_node_{}_assistant_hidden", id), Some("chat"), "ConstellationMessageHidden");
                placeholder.render(renderer, app, vertices, None);
                renderer.pop_parent();
            }
            // Citations below assistant content (same style as linear chat expanded citations)
            let text_start_for_citations =
                Vec2::new(bubble_x + padding, bubble_y + padding - assistant_scroll);
            for (citation_idx, citation_value) in node.shard.citations.iter().enumerate() {
                let citation_text = format_constellation_citation(citation_idx, citation_value);
                let citation_y = text_start_for_citations.y
                    + text_height
                    + citation_gap
                    + citation_idx as f32 * citation_line_height;
                let citation_rect = Rect::new(bubble_x + padding, citation_y, bubble_w - padding * 2.0, citation_line_height);
                let mut citation_text_component = Text::new_for_render(&citation_text)
                    .with_font_size(font_small)
                    .with_color(style::text::SECONDARY)
                    .with_alignment(TextAlignment::Left);
                citation_text_component.update_layout(citation_rect, None, None);
                let component_id = format!("constellation_citation_{}_{}", id, citation_idx);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(&component_id, Some("chat"), "ConstellationCitation");
                citation_text_component.render(renderer, app, vertices, None);
                renderer.pop_parent();
            }
            renderer.pop_scissor();
            let edit_pos = layout::center(&Rect::new(bubble_x + bubble_w - msg_btn * 2.0 - 4.0 * scale, bubble_y + bubble_h - msg_btn - 4.0 * scale, msg_btn, msg_btn), Vec2::splat(icon_size));
            let hide_pos = layout::center(&Rect::new(bubble_x + bubble_w - msg_btn - 4.0 * scale, bubble_y + bubble_h - msg_btn - 4.0 * scale, msg_btn, msg_btn), Vec2::splat(icon_size));
            renderer.queue_icon(icon_names::PENCIL, edit_pos, icon_size, button_color);
            renderer.queue_icon(if node.shard.assistant_visible { icon_names::EYE_CLOSED } else { icon_names::EYE_OPEN }, hide_pos, icon_size, button_color);
            y += bubble_h + bubble_spacing;
        }

        // Notes list (mirroring linear chat)
        let note_line_h = 18.0f32 * scale;
        let notes_gap = 4.0f32 * scale;
        if !node.shard.notes.is_empty() {
            let notes_start_y = y + notes_gap;
            for (note_idx, note) in node.shard.notes.iter().enumerate() {
                let line_y = notes_start_y + note_idx as f32 * note_line_h;
                let note_text = if note.len() > 60 { format!("• {}...", &note[..60]) } else { format!("• {}", note) };
                let mut note_text_component = Text::new_for_render(&note_text)
                    .with_font_size(font_small)
                    .with_color(style::text::SECONDARY)
                    .with_alignment(TextAlignment::Left);
                let note_rect = Rect::new(screen_pos.x + padding, line_y, screen_size.x - padding * 2.0 - 52.0 * scale, note_line_h);
                note_text_component.update_layout(note_rect, None, None);
                let component_id = format!("constellation_note_{}_{}", id, note_idx);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(&component_id, Some("chat"), "ConstellationNote");
                note_text_component.render(renderer, app, vertices, None);
                renderer.pop_parent();
                let edit_w = 20.0f32 * scale;
                let edit_rect = Rect::new(screen_pos.x + screen_size.x - padding - 48.0 * scale, line_y, edit_w, note_line_h);
                let remove_rect = Rect::new(screen_pos.x + screen_size.x - padding - 24.0 * scale, line_y, edit_w, note_line_h);
                let edit_icon_pos = layout::center(&edit_rect, Vec2::splat(icon_size));
                let remove_icon_pos = layout::center(&remove_rect, Vec2::splat(icon_size));
                renderer.queue_icon(icon_names::PENCIL, edit_icon_pos, icon_size, button_color);
                renderer.queue_icon(icon_names::TRASH, remove_icon_pos, icon_size, button_color);
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
        let pin_icon = if is_pinned { icon_names::PIN_RED } else { icon_names::PIN };
        let pin_rect = Rect::from_pos_size(pin_pos, Vec2::splat(btn_size));
        let icon_pos = layout::center(&pin_rect, Vec2::splat(icon_size_14));
        renderer.queue_icon(
            pin_icon,
            icon_pos,
            icon_size_14,
            if is_pinned { Vec4::new(0.9, 0.3, 0.3, 1.0) } else { button_color },
        );

        // Action buttons at bottom: hide, note, add context, more
        let action_y = screen_pos.y + screen_size.y - row_padding - btn_size;
        let action_area = Rect::new(
            screen_pos.x + padding,
            action_y,
            screen_size.x - padding * 2.0,
            btn_size,
        );
        let widths = [btn_size, btn_size, btn_size, btn_size];
        let rects = layout::stack_horizontal(&action_area, &widths, btn_spacing, 0.0);
        let icons = [icon_names::EYE_CLOSED, icon_names::PLUS, icon_names::FOLDER, icon_names::DOTS_6_VERTICAL];
        for (i, rect) in rects.iter().enumerate() {
            let icon_pos = layout::center(rect, Vec2::splat(icon_size_14));
            renderer.queue_icon(icons[i], icon_pos, icon_size_14, button_color);
        }

        renderer.pop_scissor();
    }

    if let Some(ref chat) = app.chat_window {
        *chat.constellation_layout_cache.borrow_mut() = Some(layout_cache);
    }
    if app.debug_text_stats {
        renderer.set_debug_constellation_visible_nodes(visible_node_count);
    }

    // Note input popup: below (or above if would overflow) the shard when adding/editing note
    let note_msg_idx = chat.adding_note_msg_idx.or(chat.editing_note.map(|(m, _)| m));
    if let Some(msg_idx) = note_msg_idx {
        if let Some(shard_id) = chat.messages.get(msg_idx).and_then(|m| m.shard_id.as_ref()) {
            if graph.get_node(shard_id).is_some() {
                let mut note_input = chat.note_input.clone();
                note_input.cursor_visible = app.cursor_visible;
                note_input.cursor_animation_value = app.cursor_position_animation.value;
                let note_font = (style::font_size::NORMAL * scale).max(MIN_FONT);
                let note_padding = style::padding::SMALL * scale;
                let note_corner = style::corner_radius::MEDIUM * scale;
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
            }
        }
    }
}

pub fn render_chat_window(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if let Some(ref chat) = app.chat_window {
        if app.ui_state.active_tab == Tab::Chat {
            renderer.push_parent("chat".to_string());

            // Constellation view (graph) or linear message list
            if app.graph_state.graph_id.is_some() {
                render_constellation(renderer, app, vertices);
            } else {
            // Linear message list
            let bubbles = chat.get_message_bubbles(|text, size| renderer.measure_text(text, size));
            let messages_start_y = chat.message_list.position.y;
            let messages_height = chat.message_list.size.y;
            
            for bubble in &bubbles {
                // Skip if bubble is outside visible area
                if bubble.position.y + bubble.size.y < messages_start_y {
                    continue;
                }
                if bubble.position.y > messages_start_y + messages_height {
                    break;
                }
                
                // Determine bubble background color based on role and muted state
                let bubble_bg_color = if bubble.is_muted {
                    style::bg::MUTED_MESSAGE
                } else {
                    match bubble.role {
                        crate::ui::chat_window::MessageRole::User => style::bg::USER_MESSAGE,
                        crate::ui::chat_window::MessageRole::Assistant => style::bg::ASSISTANT_MESSAGE,
                    }
                };
                
                // Render bubble background with rounded corners (bubble effect = velocity-driven border)
                let bubble_bg = Quad {
                    position: bubble.position,
                    size: bubble.size,
                    color: bubble_bg_color,
                    corner_radius: style::corner_radius::MEDIUM,
                    bubble_effect: true,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&bubble_bg.to_vertices());
                
                // Render message content
                if bubble.is_muted {
                    // Render muted message placeholder
                    let muted_text_rect = Rect::new(
                        bubble.position.x + style::padding::MEDIUM,
                        bubble.position.y + style::padding::MEDIUM,
                        bubble.size.x - style::padding::MEDIUM * 2.0,
                        bubble.size.y - style::padding::MEDIUM * 2.0,
                    );
                    
                    let mut muted_text = Text::new_for_render("(Message hidden)")
                        .with_font_size(style::font_size::SMALL)
                        .with_color(style::text::TERTIARY)
                        .with_alignment(TextAlignment::Left);
                    muted_text.update_layout(muted_text_rect, None, None);
                    
                    let component_id = format!("chat_message_muted_{}", bubble.message_idx);
                    renderer.push_parent(component_id.clone());
                    renderer.validate_component(&component_id, Some("chat"), "ChatMessageMuted");
                    muted_text.render(renderer, app, vertices, None);
                    renderer.pop_parent();
                } else {
                    // Check if this message is being edited
                    if chat.editing_message_idx == Some(bubble.message_idx) {
                        let edit_max_width = bubble.size.x - style::padding::MEDIUM * 2.0;
                        const EDIT_FONT_SIZE: f32 = style::font_size::NORMAL;
                        let edit_line_height = EDIT_FONT_SIZE * 1.2;
                        let mut line_count = 0u32;
                        let words: Vec<&str> = chat.edit_textarea.text.split_whitespace().collect();
                        let mut current_line = String::new();
                        for word in words {
                            let test_line = if current_line.is_empty() {
                                word.to_string()
                            } else {
                                format!("{} {}", current_line, word)
                            };
                            let test_width = renderer.measure_text(&test_line, EDIT_FONT_SIZE).x;
                            if test_width > edit_max_width && !current_line.is_empty() {
                                line_count += 1;
                                current_line = word.to_string();
                            } else {
                                current_line = test_line;
                            }
                        }
                        if !current_line.is_empty() {
                            line_count += 1;
                        }
                        let line_count = line_count.max(1).min(20);
                        let edit_height = (line_count as f32 * edit_line_height) + style::padding::MEDIUM * 2.0;
                        let edit_height = edit_height.clamp(edit_line_height * 2.0 + style::padding::MEDIUM * 2.0, 400.0);
                        let edit_rect = Rect::new(
                            bubble.position.x + style::padding::MEDIUM,
                            bubble.text_start_y,
                            edit_max_width,
                            edit_height,
                        );
                        let mut edit_textarea = chat.edit_textarea.clone();
                        edit_textarea.position = edit_rect.position();
                        edit_textarea.size = edit_rect.size();
                        
                        // Render the edit textarea
                        text_input_render::render_text_input(
                            renderer,
                            &edit_textarea,
                            app,
                            vertices,
                            Some(style::font_size::NORMAL),
                            Some(style::padding::MEDIUM),
                            Some(style::corner_radius::MEDIUM),
                            false,
                        );
                    } else {
                        // Render actual message content with markdown support (always when not editing message)
                        let content_rect = Rect::new(
                            bubble.position.x + style::padding::MEDIUM,
                            bubble.text_start_y,
                            bubble.size.x - style::padding::MEDIUM * 2.0,
                            bubble.size.y - style::padding::MEDIUM * 2.0,
                        );
                        
                        let max_width = content_rect.width;
                        const FONT_SIZE: f32 = style::font_size::NORMAL;
                        
                        let text_color = match bubble.role {
                            crate::ui::chat_window::MessageRole::User => style::text::PRIMARY,
                            crate::ui::chat_window::MessageRole::Assistant => style::text::PRIMARY,
                        };
                        
                        let start_pos = Vec2::new(content_rect.x, content_rect.y);
                        let component_prefix = format!("chat_msg_{}", bubble.message_idx);
                        
                        // Only render if visible
                        if content_rect.y < messages_start_y + messages_height && 
                           content_rect.y + content_rect.height >= messages_start_y {
                        render_markdown_text(
                            renderer,
                            app,
                            &bubble.content,
                            start_pos,
                            max_width,
                            FONT_SIZE,
                            text_color,
                            vertices,
                            &component_prefix,
                            None,
                        );
                        }
                        
                        // If add/edit note is open for this message, also render the note input below the bubble
                        if chat.adding_note_msg_idx == Some(bubble.message_idx) || chat.editing_note.map(|(m, _)| m) == Some(bubble.message_idx) {
                            let note_input_width = (bubble.size.x - style::padding::MEDIUM * 2.0).max(200.0);
                            let note_rect = Rect::new(
                                bubble.position.x + style::padding::MEDIUM,
                                bubble.position.y + bubble.size.y + 4.0,
                                note_input_width,
                                chat.note_input.size.y,
                            );
                            let mut note_input = chat.note_input.clone();
                            note_input.position = note_rect.position();
                            note_input.size = Vec2::new(note_rect.width, note_rect.height);
                            note_input.cursor_visible = app.cursor_visible;
                            note_input.cursor_animation_value = app.cursor_position_animation.value;
                            text_input_render::render_text_input(
                                renderer,
                                &note_input,
                                app,
                                vertices,
                                Some(style::font_size::NORMAL),
                                Some(style::padding::SMALL),
                                Some(style::corner_radius::MEDIUM),
                                false,
                            );
                        }
                    }
                }
                
                // Render citations if present
                if let Some(ref msg) = bubble.message {
                    if !msg.citations.is_empty() {
                        let is_expanded = chat.citations_expanded.contains(&bubble.message_idx);
                        
                        if is_expanded {
                            // Render all citations with details
                            for (citation_pos, citation_size, citation_idx) in &bubble.citation_positions {
                                if citation_pos.y >= messages_start_y && citation_pos.y < messages_start_y + messages_height {
                                    if let Some(citation) = msg.citations.get(*citation_idx) {
                                        // Format citation text
                                        let mut citation_text = format!("[{}] ", citation_idx + 1);
                                        if let Some(ref title) = citation.title {
                                            citation_text.push_str(title);
                                        }
                                        citation_text.push_str(" (");
                                        citation_text.push_str(&citation.source);
                                        if let Some(ref year) = citation.year {
                                            citation_text.push_str(", ");
                                            citation_text.push_str(year);
                                        }
                                        citation_text.push(')');
                                        if let Some(ref section) = citation.section {
                                            citation_text.push_str(" – ");
                                            citation_text.push_str(section);
                                        }
                                        if let Some(page) = citation.page {
                                            citation_text.push_str(&format!(", p.{}", page));
                                        }
                                        
                                        // Render citation text
                                        let citation_text_rect = Rect::new(
                                            citation_pos.x,
                                            citation_pos.y,
                                            citation_size.x - 25.0, // Leave space for icon
                                            citation_size.y,
                                        );
                                        
                                        let mut citation_text_component = Text::new_for_render(&citation_text)
                                            .with_font_size(style::font_size::SMALL)
                                            .with_color(style::text::SECONDARY)
                                            .with_alignment(TextAlignment::Left);
                                        citation_text_component.update_layout(citation_text_rect, None, None);
                                        
                                        let component_id = format!("chat_citation_{}_{}", bubble.message_idx, citation_idx);
                                        renderer.push_parent(component_id.clone());
                                        renderer.validate_component(&component_id, Some("chat"), "ChatCitation");
                                        citation_text_component.render(renderer, app, vertices, None);
                                        renderer.pop_parent();
                                        
                                        // Render magnify icon
                                        use crate::ui::icons::icon_names;
                                        let icon_pos = Vec2::new(
                                            citation_pos.x + citation_size.x - 20.0,
                                            citation_pos.y + citation_size.y / 2.0 - 7.0,
                                        );
                                        renderer.queue_icon(
                                            icon_names::MAGNIFY,
                                            icon_pos,
                                            14.0,
                                            style::text::SECONDARY,
                                        );
                                    }
                                }
                            }
                        } else {
                            // Render collapsed "Sources" summary
                            if let Some((citation_pos, citation_size, _)) = bubble.citation_positions.first() {
                                if citation_pos.y >= messages_start_y && citation_pos.y < messages_start_y + messages_height {
                                    let citation_text = format!("Sources ({})", msg.citations.len());
                                    
                                    let citation_text_rect = Rect::new(
                                        citation_pos.x,
                                        citation_pos.y,
                                        citation_size.x - 25.0,
                                        citation_size.y,
                                    );
                                    
                                    let mut citation_text_component = Text::new_for_render(&citation_text)
                                        .with_font_size(style::font_size::SMALL)
                                        .with_color(style::text::SECONDARY)
                                        .with_alignment(TextAlignment::Left);
                                    citation_text_component.update_layout(citation_text_rect, None, None);
                                    
                                    let component_id = format!("chat_citation_summary_{}", bubble.message_idx);
                                    renderer.push_parent(component_id.clone());
                                    renderer.validate_component(&component_id, Some("chat"), "ChatCitationSummary");
                                    citation_text_component.render(renderer, app, vertices, None);
                                    renderer.pop_parent();
                                }
                            }
                        }
                    }
                }
                
                // Render action buttons (edit, delete, mute)
                use crate::ui::icons::icon_names;
                let button_icon_size = 14.0;
                let button_color = style::text::SECONDARY;
                
                // Render action buttons using layout functions for icon centering
                use crate::ui::core::layout;
                
                if let Some(edit_pos) = bubble.edit_button_position {
                    if edit_pos.y >= messages_start_y && edit_pos.y < messages_start_y + messages_height {
                        let button_rect = Rect::from_pos_size(edit_pos, bubble.action_button_size);
                        let icon_pos = layout::center(&button_rect, Vec2::new(button_icon_size, button_icon_size));
                        renderer.queue_icon(
                            icon_names::PENCIL,
                            icon_pos,
                            button_icon_size,
                            button_color,
                        );
                    }
                }
                
                if let Some(add_note_pos) = bubble.add_note_button_position {
                    if add_note_pos.y >= messages_start_y && add_note_pos.y < messages_start_y + messages_height {
                        let button_rect = Rect::from_pos_size(add_note_pos, bubble.action_button_size);
                        let icon_pos = layout::center(&button_rect, Vec2::new(button_icon_size, button_icon_size));
                        renderer.queue_icon(
                            icon_names::PLUS,
                            icon_pos,
                            button_icon_size,
                            button_color,
                        );
                    }
                }
                
                if let Some(delete_pos) = bubble.delete_button_position {
                    if delete_pos.y >= messages_start_y && delete_pos.y < messages_start_y + messages_height {
                        let button_rect = Rect::from_pos_size(delete_pos, bubble.action_button_size);
                        let icon_pos = layout::center(&button_rect, Vec2::new(button_icon_size, button_icon_size));
                        renderer.queue_icon(
                            icon_names::TRASH,
                            icon_pos,
                            button_icon_size,
                            button_color,
                        );
                    }
                }
                
                if let Some(mute_pos) = bubble.mute_button_position {
                    if mute_pos.y >= messages_start_y && mute_pos.y < messages_start_y + messages_height {
                        let button_rect = Rect::from_pos_size(mute_pos, bubble.action_button_size);
                        let icon_pos = layout::center(&button_rect, Vec2::new(button_icon_size, button_icon_size));
                        let icon_name = if bubble.is_muted {
                            icon_names::EYE_OPEN
                        } else {
                            icon_names::EYE_CLOSED
                        };
                        renderer.queue_icon(
                            icon_name,
                            icon_pos,
                            button_icon_size,
                            button_color,
                        );
                    }
                }
                
                // Render note edit and remove buttons (pencil and × per note)
                for (note_pos, note_size, _) in &bubble.note_edit_positions {
                    if note_pos.y >= messages_start_y && note_pos.y < messages_start_y + messages_height {
                        let note_rect = Rect::from_pos_size(*note_pos, *note_size);
                        let icon_pos = layout::center(&note_rect, Vec2::new(button_icon_size * 0.8, button_icon_size * 0.8));
                        renderer.queue_icon(
                            icon_names::PENCIL,
                            icon_pos,
                            button_icon_size * 0.8,
                            button_color,
                        );
                    }
                }
                for (note_pos, note_size, _) in &bubble.note_remove_positions {
                    if note_pos.y >= messages_start_y && note_pos.y < messages_start_y + messages_height {
                        let note_rect = Rect::from_pos_size(*note_pos, *note_size);
                        let icon_pos = layout::center(&note_rect, Vec2::new(button_icon_size * 0.8, button_icon_size * 0.8));
                        renderer.queue_icon(
                            icon_names::CLOSE,
                            icon_pos,
                            button_icon_size * 0.8,
                            button_color,
                        );
                    }
                }
                
                // Render pin button (for assistant messages only)
                if let Some(pin_pos) = bubble.pin_button_position {
                    if pin_pos.y >= messages_start_y && pin_pos.y < messages_start_y + messages_height {
                        // Check if message is already pinned
                        let is_pinned = if let Some(ref msg) = bubble.message {
                            app.insights_state.insights.iter()
                                .any(|insight| insight.text == msg.content)
                        } else {
                            false
                        };
                        
                        let pin_icon = if is_pinned {
                            icon_names::PIN_RED
                        } else {
                            icon_names::PIN
                        };
                        
                        // Pin button background (subtle, only visible on hover)
                        let pin_button_rect = Rect::from_pos_size(pin_pos, bubble.pin_button_size);
                        // Note: Background is optional, can be added if needed for hover state
                        
                        // Use layout functions for icon centering
                        use crate::ui::core::layout;
                        let pin_button_rect = Rect::from_pos_size(pin_pos, bubble.pin_button_size);
                        let icon_pos = layout::center(&pin_button_rect, Vec2::new(button_icon_size, button_icon_size));
                        renderer.queue_icon(
                            pin_icon,
                            icon_pos,
                            button_icon_size,
                            if is_pinned {
                                Vec4::new(0.9, 0.3, 0.3, 1.0) // Red tint for pinned
                            } else {
                                style::text::SECONDARY
                            },
                        );
                    }
                }
            }
            } // end else (linear message list)
            
            // Render "Generating..." indicator if sending (both graph and linear)
            if chat.is_sending {
                use crate::ui::core::layout;
                
                // Position indicator at bottom of message list with padding
                let message_list_rect = Rect::from_pos_size(chat.message_list.position, chat.message_list.size);
                let generating_height = 40.0;
                let generating_width = 150.0;
                let generating_padding = style::padding::MEDIUM;
                
                // Position at bottom of message list
                let generating_y = layout::align_bottom(&message_list_rect, generating_height, generating_padding);
                let generating_bubble_rect = Rect::new(
                    message_list_rect.x + generating_padding,
                    generating_y,
                    generating_width,
                    generating_height,
                );
                
                let generating_bubble = Quad {
                    position: generating_bubble_rect.position(),
                    size: generating_bubble_rect.size(),
                    color: style::bg::ASSISTANT_MESSAGE,
                    corner_radius: style::corner_radius::MEDIUM,
                    bubble_effect: true,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&generating_bubble.to_vertices());
                
                // Text rect with padding
                let generating_text_rect = generating_bubble_rect.inset(generating_padding);
                
                let mut generating_text = Text::new_for_render("Generating...")
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(style::text::PRIMARY)
                    .with_alignment(TextAlignment::Left);
                generating_text.update_layout(generating_text_rect, None, None);
                
                renderer.push_parent("chat_generating".to_string());
                renderer.validate_component("chat_generating", Some("chat"), "ChatGenerating");
                generating_text.render(renderer, app, vertices, None);
                renderer.pop_parent();
            }
            
            // Now render input bar and dropdown ON TOP of messages
            // Use unified text input rendering system
            // Update cursor state from App (like data/settings tabs do)
            let mut input_field = chat.input_field.clone();
            input_field.cursor_visible = app.cursor_visible;
            input_field.cursor_animation_value = app.cursor_position_animation.value;
            
            // Context pool button (book icon)
            let context_pool_rect = Rect::from_pos_size(chat.context_pool_button_position, chat.context_pool_button_size);
            let context_pool_bg = Quad {
                position: context_pool_rect.position(),
                size: context_pool_rect.size(),
                color: style::button::SECONDARY,
                corner_radius: style::corner_radius::MEDIUM,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&context_pool_bg.to_vertices());
            
            // Book icon
            use crate::ui::icons::icon_names;
            let icon_size = 20.0;
            let icon_pos = Vec2::new(
                context_pool_rect.x + context_pool_rect.width / 2.0 - icon_size / 2.0,
                context_pool_rect.y + context_pool_rect.height / 2.0 - icon_size / 2.0,
            );
            renderer.queue_icon(
                icon_names::BOOK,
                icon_pos,
                icon_size,
                style::text::PRIMARY,
            );
            
            // Render dropdown menu using component system
            use crate::ui::components::Renderable;
            renderer.push_parent("chat_context_pool_dropdown".to_string());
            renderer.validate_component("chat_context_pool_dropdown", Some("chat"), "ContextPoolDropdown");
            chat.context_pool_dropdown.render(renderer, app, vertices, None);
            renderer.pop_parent();
            
            // Render input field using unified component system
            text_input_render::render_text_input(
                renderer,
                &input_field,
                app,
                vertices,
                Some(style::font_size::NORMAL), // Use same font size as before
                Some(style::padding::MEDIUM),  // Use same padding as before
                Some(style::corner_radius::MEDIUM), // Use same corner radius as before
                false,
            );

            // Send button
            let send_rect = Rect::from_pos_size(chat.send_button_position, chat.send_button_size);
            let send_bg = Quad {
                position: send_rect.position(),
                size: send_rect.size(),
                color: style::button::PRIMARY,
                corner_radius: style::corner_radius::MEDIUM,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&send_bg.to_vertices());
            
            // "Send" button text - use Text component
            renderer.push_parent("chat_send_button".to_string());
            renderer.validate_component("chat_send_button", Some("chat"), "SendButton");
            let mut send_text = Text::new_for_render("Send")
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY)
                .with_alignment(TextAlignment::Center);
            send_text.update_layout(send_rect, None, None);
            send_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
            
            // Pop chat parent
            renderer.pop_parent();
        }
    }
}
