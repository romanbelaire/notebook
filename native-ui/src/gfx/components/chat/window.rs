use crate::app::App;
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::gfx::text_layout::{
    measure_default_brush, ParagraphCacheKey, PARAGRAPH_MEASURE_BRUSH_BITS,
};
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::{text_input_render, Rect};
use crate::ui::style;
use crate::ui::tab_bar::Tab;
use crate::ui::{Text, TextAlignment};
use glam::{Vec2, Vec4};

use super::constellation::render_constellation;
use super::markdown::render_markdown_text;

pub fn render_chat_window(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if let Some(ref chat) = app.chat_window {
        if app.ui_state.active_tab == Tab::Chat {
            renderer.push_parent("chat".to_string());

            // Constellation view (graph) or linear message list
            if app.graph_state.constellation_view_active() {
                render_constellation(renderer, app, vertices);
                // Constellation tooltip overlay
                if let Some(ref tooltip) = app.constellation_tooltip {
                    const TOOLTIP_PAD: f32 = 6.0;
                    const TOOLTIP_FONT: f32 = 12.0;
                    let (lines, pos, max_w) = match tooltip {
                        crate::app::ConstellationTooltip::Simple { text, position } => {
                            (vec![text.clone()], *position, 280.0f32)
                        }
                        crate::app::ConstellationTooltip::MacroNode {
                            title,
                            plus_line,
                            meta_line,
                            position,
                        } => (
                            vec![title.clone(), plus_line.clone(), meta_line.clone()],
                            *position,
                            style::constellation::MACRO_TOOLTIP_WIDTH,
                        ),
                    };
                    let joined = lines.join("\n");
                    let mut tooltip_text = Text::new_for_render(joined.as_str())
                        .with_font_size(TOOLTIP_FONT)
                        .with_color(style::text::PRIMARY())
                        .with_alignment(crate::ui::TextAlignment::Left);
                    let measure = renderer.measure_text(joined.as_str(), TOOLTIP_FONT);
                    let tw = measure.x.min(max_w) + TOOLTIP_PAD * 2.0;
                    let th = (measure.y.max(TOOLTIP_FONT * lines.len() as f32 * 1.15)) + TOOLTIP_PAD * 2.0;
                    let tx = (pos.x + 12.0).min(app.viewport_size.x - tw - 4.0).max(4.0);
                    let ty = (pos.y - th - 8.0).max(4.0);
                    tooltip_text.update_layout(
                        Rect::new(
                            tx + TOOLTIP_PAD,
                            ty + TOOLTIP_PAD,
                            tw - TOOLTIP_PAD * 2.0,
                            th - TOOLTIP_PAD * 2.0,
                        ),
                        None,
                        None,
                    );
                    let tooltip_bg = Quad {
                        position: Vec2::new(tx, ty),
                        size: Vec2::new(tw, th),
                        color: style::bg::PANEL_POPUP(),
                        corner_radius: 4.0,
                        bubble_effect: false,
                        slider_effect: false,
                    };
                    renderer.set_composite_layer(CompositeLayer::HudChrome);
                    renderer.add_quad(&tooltip_bg, None);
                    renderer.push_parent("constellation_tooltip".to_string());
                    renderer.validate_component(
                        "constellation_tooltip",
                        Some("chat"),
                        "ConstellationTooltip",
                    );
                    tooltip_text.render(renderer, app, vertices, None);
                    renderer.pop_parent();
                    renderer.set_composite_layer(CompositeLayer::MainContent);
                }
                // Constellation context menu (right-click). Use HudChrome so labels are not covered
                // by MainContent + ConstellationText composite order (see COMPOSITE_DRAW_ORDER in renderer).
                if let Some(ref menu) = app.constellation_context_menu {
                    use crate::app::ConstellationContextMenu;
                    const MENU_WIDTH: f32 = 180.0;
                    const ITEM_H: f32 = 28.0;
                    const MENU_PAD: f32 = 4.0;
                    let (position, items): (Vec2, &[&str]) = match menu {
                        ConstellationContextMenu::Shard { position, .. } => (
                            *position,
                            &[
                                "Hide",
                                "Pin to insights",
                                "Center",
                                "Open in modal",
                                "Copy message",
                                "Mute",
                            ],
                        ),
                        ConstellationContextMenu::Background { position } => {
                            (*position, &["Reset view", "Layout options"])
                        }
                    };
                    let n = items.len();
                    let menu_h = n as f32 * ITEM_H + MENU_PAD * 2.0;
                    let menu_w = MENU_WIDTH + MENU_PAD * 2.0;
                    let mx = position.x.min(app.viewport_size.x - menu_w - 4.0).max(4.0);
                    let my = position.y.min(app.viewport_size.y - menu_h - 4.0).max(4.0);
                    let menu_bg = Quad {
                        position: Vec2::new(mx, my),
                        size: Vec2::new(menu_w, menu_h),
                        color: style::bg::PANEL_POPUP(),
                        corner_radius: 6.0,
                        bubble_effect: false,
                        slider_effect: false,
                    };
                    renderer.set_composite_layer(CompositeLayer::HudChrome);
                    renderer.add_quad(&menu_bg, None);
                    let font_size = style::font_size::TOOLTIP;
                    for (i, label) in items.iter().enumerate() {
                        let item_y = my + MENU_PAD + i as f32 * ITEM_H;
                        let mut item_text = Text::new_for_render(*label)
                            .with_font_size(font_size)
                            .with_color(style::text::PRIMARY())
                            .with_alignment(crate::ui::TextAlignment::Left);
                        item_text.update_layout(
                            Rect::new(mx + MENU_PAD, item_y, MENU_WIDTH, ITEM_H),
                            None,
                            None,
                        );
                        renderer.push_parent(format!("constellation_context_menu_item_{}", i));
                        renderer.validate_component(
                            &format!("constellation_context_menu_{}", i),
                            Some("chat"),
                            "ConstellationContextMenuItem",
                        );
                        item_text.render(renderer, app, vertices, None);
                        renderer.pop_parent();
                    }
                    renderer.set_composite_layer(CompositeLayer::MainContent);
                }
            } else {
                // Linear message list (geometry from [`ChatWindow::refresh_linear_message_layout`], run before root render)
                let messages_start_y = chat.message_list.position.y;
                let messages_height = chat.message_list.size.y;

                for bubble in &chat.linear_message_bubbles_cache {
                    // Skip if bubble is outside visible area
                    if bubble.position.y + bubble.size.y < messages_start_y {
                        continue;
                    }
                    if bubble.position.y > messages_start_y + messages_height {
                        break;
                    }

                    // Determine bubble background color based on role and muted state
                    let bubble_bg_color = if bubble.is_muted {
                        style::bg::MUTED_MESSAGE()
                    } else {
                        match bubble.role {
                            crate::ui::chat_window::MessageRole::User => style::bg::USER_MESSAGE(),
                            crate::ui::chat_window::MessageRole::Assistant => {
                                style::bg::ASSISTANT_MESSAGE()
                            }
                        }
                    };

                    let rim_inset = style::stroke::BUBBLE_RIM_PX;
                    let bubble_rim = Quad {
                        position: bubble.position - Vec2::splat(rim_inset),
                        size: bubble.size + Vec2::splat(rim_inset * 2.0),
                        color: style::border::ACCENT(),
                        corner_radius: style::corner_radius::MEDIUM + rim_inset,
                        bubble_effect: false,
                        slider_effect: false,
                    };
                    vertices.extend_from_slice(&bubble_rim.to_vertices());

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
                            .with_color(style::text::TERTIARY())
                            .with_alignment(TextAlignment::Left);
                        muted_text.update_layout(muted_text_rect, None, None);

                        let component_id = format!("chat_message_muted_{}", bubble.message_idx);
                        renderer.push_parent(component_id.clone());
                        renderer.validate_component(
                            &component_id,
                            Some("chat"),
                            "ChatMessageMuted",
                        );
                        muted_text.render(renderer, app, vertices, None);
                        renderer.pop_parent();
                    } else {
                        // Check if this message is being edited
                        if chat.editing_message_idx == Some(bubble.message_idx) {
                            let edit_max_width = bubble.size.x - style::padding::MEDIUM * 2.0;
                            const EDIT_FONT_SIZE: f32 = style::font_size::MESSAGE_BODY;
                            let edit_line_height =
                                EDIT_FONT_SIZE * style::font_size::LINE_HEIGHT_RATIO;
                            let edit_key = ParagraphCacheKey::new(
                                &chat.edit_textarea.text,
                                EDIT_FONT_SIZE,
                                Some(edit_max_width),
                                PARAGRAPH_MEASURE_BRUSH_BITS,
                                false,
                                false,
                            );
                            let line_count = renderer
                                .paragraph_cache_get_or_insert(
                                    edit_key,
                                    &chat.edit_textarea.text,
                                    EDIT_FONT_SIZE,
                                    Some(edit_max_width),
                                    measure_default_brush(),
                                )
                                .layout
                                .len()
                                .max(1)
                                .min(20) as u32;
                            let edit_height = (line_count as f32 * edit_line_height)
                                + style::padding::MEDIUM * 2.0;
                            let edit_height = edit_height.clamp(
                                edit_line_height * 2.0 + style::padding::MEDIUM * 2.0,
                                400.0,
                            );
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
                                Some(style::font_size::MESSAGE_BODY),
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
                            const FONT_SIZE: f32 = style::font_size::MESSAGE_BODY;

                            let text_color = match bubble.role {
                                crate::ui::chat_window::MessageRole::User => style::text::PRIMARY(),
                                crate::ui::chat_window::MessageRole::Assistant => {
                                    style::text::PRIMARY()
                                }
                            };

                            let start_pos = Vec2::new(content_rect.x, content_rect.y);
                            let component_prefix = format!("chat_msg_{}", bubble.message_idx);

                            // Only render if visible
                            if content_rect.y < messages_start_y + messages_height
                                && content_rect.y + content_rect.height >= messages_start_y
                            {
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
                            if chat.adding_note_msg_idx == Some(bubble.message_idx)
                                || chat.editing_note.map(|(m, _)| m) == Some(bubble.message_idx)
                            {
                                let note_input_width =
                                    (bubble.size.x - style::padding::MEDIUM * 2.0).max(200.0);
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
                                note_input.cursor_animation_value =
                                    app.cursor_position_animation.value;
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
                                for (citation_pos, citation_size, citation_idx) in
                                    &bubble.citation_positions
                                {
                                    if citation_pos.y >= messages_start_y
                                        && citation_pos.y < messages_start_y + messages_height
                                    {
                                        if let Some(citation) = msg.citations.get(*citation_idx) {
                                            // Format citation text
                                            let mut citation_text =
                                                format!("[{}] ", citation_idx + 1);
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

                                            let mut citation_text_component =
                                                Text::new_for_render(&citation_text)
                                                    .with_font_size(style::font_size::SMALL)
                                                    .with_color(style::text::SECONDARY())
                                                    .with_alignment(TextAlignment::Left);
                                            citation_text_component.update_layout(
                                                citation_text_rect,
                                                None,
                                                None,
                                            );

                                            let component_id = format!(
                                                "chat_citation_{}_{}",
                                                bubble.message_idx, citation_idx
                                            );
                                            renderer.push_parent(component_id.clone());
                                            renderer.validate_component(
                                                &component_id,
                                                Some("chat"),
                                                "ChatCitation",
                                            );
                                            citation_text_component
                                                .render(renderer, app, vertices, None);
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
                                                style::text::SECONDARY(),
                                            );
                                        }
                                    }
                                }
                            } else {
                                // Render collapsed "Sources" summary
                                if let Some((citation_pos, citation_size, _)) =
                                    bubble.citation_positions.first()
                                {
                                    if citation_pos.y >= messages_start_y
                                        && citation_pos.y < messages_start_y + messages_height
                                    {
                                        let citation_text =
                                            format!("Sources ({})", msg.citations.len());

                                        let citation_text_rect = Rect::new(
                                            citation_pos.x,
                                            citation_pos.y,
                                            citation_size.x - 25.0,
                                            citation_size.y,
                                        );

                                        let mut citation_text_component =
                                            Text::new_for_render(&citation_text)
                                                .with_font_size(style::font_size::SMALL)
                                                .with_color(style::text::SECONDARY())
                                                .with_alignment(TextAlignment::Left);
                                        citation_text_component.update_layout(
                                            citation_text_rect,
                                            None,
                                            None,
                                        );

                                        let component_id =
                                            format!("chat_citation_summary_{}", bubble.message_idx);
                                        renderer.push_parent(component_id.clone());
                                        renderer.validate_component(
                                            &component_id,
                                            Some("chat"),
                                            "ChatCitationSummary",
                                        );
                                        citation_text_component
                                            .render(renderer, app, vertices, None);
                                        renderer.pop_parent();
                                    }
                                }
                            }
                        }
                    }

                    // Render action buttons (edit, delete, mute)
                    use crate::ui::icons::icon_names;
                    let button_icon_size = 14.0;
                    let button_color = style::text::SECONDARY();

                    // Render action buttons using layout functions for icon centering
                    use crate::ui::core::layout;

                    if let Some(edit_pos) = bubble.edit_button_position {
                        if edit_pos.y >= messages_start_y
                            && edit_pos.y < messages_start_y + messages_height
                        {
                            let button_rect =
                                Rect::from_pos_size(edit_pos, bubble.action_button_size);
                            let icon_pos = layout::center(
                                &button_rect,
                                Vec2::new(button_icon_size, button_icon_size),
                            );
                            renderer.queue_icon(
                                icon_names::PENCIL,
                                icon_pos,
                                button_icon_size,
                                button_color,
                            );
                        }
                    }

                    if let Some(add_note_pos) = bubble.add_note_button_position {
                        if add_note_pos.y >= messages_start_y
                            && add_note_pos.y < messages_start_y + messages_height
                        {
                            let button_rect =
                                Rect::from_pos_size(add_note_pos, bubble.action_button_size);
                            let icon_pos = layout::center(
                                &button_rect,
                                Vec2::new(button_icon_size, button_icon_size),
                            );
                            renderer.queue_icon(
                                icon_names::PLUS,
                                icon_pos,
                                button_icon_size,
                                button_color,
                            );
                        }
                    }

                    if let Some(delete_pos) = bubble.delete_button_position {
                        if delete_pos.y >= messages_start_y
                            && delete_pos.y < messages_start_y + messages_height
                        {
                            let button_rect =
                                Rect::from_pos_size(delete_pos, bubble.action_button_size);
                            let icon_pos = layout::center(
                                &button_rect,
                                Vec2::new(button_icon_size, button_icon_size),
                            );
                            renderer.queue_icon(
                                icon_names::TRASH,
                                icon_pos,
                                button_icon_size,
                                button_color,
                            );
                        }
                    }

                    if let Some(mute_pos) = bubble.mute_button_position {
                        if mute_pos.y >= messages_start_y
                            && mute_pos.y < messages_start_y + messages_height
                        {
                            let button_rect =
                                Rect::from_pos_size(mute_pos, bubble.action_button_size);
                            let icon_pos = layout::center(
                                &button_rect,
                                Vec2::new(button_icon_size, button_icon_size),
                            );
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
                        if note_pos.y >= messages_start_y
                            && note_pos.y < messages_start_y + messages_height
                        {
                            let note_rect = Rect::from_pos_size(*note_pos, *note_size);
                            let icon_pos = layout::center(
                                &note_rect,
                                Vec2::new(button_icon_size * 0.8, button_icon_size * 0.8),
                            );
                            renderer.queue_icon(
                                icon_names::PENCIL,
                                icon_pos,
                                button_icon_size * 0.8,
                                button_color,
                            );
                        }
                    }
                    for (note_pos, note_size, _) in &bubble.note_remove_positions {
                        if note_pos.y >= messages_start_y
                            && note_pos.y < messages_start_y + messages_height
                        {
                            let note_rect = Rect::from_pos_size(*note_pos, *note_size);
                            let icon_pos = layout::center(
                                &note_rect,
                                Vec2::new(button_icon_size * 0.8, button_icon_size * 0.8),
                            );
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
                        if pin_pos.y >= messages_start_y
                            && pin_pos.y < messages_start_y + messages_height
                        {
                            // Check if message is already pinned
                            let is_pinned = if let Some(ref msg) = bubble.message {
                                app.insights_state
                                    .insights
                                    .iter()
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
                            let pin_button_rect =
                                Rect::from_pos_size(pin_pos, bubble.pin_button_size);
                            // Note: Background is optional, can be added if needed for hover state

                            // Use layout functions for icon centering
                            use crate::ui::core::layout;
                            let pin_button_rect =
                                Rect::from_pos_size(pin_pos, bubble.pin_button_size);
                            let icon_pos = layout::center(
                                &pin_button_rect,
                                Vec2::new(button_icon_size, button_icon_size),
                            );
                            renderer.queue_icon(
                                pin_icon,
                                icon_pos,
                                button_icon_size,
                                if is_pinned {
                                    crate::ui::style::graph::PIN_ICON_TINT()
                                } else {
                                    style::text::SECONDARY()
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
                let message_list_rect =
                    Rect::from_pos_size(chat.message_list.position, chat.message_list.size);
                let generating_height = 40.0;
                let generating_width = 150.0;
                let generating_padding = style::padding::MEDIUM;

                // Position at bottom of message list
                let generating_y =
                    layout::align_bottom(&message_list_rect, generating_height, generating_padding);
                let generating_bubble_rect = Rect::new(
                    message_list_rect.x + generating_padding,
                    generating_y,
                    generating_width,
                    generating_height,
                );

                let generating_bubble = Quad {
                    position: generating_bubble_rect.position(),
                    size: generating_bubble_rect.size(),
                    color: style::bg::ASSISTANT_MESSAGE(),
                    corner_radius: style::corner_radius::MEDIUM,
                    bubble_effect: true,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&generating_bubble.to_vertices());

                // Text rect with padding
                let generating_text_rect = generating_bubble_rect.inset(generating_padding);

                let mut generating_text = Text::new_for_render("Generating...")
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(style::text::PRIMARY())
                    .with_alignment(TextAlignment::Left);
                generating_text.update_layout(generating_text_rect, None, None);

                renderer.push_parent("chat_generating".to_string());
                renderer.validate_component("chat_generating", Some("chat"), "ChatGenerating");
                generating_text.render(renderer, app, vertices, None);
                renderer.pop_parent();
            }

            // Split MainContent quads from composer chrome: Root flushes per child, but Chat mixes both
            // layers in one render; flush messages/constellation quads before tagging composer strip + input.
            if !vertices.is_empty() {
                renderer.add_vertices(vertices, None);
                vertices.clear();
            }
            renderer.set_composite_layer(CompositeLayer::ComposerChrome);
            let composer_sep_y = if app.graph_state.constellation_view_active() {
                chat.constellation_view.position.y + chat.constellation_view.size.y
            } else {
                chat.composer_top_y
            };
            if chat.composer_block_height > 0.0 {
                let chassis_pad = style::padding::SMALL;
                let chassis_margin = style::padding::MEDIUM;
                let plate_left = chat.position.x + chassis_margin - chassis_pad;
                let plate_w = chat.size.x - 2.0 * chassis_margin + 2.0 * chassis_pad;
                let plate_top = chat.composer_top_y - chassis_pad;
                let plate_h = chat.composer_block_height + 2.0 * chassis_pad;
                // Elevation: composer chassis floats above the chat surface. Queue shadow before
                // the chassis fill so it lands behind in the same ComposerChrome batch.
                let composer_rect = crate::ui::core::Rect::new(plate_left, plate_top, plate_w, plate_h);
                renderer.queue_shadow(
                    &composer_rect,
                    style::corner_radius::LARGE,
                    &style::elevation::MEDIUM(),
                );
                let composer_chassis = Quad {
                    position: Vec2::new(plate_left, plate_top),
                    size: Vec2::new(plate_w, plate_h),
                    color: style::chrome::COMPOSER_BACKPLATE(),
                    corner_radius: style::corner_radius::LARGE,
                    bubble_effect: false,
                    slider_effect: false,
                };
                renderer.add_quad(&composer_chassis, None);
            }

            if app.graph_state.constellation_view_active() {
                let top = composer_sep_y;
                let bottom = chat.position.y + chat.size.y;
                let h = bottom - top;
                if h > 0.0 {
                    let strip = Quad {
                        position: Vec2::new(chat.position.x, top),
                        size: Vec2::new(chat.size.x, h),
                        color: style::bg::SECONDARY(),
                        corner_radius: 0.0,
                        bubble_effect: false,
                        slider_effect: false,
                    };
                    renderer.add_quad(&strip, None);
                }
            }

            // ComposerChrome: strip already drawn above. Queue filled geometry; HudChrome gets Vello/icons/inputs.
            let mut input_field = chat.input_field.clone();
            input_field.cursor_visible = app.cursor_visible;
            input_field.cursor_animation_value = app.cursor_position_animation.value;

            let context_pool_rect = Rect::from_pos_size(
                chat.context_pool_button_position,
                chat.context_pool_button_size,
            );
            let context_pool_bg = Quad {
                position: context_pool_rect.position(),
                size: context_pool_rect.size(),
                color: style::button::SECONDARY(),
                corner_radius: style::corner_radius::MEDIUM,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&context_pool_bg.to_vertices());

            for item in chat.composer_pill_items.iter() {
                let chip = Quad {
                    position: item.body_rect.position(),
                    size: Vec2::new(
                        item.body_rect.width + item.close_rect.width,
                        item.body_rect.height,
                    ),
                    color: style::bg::TERTIARY(),
                    corner_radius: style::corner_radius::SMALL,
                    bubble_effect: false,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&chip.to_vertices());
                let close_bg = Quad {
                    position: item.close_rect.position(),
                    size: item.close_rect.size(),
                    color: style::button::SECONDARY(),
                    corner_radius: style::corner_radius::SMALL,
                    bubble_effect: false,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&close_bg.to_vertices());
            }

            if chat.mention_popup_open {
                if let Some(rect) = chat.mention_popup_rect() {
                    let bg = Quad {
                        position: rect.position(),
                        size: rect.size(),
                        color: style::bg::PANEL_POPUP(),
                        corner_radius: style::corner_radius::SMALL,
                        bubble_effect: false,
                        slider_effect: false,
                    };
                    vertices.extend_from_slice(&bg.to_vertices());
                    for (i, _row) in chat.mention_rows.iter().enumerate().take(12) {
                        let row_y = rect.y + 4.0 + i as f32 * 28.0;
                        let row_rect = Rect::new(rect.x, row_y, rect.width, 28.0);
                        if i == chat.mention_selected_index {
                            let hi = Quad {
                                position: row_rect.position(),
                                size: row_rect.size(),
                                color: style::highlight::SELECTION(),
                                corner_radius: 0.0,
                                bubble_effect: false,
                                slider_effect: false,
                            };
                            vertices.extend_from_slice(&hi.to_vertices());
                        }
                    }
                }
            }

            let send_rect = Rect::from_pos_size(chat.send_button_position, chat.send_button_size);
            let send_bg = Quad {
                position: send_rect.position(),
                size: send_rect.size(),
                color: style::button::PRIMARY(),
                corner_radius: style::corner_radius::MEDIUM,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&send_bg.to_vertices());

            if !vertices.is_empty() {
                renderer.add_vertices(vertices, None);
                vertices.clear();
            }
            renderer.set_composite_layer(CompositeLayer::HudChrome);

            use crate::ui::icons::icon_names;
            let icon_size = 20.0;
            let icon_pos = Vec2::new(
                context_pool_rect.x + context_pool_rect.width / 2.0 - icon_size / 2.0,
                context_pool_rect.y + context_pool_rect.height / 2.0 - icon_size / 2.0,
            );
            renderer.queue_icon(icon_names::BOOK, icon_pos, icon_size, style::text::PRIMARY());

            use crate::ui::components::Renderable;
            renderer.push_parent("chat_context_pool_dropdown".to_string());
            renderer.validate_component(
                "chat_context_pool_dropdown",
                Some("chat"),
                "ContextPoolDropdown",
            );
            chat.context_pool_dropdown
                .render(renderer, app, vertices, None);
            renderer.pop_parent();

            renderer.push_parent("chat_system_prompt_dropdown".to_string());
            renderer.validate_component(
                "chat_system_prompt_dropdown",
                Some("chat"),
                "SystemPromptDropdown",
            );
            chat.system_prompt_dropdown
                .render(renderer, app, vertices, None);
            renderer.pop_parent();

            for (pi, item) in chat.composer_pill_items.iter().enumerate() {
                let mut pill_text = Text::new_for_render(&item.label)
                    .with_font_size(style::font_size::SMALL)
                    .with_color(style::text::PRIMARY())
                    .with_alignment(TextAlignment::Left)
                    .with_scissor(None);
                let label_rect = item.body_rect.inset(4.0);
                pill_text.update_layout(label_rect, None, None);
                let pill_id = format!("chat_composer_pill_{}", pi);
                renderer.push_parent(pill_id.clone());
                renderer.validate_component(&pill_id, Some("chat"), "ComposerPill");
                pill_text.render(renderer, app, vertices, None);
                renderer.pop_parent();
                let cx = item.close_rect.x + item.close_rect.width * 0.5 - 6.0;
                let cy = item.close_rect.y + item.close_rect.height * 0.5 - 6.0;
                renderer.queue_icon(
                    icon_names::CLOSE,
                    Vec2::new(cx, cy),
                    12.0,
                    style::text::SECONDARY(),
                );
            }

            if chat.mention_popup_open {
                if let Some(rect) = chat.mention_popup_rect() {
                    for (i, row) in chat.mention_rows.iter().enumerate().take(12) {
                        let row_y = rect.y + 4.0 + i as f32 * 28.0;
                        let row_rect = Rect::new(rect.x, row_y, rect.width, 28.0);
                        let label = match row {
                            crate::ui::chat_window::MentionEntry::Paper(id) => app
                                .papers_cache
                                .iter()
                                .find(|p| p.id == *id)
                                .map(|p| {
                                    format!(
                                        "Paper: {}",
                                        p.title.as_deref().unwrap_or(p.filename.as_str())
                                    )
                                })
                                .unwrap_or_else(|| format!("Paper {}", id)),
                            crate::ui::chat_window::MentionEntry::Shard { graph_id, shard_id } => {
                                let ss = if shard_id.len() <= 10 {
                                    shard_id.as_str()
                                } else {
                                    &shard_id[..10]
                                };
                                let gs = if graph_id.len() <= 10 {
                                    graph_id.as_str()
                                } else {
                                    &graph_id[..10]
                                };
                                format!("Shard {} · graph {}", ss, gs)
                            }
                            crate::ui::chat_window::MentionEntry::Graph { graph_id } => {
                                let g = if graph_id.len() <= 20 {
                                    graph_id.as_str()
                                } else {
                                    &graph_id[..20]
                                };
                                format!("Graph {}", g)
                            }
                            crate::ui::chat_window::MentionEntry::Notepad {
                                title,
                                document_id,
                            } => {
                                format!("Note: {} ({})", title, document_id)
                            }
                        };
                        let mut row_text = Text::new_for_render(&label)
                            .with_font_size(style::font_size::NORMAL)
                            .with_color(style::text::PRIMARY())
                            .with_alignment(TextAlignment::Left)
                            .with_scissor(None);
                        let text_rect = row_rect.inset(8.0);
                        row_text.update_layout(text_rect, None, None);
                        let cid = format!("chat_mention_row_{}", i);
                        renderer.push_parent(cid.clone());
                        renderer.validate_component(&cid, Some("chat"), "MentionRow");
                        row_text.render(renderer, app, vertices, None);
                        renderer.pop_parent();
                    }
                }
            }

            text_input_render::render_text_input(
                renderer,
                &input_field,
                app,
                vertices,
                Some(style::font_size::NORMAL),
                Some(style::padding::MEDIUM),
                Some(style::corner_radius::MEDIUM),
                false,
            );

            renderer.push_parent("chat_send_button".to_string());
            renderer.validate_component("chat_send_button", Some("chat"), "SendButton");
            let mut send_text = Text::new_for_render("Send")
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY())
                .with_alignment(TextAlignment::Center)
                .with_scissor(None);
            send_text.update_layout(send_rect, None, None);
            send_text.render(renderer, app, vertices, None);
            renderer.pop_parent();

            // Pop chat parent
            renderer.pop_parent();
        }
    }
}

/// Stateless [`Renderable`] for the chat window; delegates to [`render_chat_window`].
pub struct ChatViewport;

pub const CHAT_VIEWPORT: ChatViewport = ChatViewport;

/// Opt-in drop shadow for the chat viewport chassis.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for ChatViewport {
    fn render(
        &self,
        renderer: &mut Renderer,
        app: &App,
        vertices: &mut Vec<Vertex>,
        _dirty_rect: Option<Rect>,
    ) {
        if let Some(spec) = SHADOW.get() {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), &spec);
            }
        }
        render_chat_window(renderer, app, vertices);
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.chat_window
            .as_ref()
            .map(|w| Rect::new(w.position.x, w.position.y, w.size.x, w.size.y))
    }

    fn update_layout(
        &mut self,
        _available_rect: Rect,
        _dirty_rect: Option<Rect>,
        _app: Option<&App>,
    ) {
    }
}
