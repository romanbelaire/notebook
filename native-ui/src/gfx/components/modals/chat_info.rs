use super::{render_button, render_modal_container};
use crate::api::models::Insight;
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::{layout, text_input_render, Rect};
use crate::ui::style;
use glam::Vec2;

pub(super) fn render_chat_info_dialog(
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    let modal = &app.chat_info_dialog;

    if modal.conversation_id.is_none() {
        return;
    }

    // Validate and push "chat_info_dialog" as parent for all components in this modal
    renderer.validate_component("chat_info_dialog", Some("modals"), "ChatInfoDialog");
    renderer.push_parent("chat_info_dialog".to_string());

    // Render modal container
    render_modal_container(modal.position, modal.size, renderer, vertices);

    const PADDING: f32 = 20.0;

    // Create container for vertical stacking
    let container = Rect::new(
        modal.position.x + PADDING,
        modal.position.y + PADDING,
        modal.size.x - PADDING * 2.0,
        modal.size.y - PADDING * 2.0 - 50.0, // Reserve space for footer
    );

    // Build vertical stack: header, citations label, citations list, insights label, insights list
    let header_height = 50.0;
    let label_height = 30.0;
    let citations_list_height = modal.citations_list.size.y;
    let insights_list_height = modal.insights_list.size.y;

    let section_heights = vec![
        header_height,
        label_height,
        citations_list_height,
        label_height,
        insights_list_height,
    ];

    // Stack sections vertically
    let section_rects = layout::stack_vertical(&container, &section_heights, PADDING, 0.0);

    // Header with title
    let header_rect = section_rects[0];
    if modal.is_editing_title {
        // Title input field - use standard text input rendering
        let mut title_input = modal.title_input.clone();
        title_input.text = modal.draft_title.clone();
        title_input.position = header_rect.position();
        title_input.size = Vec2::new(header_rect.width, header_rect.height);
        text_input_render::render_text_input(
            renderer,
            &title_input,
            app,
            vertices,
            Some(style::font_size::XLARGE),
            None,
            None,
            false,
        );
    } else {
        // Title display using Text component
        let title_rect = Rect::new(
            header_rect.x,
            header_rect.y,
            header_rect.width,
            header_rect.height,
        );

        let mut title_text = crate::ui::text::Text::new_for_render(&modal.draft_title)
            .with_font_size(style::font_size::XLARGE)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Left);
        title_text.update_layout(title_rect, None, None);

        renderer.push_parent("chat_info_dialog_title".to_string());
        renderer.validate_component(
            "chat_info_dialog_title",
            Some("chat_info_dialog"),
            "ChatInfoDialogTitle",
        );
        title_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }

    // Close button
    render_button(
        &modal.close_button,
        "chat_info_dialog_close_button",
        "chat_info_dialog",
        renderer,
        app,
        vertices,
    );

    // Citations section label using Text component
    let citations_label_rect = section_rects[1];
    let label_rect =
        Rect::from_pos_size(citations_label_rect.position(), citations_label_rect.size());

    let mut citations_label = crate::ui::text::Text::new_for_render("Citations:")
        .with_font_size(style::font_size::MEDIUM)
        .with_color(style::text::PRIMARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    citations_label.update_layout(label_rect, None, None);

    renderer.push_parent("chat_info_dialog_citations_label".to_string());
    renderer.validate_component(
        "chat_info_dialog_citations_label",
        Some("chat_info_dialog"),
        "ChatInfoDialogCitationsLabel",
    );
    citations_label.render(renderer, app, vertices, None);
    renderer.pop_parent();

    // Mode toggle button
    let mode_text = match modal.citation_mode {
        crate::ui::CitationMode::All => "Show Unique",
        crate::ui::CitationMode::Unique => "Show All",
    };
    // Create a temporary button with the correct label
    let mode_button = crate::ui::Button::new(
        modal.mode_toggle_button.position,
        modal.mode_toggle_button.size,
        mode_text,
    );
    render_button(
        &mode_button,
        "chat_info_dialog_mode_toggle",
        "chat_info_dialog",
        renderer,
        app,
        vertices,
    );

    // Citations list area
    let citations_rect = section_rects[2];
    let citations_bg = Quad {
        position: citations_rect.position(),
        size: citations_rect.size(),
        color: style::bg::SECONDARY(),
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&citations_bg.to_vertices());

    // Get citations from conversation
    let all_citations: Vec<crate::ui::chat_window::Citation> =
        if let Some(conv_id) = &modal.conversation_id {
            app.chat_state
                .conversations
                .iter()
                .find(|c| c.id == *conv_id)
                .map(|c| {
                    // Get citations directly from shards
                    c.shards
                        .iter()
                        .filter_map(|s| {
                            if matches!(
                                s.metadata.role,
                                crate::ui::chat_window::MessageRole::Assistant
                            ) {
                                Some(s.metadata.citations.iter().cloned())
                            } else {
                                None
                            }
                        })
                        .flatten()
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

    // Filter citations based on mode
    let citations: Vec<crate::ui::chat_window::Citation> = match modal.citation_mode {
        crate::ui::CitationMode::All => all_citations,
        crate::ui::CitationMode::Unique => {
            let mut seen = std::collections::HashSet::new();
            all_citations
                .into_iter()
                .filter(|cit| {
                    let key = format!(
                        "{}:{}",
                        cit.source,
                        cit.title.as_ref().unwrap_or(&String::new())
                    );
                    seen.insert(key)
                })
                .collect()
        }
    };

    // Render citations with full details
    if citations.is_empty() {
        let no_citations_rect = Rect::new(
            citations_rect.x + PADDING,
            citations_rect.y + PADDING,
            citations_rect.width - PADDING * 2.0,
            30.0,
        );

        let mut no_citations_text = crate::ui::text::Text::new_for_render("No citations")
            .with_font_size(style::font_size::SMALL)
            .with_color(style::text::SECONDARY())
            .with_alignment(crate::ui::text::TextAlignment::Left);
        no_citations_text.update_layout(no_citations_rect, None, None);

        renderer.push_parent("chat_info_dialog_no_citations".to_string());
        renderer.validate_component(
            "chat_info_dialog_no_citations",
            Some("chat_info_dialog"),
            "ChatInfoDialogNoCitations",
        );
        no_citations_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else {
        // Render citations list
        let item_height = 25.0;
        let scroll_offset = modal.citations_list.scroll_offset;
        let mut y_offset = citations_rect.y + PADDING - scroll_offset;

        for (i, citation) in citations.iter().enumerate() {
            if y_offset + item_height < citations_rect.y {
                y_offset += item_height;
                continue;
            }
            if y_offset > citations_rect.y + citations_rect.height {
                break;
            }

            // Format citation: Title (Source, Year) – Section, p.Page
            let mut citation_text = String::new();
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

            let citation_item_rect = Rect::new(
                citations_rect.x + PADDING,
                y_offset,
                citations_rect.width - PADDING * 2.0 - 25.0, // Space for magnify icon
                item_height,
            );

            let mut citation_text_component = crate::ui::text::Text::new_for_render(&citation_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::PRIMARY())
                .with_alignment(crate::ui::text::TextAlignment::Left);
            citation_text_component.update_layout(citation_item_rect, None, None);

            let component_id = format!("chat_info_citation_{}", i);
            renderer.push_parent(component_id.clone());
            renderer.validate_component(&component_id, Some("modals"), "ChatInfoCitation");
            citation_text_component.render(renderer, app, vertices, None);
            renderer.pop_parent();

            // Render magnify icon
            use crate::ui::icons::icon_names;
            let icon_pos = Vec2::new(
                citations_rect.x + citations_rect.width - 30.0,
                y_offset + item_height / 2.0 - 7.0,
            );
            renderer.queue_icon(icon_names::MAGNIFY, icon_pos, 14.0, style::text::SECONDARY());

            y_offset += item_height;
        }

        // Update scroll content height
        let _total_height = citations.len() as f32 * item_height + PADDING * 2.0;
        // Note: We can't directly mutate modal here, but the scroll view should handle this
    }

    // Insights section label using Text component
    let insights_label_rect = section_rects[3];
    let label_rect =
        Rect::from_pos_size(insights_label_rect.position(), insights_label_rect.size());

    let mut insights_label = crate::ui::text::Text::new_for_render("Pinned Insights:")
        .with_font_size(style::font_size::MEDIUM)
        .with_color(style::text::PRIMARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    insights_label.update_layout(label_rect, None, None);

    renderer.push_parent("chat_info_dialog_insights_label".to_string());
    renderer.validate_component(
        "chat_info_dialog_insights_label",
        Some("chat_info_dialog"),
        "ChatInfoDialogInsightsLabel",
    );
    insights_label.render(renderer, app, vertices, None);
    renderer.pop_parent();

    // Insights list area
    let insights_rect = Rect::new(
        section_rects[4].x,
        section_rects[4].y,
        section_rects[4].width,
        section_rects[4].height,
    );
    let insights_bg = Quad {
        position: insights_rect.position(),
        size: insights_rect.size(),
        color: style::bg::SECONDARY(),
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&insights_bg.to_vertices());

    // Get insights for this conversation
    let conversation_insights: Vec<&Insight> = if let Some(conv_id) = &modal.conversation_id {
        // Match insights by checking if their text matches any message content
        if let Some(conv) = app
            .chat_state
            .conversations
            .iter()
            .find(|c| c.id == *conv_id)
        {
            let message_texts: std::collections::HashSet<String> =
                conv.shards.iter().map(|s| s.text.clone()).collect();

            app.insights_state
                .insights
                .iter()
                .filter(|insight| message_texts.contains(&insight.text))
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // Render insights list
    if conversation_insights.is_empty() {
        let no_insights_rect = Rect::new(
            insights_rect.x + PADDING,
            insights_rect.y + PADDING,
            insights_rect.width - PADDING * 2.0,
            30.0,
        );

        let mut no_insights_text =
            crate::ui::text::Text::new_for_render("No pinned insights from this chat")
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::SECONDARY())
                .with_alignment(crate::ui::text::TextAlignment::Left);
        no_insights_text.update_layout(no_insights_rect, None, None);

        renderer.push_parent("chat_info_dialog_no_insights".to_string());
        renderer.validate_component(
            "chat_info_dialog_no_insights",
            Some("chat_info_dialog"),
            "ChatInfoDialogNoInsights",
        );
        no_insights_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else {
        let item_height = 30.0;
        let scroll_offset = modal.insights_list.scroll_offset;
        let mut y_offset = insights_rect.y + PADDING - scroll_offset;

        for (i, insight) in conversation_insights.iter().enumerate() {
            if y_offset + item_height < insights_rect.y {
                y_offset += item_height;
                continue;
            }
            if y_offset > insights_rect.y + insights_rect.height {
                break;
            }

            let display_text = if !insight.title.is_empty() {
                if insight.title.len() > 60 {
                    format!("{}...", &insight.title[..60])
                } else {
                    insight.title.clone()
                }
            } else if insight.text.len() > 60 {
                format!("{}...", &insight.text[..60])
            } else {
                insight.text.clone()
            };

            let insight_item_rect = Rect::new(
                insights_rect.x + PADDING,
                y_offset,
                insights_rect.width - PADDING * 2.0,
                item_height,
            );

            let mut insight_text_component = crate::ui::text::Text::new_for_render(&display_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::PRIMARY())
                .with_alignment(crate::ui::text::TextAlignment::Left);
            insight_text_component.update_layout(insight_item_rect, None, None);

            let component_id = format!("chat_info_insight_{}", i);
            renderer.push_parent(component_id.clone());
            renderer.validate_component(&component_id, Some("chat_info_dialog"), "ChatInfoInsight");
            insight_text_component.render(renderer, app, vertices, None);
            renderer.pop_parent();

            y_offset += item_height;
        }
    }

    // Footer buttons
    let footer_y = modal.position.y + modal.size.y - 50.0;
    render_button(
        &modal.delete_button,
        "chat_info_dialog_delete_button",
        "chat_info_dialog",
        renderer,
        app,
        vertices,
    );

    let close_footer = crate::ui::Button::new(
        Vec2::new(modal.position.x + modal.size.x - 100.0, footer_y),
        Vec2::new(80.0, 30.0),
        "Close",
    );
    render_button(
        &close_footer,
        "chat_info_dialog_close_footer",
        "chat_info_dialog",
        renderer,
        app,
        vertices,
    );

    // Pop "chat_info_dialog" parent
    renderer.pop_parent();
}
