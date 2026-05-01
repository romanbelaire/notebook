use super::{render_button, render_modal_container};
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::{layout, text_input_render, Rect};
use crate::ui::style;
use glam::Vec2;

pub(super) fn render_insight_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let modal = &app.insight_modal;

    if modal.insight.is_none() {
        return;
    }

    let insight = modal.insight.as_ref().unwrap();

    // Validate and push "insight_modal" as parent for all components in this modal
    renderer.validate_component("insight_modal", Some("modals"), "InsightModal");
    renderer.push_parent("insight_modal".to_string());

    // Render modal container
    render_modal_container(modal.position, modal.size, renderer, vertices);

    const PADDING: f32 = 20.0;
    const HEADER_HEIGHT: f32 = 60.0;

    // Create container for vertical stacking
    let container = Rect::new(
        modal.position.x + PADDING,
        modal.position.y + PADDING,
        modal.size.x - PADDING * 2.0,
        modal.size.y - PADDING * 2.0 - 50.0, // Reserve space for footer
    );

    // Build vertical stack: header, content label, content area
    let mut section_heights = vec![HEADER_HEIGHT, 20.0]; // Header and "Content:" label

    // Add content area height
    if modal.is_editing_text {
        section_heights.push(modal.text_input.size.y); // Text input area
        section_heights.push(30.0); // Save button
    } else {
        section_heights.push(200.0); // Content display area
        section_heights.push(30.0); // Edit button
    }

    // Stack sections vertically
    let section_rects = layout::stack_vertical(&container, &section_heights, PADDING, 0.0);

    // Header with title and close button
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
        // Title display with edit button using Text component
        let title_text = if insight.title.is_empty() {
            "Insight"
        } else {
            &insight.title
        };
        let title_rect = Rect::new(
            header_rect.x,
            header_rect.y,
            header_rect.width,
            header_rect.height,
        );

        let mut title_text_component = crate::ui::text::Text::new_for_render(title_text)
            .with_font_size(style::font_size::XLARGE)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Left);
        title_text_component.update_layout(title_rect, None, None);

        renderer.push_parent("insight_modal_title".to_string());
        renderer.validate_component("insight_modal_title", Some("modals"), "InsightModalTitle");
        title_text_component.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }

    // Close button
    render_button(
        &modal.close_button,
        "insight_modal_close_button",
        "insight_modal",
        renderer,
        app,
        vertices,
    );

    // Content section label using Text component
    let content_label_rect = section_rects[1];
    let label_rect = Rect::from_pos_size(content_label_rect.position(), content_label_rect.size());

    let mut content_label = crate::ui::text::Text::new_for_render("Content:")
        .with_font_size(style::font_size::SMALL)
        .with_color(style::text::SECONDARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    content_label.update_layout(label_rect, None, None);

    renderer.push_parent("insight_modal_content_label".to_string());
    renderer.validate_component(
        "insight_modal_content_label",
        Some("modals"),
        "InsightModalContentLabel",
    );
    content_label.render(renderer, app, vertices, None);
    renderer.pop_parent();

    if modal.is_editing_text {
        // Text input area - use standard text input rendering
        let content_rect = section_rects[2];
        let mut text_input_field = modal.text_input.clone();
        text_input_field.text = modal.draft_text.clone();
        text_input_field.position = content_rect.position();
        text_input_field.size = content_rect.size();
        text_input_render::render_text_input(
            renderer,
            &text_input_field,
            app,
            vertices,
            None,
            None,
            None,
            false,
        );

        // Save/Cancel buttons
        render_button(
            &modal.save_button,
            "insight_modal_save_button",
            "insight_modal",
            renderer,
            app,
            vertices,
        );
    } else {
        // Display markdown content (simplified - just show text for now)
        let content_rect = section_rects[2];
        let content_bg = Quad {
            position: content_rect.position(),
            size: content_rect.size(),
            color: style::bg::SECONDARY(),
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&content_bg.to_vertices());

        // Render text with word wrapping
        let text_pos = Vec2::new(
            content_rect.x + style::padding::SMALL,
            content_rect.y + style::padding::SMALL,
        );
        let max_width = content_rect.width - style::padding::SMALL * 2.0;
        let words: Vec<&str> = insight.text.split_whitespace().collect();
        let mut line = String::new();
        let mut y_offset = 0.0;
        let mut line_index = 0;

        for word in words {
            let test_line = if line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", line, word)
            };

            let test_width = renderer
                .measure_text(&test_line, style::font_size::NORMAL)
                .x;

            if test_width > max_width {
                if !line.is_empty() {
                    // Render line using Text component
                    let line_rect = Rect::new(text_pos.x, text_pos.y + y_offset, max_width, 18.0);

                    let mut line_text = crate::ui::text::Text::new_for_render(&line)
                        .with_font_size(style::font_size::NORMAL)
                        .with_color(style::text::PRIMARY())
                        .with_alignment(crate::ui::text::TextAlignment::Left);
                    line_text.update_layout(line_rect, None, None);

                    let component_id = format!("insight_modal_content_line_{}", line_index);
                    renderer.push_parent(component_id.clone());
                    renderer.validate_component(
                        &component_id,
                        Some("modals"),
                        "InsightModalContentLine",
                    );
                    line_text.render(renderer, app, vertices, None);
                    renderer.pop_parent();

                    y_offset += 18.0;
                    line_index += 1;
                    line.clear();
                }
                line = word.to_string();
            } else {
                line = test_line;
            }
        }

        if !line.is_empty() {
            // Render line using Text component
            let line_rect = Rect::new(text_pos.x, text_pos.y + y_offset, max_width, 18.0);

            let mut line_text = crate::ui::text::Text::new_for_render(&line)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY())
                .with_alignment(crate::ui::text::TextAlignment::Left);
            line_text.update_layout(line_rect, None, None);

            let component_id = format!("insight_modal_content_line_{}", line_index);
            renderer.push_parent(component_id.clone());
            renderer.validate_component(&component_id, Some("modals"), "InsightModalContentLine");
            line_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }

        // Edit button
        let edit_button_rect = section_rects[3];
        let edit_button =
            crate::ui::Button::new(edit_button_rect.position(), Vec2::new(60.0, 30.0), "Edit");
        render_button(
            &edit_button,
            "insight_modal_edit_button",
            "insight_modal",
            renderer,
            app,
            vertices,
        );
    }

    // Footer buttons
    let footer_y = modal.position.y + modal.size.y - 50.0;
    render_button(
        &modal.delete_button,
        "insight_modal_delete_button",
        "insight_modal",
        renderer,
        app,
        vertices,
    );

    // Close button in footer (if not already rendered in header)
    let close_footer = crate::ui::Button::new(
        Vec2::new(modal.position.x + modal.size.x - 100.0, footer_y),
        Vec2::new(80.0, 30.0),
        "Close",
    );
    render_button(
        &close_footer,
        "insight_modal_close_footer",
        "insight_modal",
        renderer,
        app,
        vertices,
    );

    // Pop "insight_modal" parent
    renderer.pop_parent();
}
