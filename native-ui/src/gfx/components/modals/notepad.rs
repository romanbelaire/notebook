use super::{render_button, render_modal_container};
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use crate::ui::style;
use glam::Vec2;

pub(super) fn render_notepad_modal_from_window(
    renderer: &mut Renderer,
    notepad: &crate::ui::NotepadWindow,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    let modal = &notepad.notepad_modal;

    // Validate and push "notepad_modal" as parent for all components in this modal
    renderer.validate_component("notepad_modal", Some("modals"), "NotepadModal");
    renderer.push_parent("notepad_modal".to_string());

    // Render modal container
    render_modal_container(modal.position, modal.size, renderer, vertices);

    const PADDING: f32 = 20.0;
    const HEADER_HEIGHT: f32 = 50.0;

    // Header
    let header_bg = Quad {
        position: Vec2::new(modal.position.x, modal.position.y),
        size: Vec2::new(modal.size.x, HEADER_HEIGHT),
        color: style::bg::SECONDARY(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&header_bg.to_vertices());

    // Header title using Text component
    let header_title_rect = Rect::new(
        modal.position.x + PADDING,
        modal.position.y,
        modal.size.x - PADDING * 2.0,
        HEADER_HEIGHT,
    );

    let mut header_title = crate::ui::text::Text::new_for_render("Notepads")
        .with_font_size(style::font_size::LARGE)
        .with_color(style::text::PRIMARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    header_title.update_layout(header_title_rect, None, None);

    renderer.push_parent("notepad_modal_title".to_string());
    renderer.validate_component("notepad_modal_title", Some("modals"), "NotepadModalTitle");
    header_title.render(renderer, app, vertices, None);
    renderer.pop_parent();

    // Close button
    render_button(
        &modal.close_button,
        "notepad_modal_close_button",
        "notepad_modal",
        renderer,
        app,
        vertices,
    );

    // Notes list area
    let list_rect = Rect::from_pos_size(modal.papers_list.position, modal.papers_list.size);
    let list_bg = Quad {
        position: list_rect.position(),
        size: list_rect.size(),
        color: style::bg::SECONDARY(),
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&list_bg.to_vertices());

    // Render notes
    if modal.filtered_papers.is_empty() {
        // Empty state using Text component
        let empty_rect = Rect::new(
            list_rect.x + PADDING,
            list_rect.y + PADDING,
            list_rect.width - PADDING * 2.0,
            30.0,
        );

        let mut empty_text = crate::ui::text::Text::new_for_render("No notepads found.")
            .with_font_size(style::font_size::SMALL)
            .with_color(style::text::SECONDARY())
            .with_alignment(crate::ui::text::TextAlignment::Left);
        empty_text.update_layout(empty_rect, None, None);

        renderer.push_parent("notepad_modal_empty".to_string());
        renderer.validate_component("notepad_modal_empty", Some("modals"), "NotepadModalEmpty");
        empty_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else {
        let item_height = 40.0;
        let scroll_offset = modal.papers_list.scroll_offset;
        let mut y_offset = PADDING - scroll_offset;

        for (idx, paper) in modal.filtered_papers.iter().enumerate() {
            if y_offset + item_height < 0.0 {
                y_offset += item_height;
                continue;
            }
            if y_offset > list_rect.height {
                break;
            }

            let item_rect = Rect::new(
                list_rect.x,
                list_rect.y + y_offset,
                list_rect.width,
                item_height,
            );

            // Highlight selected item
            if Some(idx) == modal.selected_paper_index {
                let highlight = Quad {
                    position: item_rect.position(),
                    size: item_rect.size(),
                    color: style::highlight::SELECTION(),
                    corner_radius: 0.0,
                    bubble_effect: false,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&highlight.to_vertices());
            }

            // Paper title using Text component
            let display_name = paper.title.as_ref().unwrap_or(&paper.filename);
            let title_rect = Rect::new(
                item_rect.x + PADDING,
                item_rect.y,
                item_rect.width - PADDING * 2.0,
                item_rect.height,
            );

            let mut paper_title = crate::ui::text::Text::new_for_render(display_name)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY())
                .with_alignment(crate::ui::text::TextAlignment::Left);
            paper_title.update_layout(title_rect, None, None);

            renderer.push_parent("notepad_modal_paper_title".to_string());
            renderer.validate_component(
                "notepad_modal_paper_title",
                Some("modals"),
                "NotepadModalPaperTitle",
            );
            paper_title.render(renderer, app, vertices, None);
            renderer.pop_parent();

            y_offset += item_height;
        }
    }

    // Footer buttons
    let footer_y = modal.position.y + modal.size.y - 50.0;
    render_button(
        &modal.import_button,
        "notepad_modal_import_button",
        "notepad_modal",
        renderer,
        app,
        vertices,
    );
    render_button(
        &modal.delete_button,
        "notepad_modal_delete_button",
        "notepad_modal",
        renderer,
        app,
        vertices,
    );

    // Pop "notepad_modal" parent
    renderer.pop_parent();
}
