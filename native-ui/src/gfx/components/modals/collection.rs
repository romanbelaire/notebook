use super::{render_button, render_modal_container};
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use crate::ui::style;
use glam::{Vec2, Vec4};

pub(super) fn render_collection_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let modal = &app.collection_modal;
    renderer.validate_component("collection_modal", Some("modals"), "CollectionModal");
    renderer.push_parent("collection_modal".to_string());
    render_modal_container(modal.position, modal.size, renderer, vertices);

    let header_h = 44.0;
    let header_bg = Quad {
        position: modal.position,
        size: Vec2::new(modal.size.x, header_h),
        color: style::bg::SECONDARY(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&header_bg.to_vertices());

    let title_rect = Rect::new(modal.position.x + 16.0, modal.position.y, modal.size.x - 80.0, header_h);
    let mut title_text = crate::ui::text::Text::new_for_render(&modal.collection_name)
        .with_font_size(style::font_size::NORMAL)
        .with_color(style::text::PRIMARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    title_text.update_layout(title_rect, None, None);
    renderer.validate_component("collection_modal_title", Some("collection_modal"), "CollectionModalTitle");
    title_text.render(renderer, app, vertices, None);

    render_button(&modal.close_button, "collection_modal_close", "collection_modal", renderer, app, vertices);

    crate::ui::core::text_input_render::render_text_input(renderer, &modal.search_input, app, vertices, None, None, None, false);

    let list_rect = Rect::from_pos_size(modal.papers_list.position, modal.papers_list.size);
    let list_bg = Quad {
        position: list_rect.position(),
        size: list_rect.size(),
        color: Vec4::new(0.08, 0.08, 0.09, 0.9),
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&list_bg.to_vertices());

    let mut y = modal.papers_list.position.y - modal.papers_list.scroll_offset + 8.0;
    for (idx, paper) in modal.filtered_papers.iter().enumerate() {
        if y > modal.papers_list.position.y + modal.papers_list.size.y {
            break;
        }
        let row_rect = Rect::new(modal.papers_list.position.x + 8.0, y, modal.papers_list.size.x - 16.0, 40.0);
        let selected = modal.selected_papers.contains(&paper.id);
        let row_bg = Quad {
            position: row_rect.position(),
            size: row_rect.size(),
            color: if selected {
                style::highlight::SELECTION()
            } else if paper.exists {
                style::bg::SECONDARY()
            } else {
                style::bg::SECONDARY() * Vec4::new(1.0, 1.0, 1.0, 0.45)
            },
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        renderer.add_quad(&row_bg, Some(&list_rect));
        let checkbox = Rect::new(row_rect.x + 8.0, row_rect.y + 12.0, 16.0, 16.0);
        renderer.add_quad(&Quad {
            position: checkbox.position(),
            size: checkbox.size(),
            color: if paper.exists {
                style::border::DEFAULT()
            } else {
                style::border::DEFAULT() * Vec4::new(1.0, 1.0, 1.0, 0.45)
            },
            corner_radius: 2.0,
            bubble_effect: false,
            slider_effect: false,
        }, Some(&list_rect));
        if selected {
            renderer.add_quad(&Quad {
                position: checkbox.position() + Vec2::new(3.0, 3.0),
                size: checkbox.size() - Vec2::new(6.0, 6.0),
                color: style::button::PRIMARY(),
                corner_radius: 1.0,
                bubble_effect: false,
                slider_effect: false,
            }, Some(&list_rect));
        }
        let label = paper.title.as_ref().unwrap_or(&paper.filename);
        let mut row_text = crate::ui::text::Text::new_for_render(label)
            .with_font_size(style::font_size::SMALL)
            .with_color(if paper.exists {
                style::text::PRIMARY()
            } else {
                style::text::SECONDARY() * Vec4::new(1.0, 1.0, 1.0, 0.65)
            })
            .with_alignment(crate::ui::text::TextAlignment::Left)
            .with_scissor(Some(list_rect));
        row_text.update_layout(Rect::new(row_rect.x + 32.0, row_rect.y, row_rect.width - 42.0, row_rect.height), None, None);
        renderer.validate_component(&format!("collection_modal_row_{}", idx), Some("collection_modal"), "CollectionModalRow");
        row_text.render(renderer, app, vertices, None);
        y += 48.0;
    }

    render_button(&modal.delete_button, "collection_modal_delete", "collection_modal", renderer, app, vertices);
    render_button(
        &modal.remove_from_collection_button,
        "collection_modal_remove",
        "collection_modal",
        renderer,
        app,
        vertices,
    );

    renderer.pop_parent();
}
