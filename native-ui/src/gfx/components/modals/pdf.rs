use super::{render_button, render_modal_container};
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use crate::ui::style;
use crate::ui::DocumentKind;
use glam::{Vec2, Vec4};

pub(super) fn render_pdf_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let modal = &app.pdf_modal;

    // Validate and push "pdf_modal" as parent for all components in this modal
    renderer.validate_component("pdf_modal", Some("modals"), "PdfModal");
    renderer.push_parent("pdf_modal".to_string());

    // Render modal container
    render_modal_container(modal.position, modal.size, renderer, vertices);

    const PADDING: f32 = 20.0;
    const HEADER_HEIGHT: f32 = 40.0;

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

    // Filename using Text component
    if let Some(ref filename) = modal.filename {
        let filename_rect = Rect::new(
            modal.position.x + PADDING,
            modal.position.y,
            modal.size.x - PADDING * 2.0,
            HEADER_HEIGHT,
        );

        let mut filename_text = crate::ui::text::Text::new_for_render(filename)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Left);
        filename_text.update_layout(filename_rect, None, None);

        renderer.push_parent("pdf_modal_filename".to_string());
        renderer.validate_component("pdf_modal_filename", Some("modals"), "PdfModalFilename");
        filename_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }

    // Close button
    render_button(
        &modal.close_button,
        "pdf_modal_close_button",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );

    // PDF content area (placeholder for now)
    let content_area = Rect::new(
        modal.position.x,
        modal.position.y + HEADER_HEIGHT,
        modal.size.x,
        modal.size.y - HEADER_HEIGHT - 60.0, // Space for footer
    );

    let content_bg = Quad {
        position: content_area.position(),
        size: content_area.size(),
        color: Vec4::new(0.95, 0.95, 0.95, 1.0),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&content_bg.to_vertices());

    if modal.loading {
        // Loading text using Text component
        let loading_text = "Loading PDF...";
        let loading_rect = Rect::from_pos_size(content_area.position(), content_area.size());

        let mut loading_text_component = crate::ui::text::Text::new_for_render(loading_text)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::SECONDARY())
            .with_alignment(crate::ui::text::TextAlignment::Center);
        loading_text_component.update_layout(loading_rect, None, None);

        renderer.push_parent("pdf_modal_loading".to_string());
        renderer.validate_component("pdf_modal_loading", Some("modals"), "PdfModalLoading");
        loading_text_component.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else if modal.document_kind == DocumentKind::Pdf && modal.rendered_page.is_some() {
        let rendered = modal.rendered_page.as_ref().unwrap();
        renderer.cache_rgba_image(
            &rendered.cache_key,
            &rendered.rgba,
            rendered.width,
            rendered.height,
        );

        let image_aspect = rendered.width as f32 / rendered.height as f32;
        let area_aspect = content_area.width / content_area.height;
        let (draw_w, draw_h) = if image_aspect > area_aspect {
            let w = content_area.width - 20.0;
            (w, w / image_aspect)
        } else {
            let h = content_area.height - 20.0;
            (h * image_aspect, h)
        };
        let draw_x = content_area.x + (content_area.width - draw_w) * 0.5;
        let draw_y = content_area.y + (content_area.height - draw_h) * 0.5;
        renderer.draw_cached_image(
            &rendered.cache_key,
            (draw_x, draw_y, draw_w.max(1.0), draw_h.max(1.0)),
            Some(&content_area),
        );

        // Page counter using Text component
        if let Some(total) = modal.total_pages {
            let page_meta = modal.pdf_renderer.get_page_info(modal.current_page);
            let page_text = format!(
                "Page {} of {}  |  {:.0}%  |  {:.0} x {:.0} pt",
                modal.current_page,
                total,
                modal.zoom_level * 100.0,
                page_meta.width_points,
                page_meta.height_points
            );
            let page_counter_rect = Rect::new(
                modal.position.x + PADDING,
                content_area.bottom() + 10.0,
                500.0,
                20.0,
            );

            let mut page_counter_text = crate::ui::text::Text::new_for_render(&page_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::SECONDARY())
                .with_alignment(crate::ui::text::TextAlignment::Left);
            page_counter_text.update_layout(page_counter_rect, None, None);

            renderer.push_parent("pdf_modal_page_counter".to_string());
            renderer.validate_component(
                "pdf_modal_page_counter",
                Some("modals"),
                "PdfModalPageCounter",
            );
            page_counter_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
    } else if modal.document_kind == DocumentKind::TextLike && modal.text_content.is_some() {
        let text = modal.text_content.as_ref().unwrap();
        let text_rect = Rect::new(
            content_area.x + 12.0,
            content_area.y + 12.0,
            content_area.width - 24.0,
            content_area.height - 24.0,
        );
        let mut text_view = crate::ui::text::Text::new_for_render(text)
            .with_font_size(style::font_size::SMALL)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Left);
        text_view.update_layout(text_rect, None, None);
        renderer.push_parent("pdf_modal_text_content".to_string());
        renderer.validate_component("pdf_modal_text_content", Some("modals"), "PdfModalTextContent");
        text_view.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else if modal.filename.is_some() {
        // Show status message or error
        let (status_text, text_color) = if let Some(ref error) = modal.error {
            (error.as_str(), style::accent::WARNING())
        } else if modal.loading {
            ("Loading PDF...", style::text::SECONDARY())
        } else {
            (
                "PDF content not available. Please wait for the file to load.",
                style::text::SECONDARY(),
            )
        };

        let status_rect = Rect::new(
            content_area.x,
            content_area.y + content_area.height / 2.0 - 20.0,
            content_area.width,
            40.0,
        );

        let mut status_text_component = crate::ui::text::Text::new_for_render(status_text)
            .with_font_size(style::font_size::NORMAL)
            .with_color(text_color)
            .with_alignment(crate::ui::text::TextAlignment::Center);
        status_text_component.update_layout(status_rect, None, None);

        renderer.push_parent("pdf_modal_status".to_string());
        renderer.validate_component("pdf_modal_status", Some("modals"), "PdfModalStatus");
        status_text_component.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // Page counter using Text component
        if let Some(total) = modal.total_pages {
            let page_text = format!("Page {} of {}", modal.current_page, total);
            let page_counter_rect = Rect::new(
                modal.position.x + PADDING,
                content_area.bottom() + 10.0,
                200.0,
                20.0,
            );

            let mut page_counter_text = crate::ui::text::Text::new_for_render(&page_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::SECONDARY())
                .with_alignment(crate::ui::text::TextAlignment::Left);
            page_counter_text.update_layout(page_counter_rect, None, None);

            renderer.push_parent("pdf_modal_page_counter".to_string());
            renderer.validate_component(
                "pdf_modal_page_counter",
                Some("modals"),
                "PdfModalPageCounter",
            );
            page_counter_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
    }

    // Navigation buttons
    let _footer_y = modal.position.y + modal.size.y - 50.0;
    render_button(
        &modal.prev_page_button,
        "pdf_modal_prev_page",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );
    render_button(
        &modal.next_page_button,
        "pdf_modal_next_page",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );
    render_button(
        &modal.zoom_out_button,
        "pdf_modal_zoom_out",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );
    render_button(
        &modal.zoom_in_button,
        "pdf_modal_zoom_in",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );
    render_button(
        &modal.zoom_reset_button,
        "pdf_modal_zoom_reset",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );

    // Pop "pdf_modal" parent
    renderer.pop_parent();
}
