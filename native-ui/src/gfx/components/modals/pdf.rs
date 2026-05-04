use super::{render_button, render_modal_container};
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::{text_input_render, Rect};
use crate::ui::pdf_modal::{
    DocumentKind, FOOTER_HEIGHT, HEADER_HEIGHT, SEARCH_BAR_HEIGHT, TOC_ENTRY_HEIGHT,
    TOC_PANEL_WIDTH,
};
use crate::ui::style;
use crate::ui::text::{Text, TextAlignment};
use glam::{Vec2, Vec4};

const PADDING: f32 = 12.0;

pub(super) fn render_pdf_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let modal = &app.pdf_modal;

    renderer.validate_component("pdf_modal", Some("modals"), "PdfModal");
    renderer.push_parent("pdf_modal".to_string());

    render_modal_container(modal.position, modal.size, renderer, vertices);

    // ── Header ───────────────────────────────────────────────────────────────
    let header_bg = Quad {
        position: modal.position,
        size: Vec2::new(modal.size.x, HEADER_HEIGHT),
        color: style::bg::SECONDARY(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&header_bg.to_vertices());

    // TOC toggle button
    render_button(
        &modal.toc_button,
        "pdf_modal_toc_button",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );

    // Annotate toggle button (highlight when active)
    render_button(
        &modal.annotate_button,
        "pdf_modal_annotate_button",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );

    // Filename label
    if let Some(ref filename) = modal.filename {
        let clean = filename.split('#').next().unwrap_or(filename.as_str());
        let rect = Rect::new(
            modal.position.x + 84.0,
            modal.position.y,
            modal.size.x - 84.0 - 50.0,
            HEADER_HEIGHT,
        );
        let mut t = Text::new_for_render(clean)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Left);
        t.update_layout(rect, None, None);
        renderer.push_parent("pdf_modal_filename".to_string());
        renderer.validate_component("pdf_modal_filename", Some("modals"), "PdfModalFilename");
        t.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }

    render_button(
        &modal.close_button,
        "pdf_modal_close_button",
        "pdf_modal",
        renderer,
        app,
        vertices,
    );

    // ── TOC panel ────────────────────────────────────────────────────────────
    if modal.toc_open && !modal.pdf_renderer.toc.is_empty() {
        let toc_area = modal.toc_area();

        let toc_bg = Quad {
            position: toc_area.position(),
            size: toc_area.size(),
            color: style::bg::SECONDARY(),
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&toc_bg.to_vertices());

        // Vertical separator
        let sep = Quad {
            position: Vec2::new(toc_area.right() - 1.0, toc_area.y),
            size: Vec2::new(1.0, toc_area.height),
            color: style::border::SUBTLE(),
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&sep.to_vertices());

        let scroll_off = modal.toc_scroll.scroll_offset;

        for (i, entry) in modal.pdf_renderer.toc.iter().enumerate() {
            let entry_y = toc_area.y + i as f32 * TOC_ENTRY_HEIGHT - scroll_off;
            if entry_y + TOC_ENTRY_HEIGHT < toc_area.y || entry_y > toc_area.bottom() {
                continue;
            }

            // Highlight current-page entry
            let is_current = entry.page == modal.current_page;
            if is_current {
                let highlight = Quad {
                    position: Vec2::new(toc_area.x, entry_y),
                    size: Vec2::new(toc_area.width, TOC_ENTRY_HEIGHT),
                    color: style::accent::POP().truncate().extend(0.15),
                    corner_radius: 0.0,
                    bubble_effect: false,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&highlight.to_vertices());
            }

            let indent = entry.level as f32 * 12.0;
            let text_rect = Rect::new(
                toc_area.x + PADDING + indent,
                entry_y,
                toc_area.width - PADDING - indent - 40.0,
                TOC_ENTRY_HEIGHT,
            );
            let page_rect = Rect::new(
                toc_area.right() - 40.0,
                entry_y,
                36.0,
                TOC_ENTRY_HEIGHT,
            );

            let text_color = if is_current {
                style::accent::POP()
            } else {
                style::text::PRIMARY()
            };

            let mut title_text = Text::new_for_render(&entry.title)
                .with_font_size(style::font_size::SMALL)
                .with_color(text_color)
                .with_alignment(TextAlignment::Left);
            title_text.update_layout(text_rect, None, None);
            renderer.push_parent(format!("pdf_toc_entry_{}", i));
            renderer.validate_component(
                &format!("pdf_toc_entry_{}", i),
                Some("modals"),
                "PdfTocEntry",
            );
            title_text.render(renderer, app, vertices, None);
            renderer.pop_parent();

            let page_str = entry.page.to_string();
            let mut page_text = Text::new_for_render(&page_str)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::SECONDARY())
                .with_alignment(TextAlignment::Right);
            page_text.update_layout(page_rect, None, None);
            renderer.push_parent(format!("pdf_toc_page_{}", i));
            renderer.validate_component(
                &format!("pdf_toc_page_{}", i),
                Some("modals"),
                "PdfTocPage",
            );
            page_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
    }

    // ── Search bar ───────────────────────────────────────────────────────────
    if modal.search_active {
        let bar = modal.search_bar_rect();
        let bar_bg = Quad {
            position: bar.position(),
            size: bar.size(),
            color: style::bg::PANEL_POPUP(),
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&bar_bg.to_vertices());

        let mut search_in = modal.search_input.clone();
        search_in.cursor_visible = app.cursor_visible;
        search_in.cursor_animation_value = app.cursor_position_animation.value;
        text_input_render::render_text_input(
            renderer,
            &search_in,
            app,
            vertices,
            Some(style::font_size::SMALL),
            Some(4.0),
            Some(4.0),
            false,
        );

        // Match count label
        let match_label = if modal.search_matches.is_empty() {
            if modal.search_input.text.is_empty() {
                String::new()
            } else {
                "No matches".to_string()
            }
        } else {
            format!(
                "{} / {}",
                modal.search_current + 1,
                modal.search_matches.len()
            )
        };
        if !match_label.is_empty() {
            let label_rect = Rect::new(
                bar.x + 296.0,
                bar.y,
                bar.width - 296.0 - PADDING,
                SEARCH_BAR_HEIGHT,
            );
            let mut lbl = Text::new_for_render(&match_label)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::SECONDARY())
                .with_alignment(TextAlignment::Left);
            lbl.update_layout(label_rect, None, None);
            renderer.push_parent("pdf_search_count".to_string());
            renderer.validate_component("pdf_search_count", Some("modals"), "PdfSearchCount");
            lbl.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
    }

    // ── Main content area ────────────────────────────────────────────────────
    let content_area = modal.content_area();

    let content_bg = Quad {
        position: content_area.position(),
        size: content_area.size(),
        color: Vec4::new(0.13, 0.13, 0.14, 1.0),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&content_bg.to_vertices());

    if modal.loading {
        render_center_text("Loading…", style::text::SECONDARY(), &content_area, renderer, app, vertices, "pdf_modal_loading");
    } else if let Some(ref err) = modal.error {
        render_center_text(err, style::accent::WARNING(), &content_area, renderer, app, vertices, "pdf_modal_error");
    } else if modal.document_kind == DocumentKind::Pdf && modal.rendered_page.is_some() {
        render_pdf_page(renderer, app, vertices, modal, &content_area);
    } else if modal.document_kind == DocumentKind::TextLike && modal.text_content.is_some() {
        let text = modal.text_content.as_ref().unwrap();
        let text_rect = Rect::new(
            content_area.x + PADDING,
            content_area.y + PADDING,
            content_area.width - PADDING * 2.0,
            content_area.height - PADDING * 2.0,
        );
        let mut tv = Text::new_for_render(text)
            .with_font_size(style::font_size::SMALL)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Left);
        tv.update_layout(text_rect, None, None);
        renderer.push_parent("pdf_modal_text_content".to_string());
        renderer.validate_component(
            "pdf_modal_text_content",
            Some("modals"),
            "PdfModalTextContent",
        );
        tv.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else if modal.filename.is_some() {
        render_center_text(
            "PDF content not available. Please wait for the file to load.",
            style::text::SECONDARY(),
            &content_area,
            renderer,
            app,
            vertices,
            "pdf_modal_status",
        );
    }

    // Annotation note input (shown at bottom of content when active)
    if modal.note_active {
        let note_bg = Quad {
            position: modal.note_input.position - Vec2::new(0.0, 4.0),
            size: modal.note_input.size + Vec2::new(0.0, 8.0),
            color: style::bg::PANEL_POPUP(),
            corner_radius: 4.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&note_bg.to_vertices());
        let mut note_in = modal.note_input.clone();
        note_in.cursor_visible = app.cursor_visible;
        note_in.cursor_animation_value = app.cursor_position_animation.value;
        text_input_render::render_text_input(
            renderer,
            &note_in,
            app,
            vertices,
            Some(style::font_size::SMALL),
            Some(4.0),
            Some(4.0),
            false,
        );
    }

    // ── Footer ───────────────────────────────────────────────────────────────
    let footer_y = modal.position.y + modal.size.y - FOOTER_HEIGHT;
    let footer_bg = Quad {
        position: Vec2::new(modal.position.x, footer_y),
        size: Vec2::new(modal.size.x, FOOTER_HEIGHT),
        color: style::bg::SECONDARY(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&footer_bg.to_vertices());

    // Footer separator
    let sep = Quad {
        position: Vec2::new(modal.position.x, footer_y),
        size: Vec2::new(modal.size.x, 1.0),
        color: style::border::SUBTLE(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&sep.to_vertices());

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

    // Page counter / goto input
    if let Some(total) = modal.total_pages {
        if modal.goto_active {
            // Editable page number input
            let mut goto_in = modal.goto_input.clone();
            goto_in.cursor_visible = app.cursor_visible;
            goto_in.cursor_animation_value = app.cursor_position_animation.value;
            text_input_render::render_text_input(
                renderer,
                &goto_in,
                app,
                vertices,
                Some(style::font_size::SMALL),
                Some(4.0),
                Some(3.0),
                false,
            );
            let of_rect = Rect::new(
                modal.goto_input.position.x + 54.0,
                footer_y,
                120.0,
                FOOTER_HEIGHT,
            );
            let of_text = format!("of {}", total);
            let mut ot = Text::new_for_render(&of_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::SECONDARY())
                .with_alignment(TextAlignment::Left);
            ot.update_layout(of_rect, None, None);
            renderer.push_parent("pdf_modal_of_label".to_string());
            renderer.validate_component("pdf_modal_of_label", Some("modals"), "PdfOfLabel");
            ot.render(renderer, app, vertices, None);
            renderer.pop_parent();
        } else {
            let search_indicator = if !modal.search_matches.is_empty() {
                format!(
                    "  🔍 {}/{}",
                    modal.search_current + 1,
                    modal.search_matches.len()
                )
            } else {
                String::new()
            };
            let page_text = format!(
                "Page {} of {}{}   {:.0}%   {:.0}×{:.0} pt",
                modal.current_page,
                total,
                search_indicator,
                modal.zoom_level * 100.0,
                modal.pdf_renderer.get_page_info(modal.current_page).width_points,
                modal.pdf_renderer.get_page_info(modal.current_page).height_points,
            );
            let counter_rect = Rect::new(modal.position.x + 200.0, footer_y, 500.0, FOOTER_HEIGHT);
            let mut ct = Text::new_for_render(&page_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::SECONDARY())
                .with_alignment(TextAlignment::Left);
            ct.update_layout(counter_rect, None, None);
            renderer.push_parent("pdf_modal_page_counter".to_string());
            renderer.validate_component(
                "pdf_modal_page_counter",
                Some("modals"),
                "PdfModalPageCounter",
            );
            ct.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
    }

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

    renderer.pop_parent();
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn render_pdf_page(
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
    modal: &crate::ui::PdfModal,
    content_area: &Rect,
) {
    let rendered = modal.rendered_page.as_ref().unwrap();

    renderer.cache_rgba_image(
        &rendered.cache_key,
        &rendered.rgba,
        rendered.width,
        rendered.height,
    );

    let (draw_x, draw_y, draw_w, draw_h) = modal.page_draw_rect();

    renderer.draw_cached_image(
        &rendered.cache_key,
        (draw_x, draw_y, draw_w.max(1.0), draw_h.max(1.0)),
        Some(content_area),
    );

    // ── Search highlight quads ───────────────────────────────────────────────
    // Highlight all pages that are in the search results (visual cue when on page).
    if !modal.search_matches.is_empty() && modal.search_matches.contains(&modal.current_page) {
        // Thin yellow border around the entire content area as a "match" indicator.
        let border_color = Vec4::new(1.0, 0.9, 0.1, 0.6);
        let bw = 3.0;
        // Top, bottom, left, right border strips
        for (bpos, bsize) in [
            (
                Vec2::new(content_area.x, content_area.y),
                Vec2::new(content_area.width, bw),
            ),
            (
                Vec2::new(content_area.x, content_area.bottom() - bw),
                Vec2::new(content_area.width, bw),
            ),
            (
                Vec2::new(content_area.x, content_area.y),
                Vec2::new(bw, content_area.height),
            ),
            (
                Vec2::new(content_area.right() - bw, content_area.y),
                Vec2::new(bw, content_area.height),
            ),
        ] {
            let q = Quad {
                position: bpos,
                size: bsize,
                color: border_color,
                corner_radius: 0.0,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&q.to_vertices());
        }
    }

    // ── Annotations ─────────────────────────────────────────────────────────
    for (idx, ann) in modal.annotations.iter().enumerate() {
        if ann.page != modal.current_page {
            continue;
        }
        let info = modal.pdf_renderer.get_page_info(modal.current_page);
        if info.width_points == 0.0 || info.height_points == 0.0 {
            continue;
        }
        let ax = draw_x + ann.x0 / info.width_points * draw_w;
        let ay = draw_y + (1.0 - ann.y1 / info.height_points) * draw_h;
        let aw = (ann.x1 - ann.x0) / info.width_points * draw_w;
        let ah = (ann.y1 - ann.y0) / info.height_points * draw_h;

        let ann_quad = Quad {
            position: Vec2::new(ax, ay),
            size: Vec2::new(aw.max(2.0), ah.max(2.0)),
            color: Vec4::from(ann.color),
            corner_radius: 2.0,
            bubble_effect: false,
            slider_effect: false,
        };
        // Draw within clip.
        let visible_x = ax.max(content_area.x);
        let visible_y = ay.max(content_area.y);
        let visible_r = (ax + aw).min(content_area.right());
        let visible_b = (ay + ah).min(content_area.bottom());
        if visible_r > visible_x && visible_b > visible_y {
            vertices.extend_from_slice(&ann_quad.to_vertices());
        }

        // Note indicator (small dot + tooltip area)
        if !ann.note.is_empty() {
            let dot_x = ax + aw - 8.0;
            let dot_y = ay;
            if dot_x >= content_area.x
                && dot_x <= content_area.right()
                && dot_y >= content_area.y
                && dot_y <= content_area.bottom()
            {
                let dot = Quad {
                    position: Vec2::new(dot_x, dot_y),
                    size: Vec2::new(8.0, 8.0),
                    color: Vec4::new(0.8, 0.5, 0.1, 1.0),
                    corner_radius: 4.0,
                    bubble_effect: false,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&dot.to_vertices());
            }
        }

        // Annotation index label for identification.
        let _ = idx;
    }

    // ── Active annotation drag rect ──────────────────────────────────────────
    if modal.annotation_mode && modal.annotation_drag_start.is_some() {
        let start = modal.annotation_drag_start.unwrap();
        let end = modal.annotation_drag_current;
        let drag_x = start.x.min(end.x);
        let drag_y = start.y.min(end.y);
        let drag_w = (end.x - start.x).abs().max(1.0);
        let drag_h = (end.y - start.y).abs().max(1.0);

        // Clip to content area
        let vis_x = drag_x.max(content_area.x);
        let vis_y = drag_y.max(content_area.y);
        let vis_r = (drag_x + drag_w).min(content_area.right());
        let vis_b = (drag_y + drag_h).min(content_area.bottom());
        if vis_r > vis_x && vis_b > vis_y {
            let drag_quad = Quad {
                position: Vec2::new(vis_x, vis_y),
                size: Vec2::new(vis_r - vis_x, vis_b - vis_y),
                color: Vec4::new(1.0, 0.95, 0.25, 0.35),
                corner_radius: 0.0,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&drag_quad.to_vertices());
        }
    }
}

fn render_center_text(
    text: &str,
    color: Vec4,
    content_area: &Rect,
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
    id: &str,
) {
    let rect = Rect::new(
        content_area.x,
        content_area.y + content_area.height * 0.5 - 20.0,
        content_area.width,
        40.0,
    );
    let mut t = Text::new_for_render(text)
        .with_font_size(style::font_size::NORMAL)
        .with_color(color)
        .with_alignment(TextAlignment::Center);
    t.update_layout(rect, None, None);
    renderer.push_parent(id.to_string());
    renderer.validate_component(id, Some("modals"), "PdfCenterText");
    t.render(renderer, app, vertices, None);
    renderer.pop_parent();
}
