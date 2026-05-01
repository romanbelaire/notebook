use super::{render_button, render_modal_container};
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::Vertex;
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use crate::ui::style;

pub(super) fn render_ingest_import_failures_modal(
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    let modal = &app.ingest_import_failures_modal;
    renderer.validate_component(
        "ingest_import_failures_modal",
        Some("modals"),
        "IngestImportFailuresModal",
    );
    renderer.push_parent("ingest_import_failures_modal".to_string());

    render_modal_container(modal.position, modal.size, renderer, vertices);

    const PADDING: f32 = 20.0;
    let header_rect = Rect::new(
        modal.position.x + PADDING,
        modal.position.y + PADDING,
        modal.size.x - PADDING * 2.0,
        28.0,
    );
    let mut header = crate::ui::text::Text::new_for_render("Import failures")
        .with_font_size(style::font_size::LARGE)
        .with_color(style::text::PRIMARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    header.update_layout(header_rect, None, None);
    header.render(renderer, app, vertices, None);

    let body = modal.lines.join("\n");
    let body_rect = Rect::new(
        modal.position.x + PADDING,
        modal.position.y + PADDING + 44.0,
        modal.size.x - PADDING * 2.0,
        modal.size.y - PADDING * 2.0 - 80.0,
    );
    let mut body_text = crate::ui::text::Text::new_for_render(&body)
        .with_font_size(style::font_size::NORMAL)
        .with_color(style::text::PRIMARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    body_text.update_layout(body_rect, None, None);
    body_text.render(renderer, app, vertices, None);

    render_button(
        &modal.close_button,
        "ingest_import_failures_close",
        "ingest_import_failures_modal",
        renderer,
        app,
        vertices,
    );

    renderer.pop_parent();
}
