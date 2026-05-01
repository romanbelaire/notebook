use crate::app::App;
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::gfx::types::{Quad, Vertex};
use crate::ui::style;
use glam::{Vec2, Vec4};

mod chat_info;
mod collection;
mod ingest_import_failures;
mod insight;
mod notepad;
mod pdf;
mod shard;
mod system_prompts;

use chat_info::render_chat_info_dialog;
use collection::render_collection_modal;
use ingest_import_failures::render_ingest_import_failures_modal;
use insight::render_insight_modal;
use notepad::render_notepad_modal_from_window;
use pdf::render_pdf_modal;
use shard::render_shard_modal;
use system_prompts::render_system_prompts_modal;

/// Render backdrop overlay for modals
pub(super) fn render_backdrop(viewport_size: Vec2, opacity: f32, vertices: &mut Vec<Vertex>) {
    let bulk = style::bg::PRIMARY();
    let backdrop = Quad {
        position: Vec2::ZERO,
        size: viewport_size,
        color: Vec4::new(bulk.x, bulk.y, bulk.z, opacity),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&backdrop.to_vertices());
}

/// Render modal container (centered window with rounded corners)
pub(super) fn render_modal_container(
    position: Vec2,
    size: Vec2,
    renderer: &mut Renderer,
    vertices: &mut Vec<Vertex>,
) {
    let r = style::stroke::COMPOSER_RULE_PX;

    let rim = Quad {
        position: position - Vec2::splat(r),
        size: size + Vec2::splat(r * 2.0),
        color: style::border::INSTRUMENT_RULE(),
        corner_radius: style::corner_radius::LARGE + r,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&rim.to_vertices());

    let modal_bg = Quad {
        position,
        size,
        color: style::bg::PRIMARY(),
        corner_radius: style::corner_radius::LARGE,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&modal_bg.to_vertices());
}

/// Modal button: validates hierarchy id, delegates drawing to [`crate::ui::Button`]'s [`Renderable`](crate::ui::components::Renderable) impl.
pub(super) fn render_button(
    button: &crate::ui::Button,
    button_id: &str,
    parent_context: &str,
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    use crate::ui::components::Renderable;
    renderer.validate_component(button_id, Some(parent_context), "ModalButton");
    button.render(renderer, app, vertices, None);
}

pub fn render_modals(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::Modal);
    let viewport_size = app.viewport_size;

    // Validate and push "modals" as the root parent for all modal components
    // "root" is added to hierarchy before rendering (see renderer.rs line 1569-1571)
    renderer.validate_component("modals", Some("root"), "Modals");
    renderer.push_parent("modals".to_string());

    // Render InsightModal
    if app.insight_modal.is_open {
        render_backdrop(viewport_size, 0.5, vertices);
        renderer.end_batch();
        render_insight_modal(renderer, app, vertices);
    }

    if app.collection_modal.is_open {
        render_backdrop(viewport_size, 0.45, vertices);
        renderer.end_batch();
        render_collection_modal(renderer, app, vertices);
    }

    // Render PdfModal (topmost over collection modal)
    if app.pdf_modal.is_open {
        render_backdrop(viewport_size, 0.7, vertices);
        renderer.end_batch();
        render_pdf_modal(renderer, app, vertices);
    }

    // Render ChatInfoDialog
    if app.chat_info_dialog.is_open {
        render_backdrop(viewport_size, 0.4, vertices);
        renderer.end_batch();
        render_chat_info_dialog(renderer, app, vertices);
    }

    // Render NotepadModal (from NotepadWindow if open)
    if let Some(ref notepad) = app.notepad_window {
        if notepad.notepad_modal.is_open {
            render_backdrop(viewport_size, 0.6, vertices);
            renderer.end_batch();
            render_notepad_modal_from_window(renderer, notepad, app, vertices);
        }
    }

    // Render ShardModal (edit shard messages)
    if app.shard_modal.is_open {
        render_backdrop(viewport_size, 0.5, vertices);
        renderer.end_batch();
        render_shard_modal(renderer, app, vertices);
    }

    if app.ingest_import_failures_modal.is_open {
        render_backdrop(viewport_size, 0.45, vertices);
        renderer.end_batch();
        render_ingest_import_failures_modal(renderer, app, vertices);
    }

    if app.system_prompts_modal.is_open {
        render_backdrop(viewport_size, 0.5, vertices);
        renderer.end_batch();
        render_system_prompts_modal(renderer, app, vertices);
    }

    render_text_edit_context_menu(renderer, app, vertices);

    // Pop "modals" parent
    renderer.pop_parent();
}

fn render_text_edit_context_menu(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    use crate::ui::components::Renderable;
    use crate::ui::core::Rect;
    use crate::ui::text::Text;
    let Some(ref menu) = app.text_edit_context_menu else {
        return;
    };
    const MENU_WIDTH: f32 = 180.0;
    const ITEM_H: f32 = 28.0;
    const MENU_PAD: f32 = 4.0;
    let items = ["Cut", "Copy", "Paste"];
    let n = items.len();
    let menu_h = n as f32 * ITEM_H + MENU_PAD * 2.0;
    let menu_w = MENU_WIDTH + MENU_PAD * 2.0;
    let mx = menu.position.x;
    let my = menu.position.y;
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
        renderer.push_parent(format!("text_context_menu_item_{}", i));
        renderer.validate_component(
            &format!("text_context_menu_{}", i),
            Some("modals"),
            "TextEditContextMenuItem",
        );
        item_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }
    renderer.set_composite_layer(CompositeLayer::Modal);
}
