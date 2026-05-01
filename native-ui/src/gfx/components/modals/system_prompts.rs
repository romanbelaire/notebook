use super::{render_button, render_modal_container};
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::Vertex;
use crate::ui::components::Renderable;
use crate::ui::core::text_input_render;
use crate::ui::style;
use crate::ui::{Text, TextAlignment};

pub fn render_system_prompts_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if !app.system_prompts_modal.is_open {
        return;
    }

    renderer.validate_component("system_prompts_modal", Some("modals"), "SystemPromptsModal");
    renderer.push_parent("system_prompts_modal".to_string());

    let m = &app.system_prompts_modal;
    render_modal_container(m.position, m.size, renderer, vertices);

    let title_rect = crate::ui::core::Rect::new(
        m.position.x + 20.0,
        m.position.y + 18.0,
        m.size.x - 40.0,
        36.0,
    );
    let mut title = Text::new_for_render("System prompts")
        .with_font_size(style::font_size::LARGE)
        .with_color(style::text::PRIMARY())
        .with_alignment(TextAlignment::Left);
    title.update_layout(title_rect, None, None);
    renderer.push_parent("system_prompts_modal_title".to_string());
    renderer.validate_component(
        "system_prompts_modal_title",
        Some("modals"),
        "SystemPromptsTitle",
    );
    title.render(renderer, app, vertices, None);
    renderer.pop_parent();

    let list_y = m.position.y + 64.0;
    let row_h = 28.0;
    for (i, p) in m.prompts.iter().enumerate().take(12) {
        let row_rect = crate::ui::core::Rect::new(
            m.position.x + 20.0,
            list_y + i as f32 * row_h,
            m.size.x - 120.0,
            row_h,
        );
        let line = format!("• {}", p.name);
        let mut t = Text::new_for_render(&line)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Left);
        t.update_layout(row_rect, None, None);
        renderer.push_parent(format!("system_prompt_row_{}", i));
        t.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }

    let mut name_in = m.name_input.clone();
    name_in.cursor_visible = app.cursor_visible;
    name_in.cursor_animation_value = app.cursor_position_animation.value;
    let mut content_in = m.content_input.clone();
    content_in.cursor_visible = app.cursor_visible;
    content_in.cursor_animation_value = app.cursor_position_animation.value;

    renderer.push_parent("system_prompts_modal_inputs".to_string());
    text_input_render::render_text_input(
        renderer,
        &name_in,
        app,
        vertices,
        Some(style::font_size::NORMAL),
        Some(style::padding::MEDIUM),
        Some(style::corner_radius::MEDIUM),
        false,
    );
    text_input_render::render_text_input(
        renderer,
        &content_in,
        app,
        vertices,
        Some(style::font_size::NORMAL),
        Some(style::padding::MEDIUM),
        Some(style::corner_radius::MEDIUM),
        false,
    );
    renderer.pop_parent();

    render_button(
        &m.close_button,
        "system_prompts_close",
        "system_prompts_modal",
        renderer,
        app,
        vertices,
    );
    render_button(
        &m.save_button,
        "system_prompts_save",
        "system_prompts_modal",
        renderer,
        app,
        vertices,
    );

    for (i, db) in m.delete_buttons.iter().enumerate() {
        render_button(
            db,
            &format!("system_prompts_del_{}", i),
            "system_prompts_modal",
            renderer,
            app,
            vertices,
        );
    }

    renderer.pop_parent();
}
