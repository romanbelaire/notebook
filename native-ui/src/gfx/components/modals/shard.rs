use super::{render_button, render_modal_container};
use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::Vertex;
use crate::ui::components::Renderable;
use crate::ui::core::{layout, text_input_render, Rect};
use crate::ui::style;

pub(super) fn render_shard_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let modal = &app.shard_modal;

    if modal.shard_id.is_none() {
        return;
    }

    renderer.validate_component("shard_modal", Some("modals"), "ShardModal");
    renderer.push_parent("shard_modal".to_string());

    render_modal_container(modal.position, modal.size, renderer, vertices);

    const PADDING: f32 = 20.0;
    const HEADER_HEIGHT: f32 = 55.0;
    const LABEL_HEIGHT: f32 = 18.0;
    const BUTTON_ROW_HEIGHT: f32 = 36.0;
    const SECTION_SPACING: f32 = 20.0;

    // Header: "Edit Shard"
    let header_rect = Rect::new(
        modal.position.x + PADDING,
        modal.position.y + PADDING,
        modal.size.x - PADDING * 2.0,
        30.0,
    );
    let mut header_title = crate::ui::text::Text::new_for_render("Edit Shard")
        .with_font_size(style::font_size::LARGE)
        .with_color(style::text::PRIMARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    header_title.update_layout(header_rect, None, None);
    renderer.validate_component(
        "shard_modal_header",
        Some("shard_modal"),
        "ShardModalHeader",
    );
    header_title.render(renderer, app, vertices, None);

    // Close button
    render_button(
        &modal.close_button,
        "shard_modal_close",
        "shard_modal",
        renderer,
        app,
        vertices,
    );

    // VStack: user label, user input, assistant label, assistant input, button row (matches ShardModal::update_layout)
    let content_top = modal.position.y + HEADER_HEIGHT;
    let content_height = modal.size.y - HEADER_HEIGHT - PADDING;
    let container = Rect::new(
        modal.position.x + PADDING,
        content_top,
        modal.size.x - PADDING * 2.0,
        content_height,
    );
    let section_heights = [
        LABEL_HEIGHT,
        modal.user_input.size.y,
        LABEL_HEIGHT,
        modal.assistant_input.size.y,
        BUTTON_ROW_HEIGHT,
    ];
    let rects = layout::stack_vertical(&container, &section_heights, SECTION_SPACING, 0.0);

    // User message label
    let mut user_label = crate::ui::text::Text::new_for_render("User message")
        .with_font_size(style::font_size::SMALL)
        .with_color(style::text::SECONDARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    user_label.update_layout(rects[0], None, None);
    user_label.render(renderer, app, vertices, None);

    // User input (position/size set by update_layout from same vstack)
    let mut user_input = modal.user_input.clone();
    user_input.cursor_visible = app.cursor_visible;
    user_input.cursor_animation_value = app.cursor_position_animation.value;
    text_input_render::render_text_input(
        renderer,
        &user_input,
        app,
        vertices,
        Some(style::font_size::NORMAL),
        Some(style::padding::SMALL),
        Some(style::corner_radius::MEDIUM),
        true,
    );

    // Assistant message label
    let mut assistant_label = crate::ui::text::Text::new_for_render("Assistant message")
        .with_font_size(style::font_size::SMALL)
        .with_color(style::text::SECONDARY())
        .with_alignment(crate::ui::text::TextAlignment::Left);
    assistant_label.update_layout(rects[2], None, None);
    assistant_label.render(renderer, app, vertices, None);

    // Assistant input
    let mut assistant_input = modal.assistant_input.clone();
    assistant_input.cursor_visible = app.cursor_visible;
    assistant_input.cursor_animation_value = app.cursor_position_animation.value;
    text_input_render::render_text_input(
        renderer,
        &assistant_input,
        app,
        vertices,
        Some(style::font_size::NORMAL),
        Some(style::padding::SMALL),
        Some(style::corner_radius::MEDIUM),
        true,
    );

    // Save button, Remove from graph button (positions set by update_layout)
    render_button(
        &modal.save_button,
        "shard_modal_save",
        "shard_modal",
        renderer,
        app,
        vertices,
    );
    render_button(
        &modal.remove_from_graph_button,
        "shard_modal_remove_from_graph",
        "shard_modal",
        renderer,
        app,
        vertices,
    );

    renderer.pop_parent();
}
