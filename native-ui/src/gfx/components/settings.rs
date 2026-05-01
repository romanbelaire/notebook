use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::app::App;
use glam::Vec2;
use crate::ui::tab_bar::Tab;
use crate::ui::style;
use crate::ui::core::{Rect, container::{SectionStack, Section}};
use crate::ui::{Text, TextAlignment, VStack};
use crate::ui::components::Renderable;

pub fn render_settings(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if app.ui_state.active_tab != Tab::Settings {
        return;
    }

    let cursor_visible = app.cursor_visible;
    let cursor_animation_value = app.cursor_position_animation.value;
    let is_openai = app.settings_state.provider == "openai";

    let mut settings_sw = match app.settings_window.borrow_mut().take() {
        Some(s) => s,
        None => return,
    };
    let app_ref: &App = app;

    renderer.push_parent("settings".to_string());

    let settings = &mut settings_sw;

    let bg = Quad {
        position: settings.position,
        size: settings.size,
        color: style::bg::PRIMARY(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    renderer.add_quad(&bg, None);
    renderer.set_composite_layer(CompositeLayer::HudChrome);

    const PADDING: f32 = 20.0;
    const SECTION_SPACING: f32 = 30.0;

    let container_rect = Rect::new(
        settings.position.x + PADDING,
        settings.position.y + PADDING,
        settings.size.x - PADDING * 2.0,
        settings.size.y - PADDING * 2.0,
    );

    let mut section_stack = SectionStack::new(SECTION_SPACING);

    let model_item_count = if is_openai { 5 } else { 6 };
    let mut model_section = Section::new("Model Settings".to_string(), 40.0);
    model_section.item_count = model_item_count;
    model_section.title_height = 40.0;
    section_stack.add_section(model_section);

    let mut generation_section = Section::new("Generation Settings".to_string(), 140.0);
    generation_section.item_count = 1;
    generation_section.title_height = 40.0;
    section_stack.add_section(generation_section);

    let mut personalization_section = Section::new("Personalization".to_string(), 40.0);
    personalization_section.item_count = 2;
    personalization_section.title_height = 40.0;
    section_stack.add_section(personalization_section);

    let layout = section_stack.layout(&container_rect);

    renderer.push_parent("settings_title".to_string());
    renderer.validate_component("settings_title", Some("settings"), "Title");
    let title_rect = Rect::new(container_rect.x, container_rect.y, container_rect.width, 50.0);
    let mut title_text = Text::new_for_render("Settings")
        .with_font_size(style::font_size::LARGE)
        .with_color(style::text::PRIMARY())
        .with_alignment(TextAlignment::Left);
    title_text.update_layout(title_rect, None, None);
    title_text.render(renderer, app_ref, vertices, None);
    renderer.pop_parent();

    let title_offset = 50.0;
    for (section_idx, y_offset) in layout {
        let section = &section_stack.sections[section_idx];
        let section_y_offset = title_offset + y_offset;

        let section_parent_id = format!("section_{}", section_idx);
        renderer.validate_component(&section_parent_id, Some("settings"), "Section");

        renderer.push_parent(section_parent_id.clone());
        let title_rect = section.title_rect(&container_rect, section_y_offset);
        let mut section_title = Text::new_for_render(&section.title)
            .with_font_size(style::font_size::MEDIUM)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Left);
        let adjusted_title_rect = Rect::new(title_rect.x, title_rect.y + 10.0, title_rect.width, title_rect.height);
        section_title.update_layout(adjusted_title_rect, None, None);
        section_title.render(renderer, app_ref, vertices, None);
        renderer.pop_parent();

        let content_area = section.content_rect(&container_rect, section_y_offset);

        match section_idx {
            0 => {
                let provider_label = if is_openai { "OpenAI" } else { "Local model" };
                let mut model_id_input = settings.model_id_input.clone();
                model_id_input.cursor_visible = cursor_visible;
                model_id_input.cursor_animation_value = cursor_animation_value;
                let mut hf_input = settings.hf_token_input.clone();
                hf_input.cursor_visible = cursor_visible;
                hf_input.cursor_animation_value = cursor_animation_value;
                if !hf_input.text.is_empty() {
                    if hf_input.text.len() > 4 {
                        hf_input.text = format!("{}...", &hf_input.text[..4]);
                    } else {
                        hf_input.text = "•".repeat(hf_input.text.len());
                    }
                }
                let mut openai_model_input = settings.openai_model_input.clone();
                openai_model_input.cursor_visible = cursor_visible;
                openai_model_input.cursor_animation_value = cursor_animation_value;

                settings.model_settings_stack = VStack::new(10.0, 0.0);
                settings.model_settings_stack.add_text_styled("Provider:", style::font_size::NORMAL, style::text::SECONDARY(), TextAlignment::Left);
                settings.model_settings_stack.add_text_styled(provider_label, style::font_size::NORMAL, style::text::PRIMARY(), TextAlignment::Left);
                if is_openai {
                    settings.model_settings_stack.add_text_styled("OpenAI model:", style::font_size::NORMAL, style::text::SECONDARY(), TextAlignment::Left);
                    settings.model_settings_stack.add_child(Box::new(openai_model_input));
                    settings.model_settings_stack.add_text_styled("API key from server environment", style::font_size::SMALL, style::text::SECONDARY(), TextAlignment::Left);
                } else {
                    settings.model_settings_stack.add_text_styled("Model ID:", style::font_size::NORMAL, style::text::SECONDARY(), TextAlignment::Left);
                    settings.model_settings_stack.add_child(Box::new(model_id_input));
                    settings.model_settings_stack.add_text_styled("HuggingFace API Key:", style::font_size::NORMAL, style::text::SECONDARY(), TextAlignment::Left);
                    settings.model_settings_stack.add_child(Box::new(hf_input));
                }

                renderer.push_parent(section_parent_id.clone());
                settings.model_settings_stack.update_layout(content_area, None, None);
                settings.model_settings_stack.render(renderer, app_ref, vertices, None);
                renderer.pop_parent();
            }
            1 => {
                settings.generation_settings_stack = VStack::new(10.0, 0.0);
                settings.generation_settings_stack.add_text_styled(
                    "Named prompts: type /name in chat, then your message.",
                    style::font_size::NORMAL,
                    style::text::SECONDARY(),
                    TextAlignment::Left,
                );
                settings.generation_settings_stack.add_text_styled(
                    "Example: /concise explain attention",
                    style::font_size::SMALL,
                    style::text::SECONDARY(),
                    TextAlignment::Left,
                );
                renderer.push_parent(section_parent_id.clone());
                settings.generation_settings_stack.update_layout(content_area, None, None);
                settings.generation_settings_stack.render(renderer, app_ref, vertices, None);
                let btn_y = content_area.y + 78.0;
                settings.manage_system_prompts_button.position = Vec2::new(content_area.x, btn_y);
                settings.manage_system_prompts_button.size = Vec2::new(240.0, 36.0);
                use crate::ui::components::Renderable;
                settings.manage_system_prompts_button.render(renderer, app_ref, vertices, None);
                renderer.pop_parent();
            }
            2 => {
                let label_rect = Rect::new(content_area.x, content_area.y, content_area.width, 22.0);
                let mut theme_label = Text::new_for_render("Theme:")
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(style::text::SECONDARY())
                    .with_alignment(TextAlignment::Left);
                theme_label.update_layout(label_rect, None, None);

                let dd_h = 36.0;
                let dd_rect = Rect::new(content_area.x, content_area.y + 26.0, 280.0_f32.min(content_area.width), dd_h);
                settings.theme_dropdown.anchor_rect = dd_rect;
                settings.theme_dropdown.button_size = dd_rect.size();

                let label_text = settings
                    .theme_dropdown
                    .selected_index
                    .and_then(|i| settings.theme_dropdown.items.get(i))
                    .map(|it| it.label.as_str())
                    .unwrap_or("Theme");

                renderer.push_parent(section_parent_id.clone());
                theme_label.render(renderer, app_ref, vertices, None);

                let btn_bg = Quad {
                    position: dd_rect.position(),
                    size: dd_rect.size(),
                    color: style::button::SECONDARY(),
                    corner_radius: style::corner_radius::MEDIUM,
                    bubble_effect: false,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&btn_bg.to_vertices());

                let mut btn_caption = Text::new_for_render(label_text)
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(style::text::PRIMARY())
                    .with_alignment(TextAlignment::Left);
                let cap = dd_rect.inset(10.0);
                btn_caption.update_layout(cap, None, None);
                btn_caption.render(renderer, app_ref, vertices, None);

                use crate::ui::components::Renderable;
                settings.theme_dropdown.render(renderer, app_ref, vertices, None);
                renderer.pop_parent();
            }
            _ => {}
        }
    }

    renderer.pop_parent();
    *app.settings_window.borrow_mut() = Some(settings_sw);
}

/// Stateless shell for settings; delegates to [`render_settings`].
pub struct SettingsViewport;

pub const SETTINGS_VIEWPORT: SettingsViewport = SettingsViewport;

/// Opt-in drop shadow for the settings window chassis.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for SettingsViewport {
    fn render(
        &self,
        renderer: &mut Renderer,
        app: &App,
        vertices: &mut Vec<Vertex>,
        _dirty_rect: Option<Rect>,
    ) {
        if let Some(spec) = SHADOW.get() {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), &spec);
            }
        }
        render_settings(renderer, app, vertices);
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.settings_window
            .borrow()
            .as_ref()
            .map(|w| Rect::new(w.position.x, w.position.y, w.size.x, w.size.y))
    }

    fn update_layout(
        &mut self,
        _available_rect: Rect,
        _dirty_rect: Option<Rect>,
        _app: Option<&App>,
    ) {
    }
}
