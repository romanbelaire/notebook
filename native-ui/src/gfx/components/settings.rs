use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use crate::ui::tab_bar::Tab;
use crate::ui::style;
use crate::ui::core::{Rect, container::{SectionStack, Section}};
use crate::ui::{Text, TextAlignment, VStack};
use crate::ui::components::Renderable;

pub fn render_settings(renderer: &mut Renderer, app: &mut App, vertices: &mut Vec<Vertex>) {
    if app.ui_state.active_tab != Tab::Settings {
        return;
    }
    
    // Extract values from app before we start mutating to avoid borrow conflicts
    let cursor_visible = app.cursor_visible;
    let cursor_animation_value = app.cursor_position_animation.value;
    
    // Create immutable reference for render calls before we start mutating
    // We'll use this for all render() calls to avoid borrow conflicts
    let app_ref: &App = unsafe { &*(app as *mut App as *const App) };
    
    // Set parent context for settings components
    // Note: "settings" is already validated by the renderer as a RenderableComponent
    // We just need to push it as parent for child components
    renderer.push_parent("settings".to_string());
    
    if let Some(ref mut settings) = app.settings_window {
        // Background
        let bg = Quad {
            position: settings.position,
            size: settings.size,
            color: style::bg::PRIMARY,
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&bg.to_vertices());

        const PADDING: f32 = 20.0;
        const SECTION_SPACING: f32 = 30.0;
        
        // Create container rect
        let container_rect = Rect::new(
            settings.position.x + PADDING,
            settings.position.y + PADDING,
            settings.size.x - PADDING * 2.0,
            settings.size.y - PADDING * 2.0,
        );

        // Build section stack for settings
        let mut section_stack = SectionStack::new(SECTION_SPACING);
        
        let is_openai = app_ref.settings_state.provider == "openai";
        let model_item_count = if is_openai { 5 } else { 6 };
        let mut model_section = Section::new("Model Settings".to_string(), 40.0);
        model_section.item_count = model_item_count;
        model_section.title_height = 40.0;
        section_stack.add_section(model_section);

        // Generation Settings section
        // We have 2 items: label (25.0) + text (25.0) with 10.0 spacing = 60.0 total
        // Section uses item_count * item_height for content_rect height
        // To get content_rect height = 60.0 with item_count = 2, we need item_height = 30.0
        // This gives us the correct content area height, and stack_vertical will position
        // the actual 25.0-high items correctly within that 60.0-high area
        let mut generation_section = Section::new("Generation Settings".to_string(), 30.0);
        generation_section.item_count = 2; // System prompts label + text
        generation_section.title_height = 40.0;
        section_stack.add_section(generation_section);

        // Personalization section
        // Same: 2 items (25.0 each) with 10.0 spacing = 60.0 total
        let mut personalization_section = Section::new("Personalization".to_string(), 30.0);
        personalization_section.item_count = 2; // Theme label + text
        personalization_section.title_height = 40.0;
        section_stack.add_section(personalization_section);

        // Calculate layout
        let layout = section_stack.layout(&container_rect);
        
        // Render title using Text component (parent is "settings" component)
        // Title is a standalone component, so we create it directly but ensure it's part of hierarchy
        renderer.push_parent("settings_title".to_string());
        renderer.validate_component("settings_title", Some("settings"), "Title");
        let title_rect = Rect::new(container_rect.x, container_rect.y, container_rect.width, 50.0);
        let mut title_text = Text::new_for_render("Settings")
            .with_font_size(style::font_size::LARGE)
            .with_color(style::text::PRIMARY)
            .with_alignment(TextAlignment::Left);
        title_text.update_layout(title_rect, None, None);
        title_text.render(renderer, app_ref, vertices, None);
        renderer.pop_parent();
        
        // Render sections using layout
        let title_offset = 50.0;
        for (section_idx, y_offset) in layout {
            let section = &section_stack.sections[section_idx];
            let section_y_offset = title_offset + y_offset;
            
            // Register section as child of settings BEFORE rendering any children
            let section_parent_id = format!("section_{}", section_idx);
            renderer.validate_component(&section_parent_id, Some("settings"), "Section");
            
            // Render section title using Text component (parent is section name)
            renderer.push_parent(section_parent_id.clone());
            let title_rect = section.title_rect(&container_rect, section_y_offset);
            let mut section_title = Text::new_for_render(&section.title)
                .with_font_size(style::font_size::MEDIUM)
                .with_color(style::text::PRIMARY)
                .with_alignment(TextAlignment::Left);
            // Adjust rect for vertical centering
            let adjusted_title_rect = Rect::new(title_rect.x, title_rect.y + 10.0, title_rect.width, title_rect.height);
            section_title.update_layout(adjusted_title_rect, None, None);
            section_title.render(renderer, app_ref, vertices, None);
            renderer.pop_parent();
            
            // Get content area using section's content_rect method (automatic positioning)
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
                    settings.model_settings_stack.add_text_styled("Provider:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
                    settings.model_settings_stack.add_text_styled(provider_label, style::font_size::NORMAL, style::text::PRIMARY, TextAlignment::Left);
                    if is_openai {
                        settings.model_settings_stack.add_text_styled("OpenAI model:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
                        settings.model_settings_stack.add_child(Box::new(openai_model_input));
                        settings.model_settings_stack.add_text_styled("API key from server environment", style::font_size::SMALL, style::text::SECONDARY, TextAlignment::Left);
                    } else {
                        settings.model_settings_stack.add_text_styled("Model ID:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
                        settings.model_settings_stack.add_child(Box::new(model_id_input));
                        settings.model_settings_stack.add_text_styled("HuggingFace API Key:", style::font_size::NORMAL, style::text::SECONDARY, TextAlignment::Left);
                        settings.model_settings_stack.add_child(Box::new(hf_input));
                    }

                    renderer.push_parent(section_parent_id.clone());
                    settings.model_settings_stack.update_layout(content_area, None, None);
                    settings.model_settings_stack.render(renderer, app_ref, vertices, None);
                    renderer.pop_parent();
                }
                1 => {
                    // Generation Settings section - use persistent VStack
                    // Update layout and render the VStack (which will render its children)
                    renderer.push_parent(section_parent_id.clone());
                    settings.generation_settings_stack.update_layout(content_area, None, None);
                    settings.generation_settings_stack.render(renderer, app_ref, vertices, None);
                    renderer.pop_parent();
                }
                2 => {
                    // Personalization section - use persistent VStack, update theme text
                    let theme_text = match settings.selected_theme {
                        0 => "Standard (Dark Blue)",
                        1 => "Sakura Light",
                        2 => "Springtime Light",
                        3 => "Forest Dark",
                        4 => "Toadstool Light",
                        5 => "Acorn Dark",
                        6 => "Basic Light",
                        7 => "Dark (High Contrast)",
                        _ => "Standard (Dark Blue)",
                    };
                    
                    // Update theme text in VStack (index 1 is the theme value)
                    if settings.personalization_stack.children.len() >= 2 {
                        use crate::ui::text::{Text, TextCreationToken};
                        let token = TextCreationToken::new();
                        let theme_text_component = Text::new_internal(theme_text, token)
                            .with_font_size(style::font_size::NORMAL)
                            .with_color(style::text::PRIMARY)
                            .with_alignment(TextAlignment::Left);
                        settings.personalization_stack.children[1] = Box::new(theme_text_component);
                    }
                    
                    // Update layout and render the VStack (which will render its children)
                    renderer.push_parent(section_parent_id.clone());
                    settings.personalization_stack.update_layout(content_area, None, None);
                    settings.personalization_stack.render(renderer, app_ref, vertices, None);
                    renderer.pop_parent();
                }
                _ => {}
            }
        }
    }
    
    // Pop settings parent
    renderer.pop_parent();
}

