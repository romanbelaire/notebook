use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::tab_bar::Tab;
use crate::ui::style;
use crate::ui::core::Rect;
use crate::ui::{Text, TextAlignment};
use crate::ui::components::{Renderable, VStack};

pub fn render_data(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if app.ui_state.active_tab != Tab::Data {
        return;
    }
    
    // Set parent context for data components
    // Note: "data" is already validated by the renderer as a RenderableComponent
    // We just need to push it as parent for child components
    renderer.push_parent("data".to_string());
    
    if let Some(ref ingest) = app.ingest_window {
        // Background
        let bg = Quad {
            position: ingest.position,
            size: ingest.size,
            color: style::bg::PRIMARY,
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&bg.to_vertices());

        const PADDING: f32 = 20.0;
        const SECTION_SPACING: f32 = 30.0;
        
        // Create container for vertical stacking
        let container = Rect::new(
            ingest.position.x + PADDING,
            ingest.position.y + PADDING,
            ingest.size.x - PADDING * 2.0,
            ingest.size.y - PADDING * 2.0,
        );
        
        // Title - use Text component (standalone)
        renderer.push_parent("data_title".to_string());
        renderer.validate_component("data_title", Some("data"), "Title");
        let title_rect = Rect::new(container.x, container.y, container.width, 40.0);
        let mut title_text = Text::new_for_render("Data Ingestion")
            .with_font_size(style::font_size::LARGE)
            .with_color(style::text::PRIMARY)
            .with_alignment(TextAlignment::Left);
        title_text.update_layout(title_rect, None, None);
        title_text.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // PDF Directory section - use VStack for proper spacing
        let input_section_y = container.y + 40.0 + SECTION_SPACING;
        let input_section_rect = Rect::new(
            container.x,
            input_section_y,
            container.width,
            container.height - (input_section_y - container.y),
        );
        
        // Create VStack with label and input
        let mut input_stack = VStack::new(10.0, 0.0);
        
        // Add label
        input_stack.add_text_styled(
            "PDF Directory:",
            style::font_size::NORMAL,
            style::text::SECONDARY,
            TextAlignment::Left,
        );
        
        // Add text input (clone and update cursor state)
        let mut pdf_input = ingest.pdf_dir_input.clone();
        pdf_input.cursor_visible = app.cursor_visible;
        pdf_input.cursor_animation_value = app.cursor_position_animation.value;
        input_stack.add_child(Box::new(pdf_input));
        
        // Update layout and render the VStack
        // Validate section before rendering VStack
        renderer.validate_component("data_input_section", Some("data"), "DataInputSection");
        renderer.push_parent("data_input_section".to_string());
        input_stack.update_layout(input_section_rect, None, None);
        input_stack.render(renderer, app, vertices, None);
        renderer.pop_parent();
        
        // Calculate button positions relative to input section
        let input_stack_height = input_stack.min_size().y;
        let button_y = input_section_y + input_stack_height + 10.0;

        // Browse button - position after input section
        let browse_rect = crate::ui::core::Rect::new(
            container.x,
            button_y,
            ingest.browse_button.size.x,
            ingest.browse_button.size.y,
        );
        let browse_bg = Quad {
            position: browse_rect.position(),
            size: browse_rect.size(),
            color: match ingest.browse_button.state {
                crate::ui::ButtonState::Pressed => style::button::PRIMARY * Vec4::new(1.0, 1.0, 1.0, 0.8),
                crate::ui::ButtonState::Hover => style::button::PRIMARY * Vec4::new(1.0, 1.0, 1.0, 0.9),
                crate::ui::ButtonState::Normal => style::button::PRIMARY,
            },
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&browse_bg.to_vertices());
        
        let _browse_text_width = renderer.measure_text(&ingest.browse_button.label, style::font_size::NORMAL).x;
        // Render browse button label using Text component
        let browse_text_rect = Rect::new(
            browse_rect.x,
            browse_rect.y,
            browse_rect.width,
            browse_rect.height,
        );
        
        let mut browse_text = crate::ui::text::Text::new_for_render(&ingest.browse_button.label)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY)
            .with_alignment(crate::ui::text::TextAlignment::Center);
        browse_text.update_layout(browse_text_rect, None, None);
        
        renderer.push_parent("data_browse_button".to_string());
        renderer.validate_component("data_browse_button", Some("data"), "BrowseButton");
        browse_text.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // Ingest button - position after browse button
        let ingest_rect = crate::ui::core::Rect::new(
            container.x + ingest.browse_button.size.x + 10.0,
            button_y,
            ingest.ingest_button.size.x,
            ingest.ingest_button.size.y,
        );
        let ingest_bg = Quad {
            position: ingest_rect.position(),
            size: ingest_rect.size(),
            color: if ingest.is_ingesting {
                style::button::PRIMARY * Vec4::new(0.7, 0.7, 0.7, 1.0)
            } else {
                match ingest.ingest_button.state {
                    crate::ui::ButtonState::Pressed => style::button::PRIMARY * Vec4::new(1.0, 1.0, 1.0, 0.8),
                    crate::ui::ButtonState::Hover => style::button::PRIMARY * Vec4::new(1.0, 1.0, 1.0, 0.9),
                    crate::ui::ButtonState::Normal => style::button::PRIMARY,
                }
            },
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&ingest_bg.to_vertices());
        
        // Render ingest button label using Text component
        let ingest_text = if ingest.is_ingesting { "Ingesting..." } else { &ingest.ingest_button.label };
        let ingest_text_rect = Rect::new(
            ingest_rect.x,
            ingest_rect.y,
            ingest_rect.width,
            ingest_rect.height,
        );
        
        let mut ingest_text_component = crate::ui::text::Text::new_for_render(ingest_text)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY)
            .with_alignment(crate::ui::text::TextAlignment::Center);
        ingest_text_component.update_layout(ingest_text_rect, None, None);
        
        renderer.push_parent("data_ingest_button".to_string());
        renderer.validate_component("data_ingest_button", Some("data"), "IngestButton");
        ingest_text_component.render(renderer, app, vertices, None);
        renderer.pop_parent();

        // Status section - use VStack for proper spacing
        let status_section_y = button_y + ingest.ingest_button.size.y + SECTION_SPACING;
        if !ingest.status_text.is_empty() {
            let status_section_rect = Rect::new(
                container.x,
                status_section_y,
                container.width,
                60.0, // Label (25.0) + spacing (10.0) + text (25.0)
            );
            
            let mut status_stack = VStack::new(10.0, 0.0);
            status_stack.add_text_styled(
                "Status:",
                style::font_size::NORMAL,
                style::text::SECONDARY,
                TextAlignment::Left,
            );
            status_stack.add_text_styled(
                &ingest.status_text,
                style::font_size::NORMAL,
                style::text::PRIMARY,
                TextAlignment::Left,
            );
            
            renderer.push_parent("data_status_section".to_string());
            status_stack.update_layout(status_section_rect, None, None);
            status_stack.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }

        // Progress bar (if ingesting)
        if ingest.is_ingesting && ingest.progress > 0.0 {
            let progress_bar_y = if !ingest.status_text.is_empty() {
                status_section_y + 60.0 + SECTION_SPACING
            } else {
                status_section_y
            };
            let progress_bar_rect = Rect::new(
                container.x,
                progress_bar_y,
                container.width,
                20.0,
            );

            // Progress bar background
            let progress_bg = Quad {
                position: progress_bar_rect.position(),
                size: progress_bar_rect.size(),
                color: style::bg::SECONDARY,
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
            slider_effect: false,
            };
            vertices.extend_from_slice(&progress_bg.to_vertices());

            // Progress bar fill
            let progress_fill_width = progress_bar_rect.width * ingest.progress.min(1.0);
            let progress_fill = Quad {
                position: progress_bar_rect.position(),
                size: Vec2::new(progress_fill_width, progress_bar_rect.height),
                color: style::button::PRIMARY,
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
            slider_effect: false,
            };
            vertices.extend_from_slice(&progress_fill.to_vertices());

            // Progress percentage text - use Text component
            renderer.push_parent("data_progress_text".to_string());
            renderer.validate_component("data_progress_text", Some("data"), "ProgressText");
            let progress_text = format!("{:.0}%", ingest.progress * 100.0);
            let mut progress_text_component = Text::new_for_render(&progress_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::PRIMARY)
                .with_alignment(TextAlignment::Center);
            progress_text_component.update_layout(progress_bar_rect, None, None);
            progress_text_component.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }

        // Drag & drop zone hint (positioned at bottom)
        let drop_zone_rect = Rect::new(
            ingest.position.x + PADDING,
            ingest.position.y + ingest.size.y - 100.0,
            ingest.size.x - PADDING * 2.0,
            80.0,
        );
        
        let drop_zone_bg = Quad {
            position: drop_zone_rect.position(),
            size: drop_zone_rect.size(),
            color: style::bg::SECONDARY,
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&drop_zone_bg.to_vertices());
        
        // Drop hint - use Text component
        renderer.push_parent("data_drop_hint".to_string());
        renderer.validate_component("data_drop_hint", Some("data"), "DropHint");
        let mut drop_hint_text = Text::new_for_render("Drag & drop PDF files here to ingest")
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::SECONDARY)
            .with_alignment(TextAlignment::Center);
        drop_hint_text.update_layout(drop_zone_rect, None, None);
        drop_hint_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }
    
    // Pop data parent
    renderer.pop_parent();
}

