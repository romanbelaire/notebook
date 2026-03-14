use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::style;
use crate::ui::core::{Rect, layout, text_input_render};
use crate::ui::components::Renderable;
use crate::api::models::Insight;

/// Render backdrop overlay for modals
fn render_backdrop(viewport_size: Vec2, opacity: f32, vertices: &mut Vec<Vertex>) {
    let backdrop = Quad {
        position: Vec2::ZERO,
        size: viewport_size,
        color: Vec4::new(0.0, 0.0, 0.0, opacity),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&backdrop.to_vertices());
}

/// Render modal container (centered window with rounded corners)
fn render_modal_container(
    position: Vec2,
    size: Vec2,
    _renderer: &mut Renderer,
    vertices: &mut Vec<Vertex>,
) {
    // Modal background
    let modal_bg = Quad {
        position,
        size,
        color: style::bg::PRIMARY,
        corner_radius: style::corner_radius::LARGE,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&modal_bg.to_vertices());
    
    // Note: Border rendering would require outline shader support
    // For now, we'll skip the border and rely on shadow effect from background
}

/// Render a button with proper styling
fn render_button(
    button: &crate::ui::Button,
    button_id: &str,
    parent_context: &str,
    renderer: &mut Renderer,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    let button_rect = Rect::from_pos_size(button.position, button.size);
    
    // Button background color based on state
    let bg_color = match button.state {
        crate::ui::ButtonState::Pressed => style::button::PRIMARY * Vec4::new(1.0, 1.0, 1.0, 0.8),
        crate::ui::ButtonState::Hover => style::button::PRIMARY * Vec4::new(1.0, 1.0, 1.0, 0.9),
        crate::ui::ButtonState::Normal => style::button::PRIMARY,
    };
    
    let button_bg = Quad {
        position: button.position,
        size: button.size,
        color: bg_color,
        corner_radius: style::corner_radius::MEDIUM,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&button_bg.to_vertices());
    
    // Check if button label matches an icon name, render as icon if so
    use crate::ui::icons::icon_names;
    let icon_name = match button.label.as_str() {
        "Close" | "✕" => Some(icon_names::CLOSE),
        "Add" | "+" => Some(icon_names::PLUS),
        "Delete" | "Trash" => Some(icon_names::TRASH),
        "Edit" => Some(icon_names::PENCIL),
        "Search" => Some(icon_names::MAGNIFY),
        _ => None,
    };
    
    if let Some(icon) = icon_name {
        // Render icon centered in button
        let icon_size = 16.0;
        let icon_pos = Vec2::new(
            button_rect.x + button_rect.width / 2.0 - icon_size / 2.0,
            button_rect.y + button_rect.height / 2.0 - icon_size / 2.0,
        );
        renderer.queue_icon(icon, icon_pos, icon_size, style::text::PRIMARY);
    } else {
        // Button text (centered) using Text component
        let text_rect = Rect::from_pos_size(button_rect.position(), button_rect.size());
        let mut button_text = crate::ui::text::Text::new_for_render(&button.label)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY)
            .with_alignment(crate::ui::text::TextAlignment::Center);
        button_text.update_layout(text_rect, None, None);
        
        renderer.push_parent(button_id.to_string());
        renderer.validate_component(button_id, Some(parent_context), "ModalButton");
        button_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }
}

pub fn render_modals(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let viewport_size = app.viewport_size;
    
    // Validate and push "modals" as the root parent for all modal components
    // "root" is added to hierarchy before rendering (see renderer.rs line 1569-1571)
    renderer.validate_component("modals", Some("root"), "Modals");
    renderer.push_parent("modals".to_string());
    
    // Render InsightModal
    if app.insight_modal.is_open {
        render_backdrop(viewport_size, 0.5, vertices);
        render_insight_modal(renderer, app, vertices);
    }
    
    // Render PdfModal
    if app.pdf_modal.is_open {
        render_backdrop(viewport_size, 0.7, vertices);
        render_pdf_modal(renderer, app, vertices);
    }
    
    // Render ChatInfoDialog
    if app.chat_info_dialog.is_open {
        render_backdrop(viewport_size, 0.4, vertices);
        render_chat_info_dialog(renderer, app, vertices);
    }
    
    // Render NotepadModal (from NotepadWindow if open)
    if let Some(ref notepad) = app.notepad_window {
        if notepad.notepad_modal.is_open {
            render_backdrop(viewport_size, 0.6, vertices);
            render_notepad_modal_from_window(renderer, notepad, app, vertices);
        }
    }
    
    // Render ShardModal (edit shard messages)
    if app.shard_modal.is_open {
        render_backdrop(viewport_size, 0.5, vertices);
        render_shard_modal(renderer, app, vertices);
    }
    
    // Pop "modals" parent
    renderer.pop_parent();
}

fn render_shard_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
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
        .with_color(style::text::PRIMARY)
        .with_alignment(crate::ui::text::TextAlignment::Left);
    header_title.update_layout(header_rect, None, None);
    renderer.validate_component("shard_modal_header", Some("shard_modal"), "ShardModalHeader");
    header_title.render(renderer, app, vertices, None);

    // Close button
    render_button(&modal.close_button, "shard_modal_close", "shard_modal", renderer, app, vertices);

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
        .with_color(style::text::SECONDARY)
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
        .with_color(style::text::SECONDARY)
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
    render_button(&modal.save_button, "shard_modal_save", "shard_modal", renderer, app, vertices);
    render_button(&modal.remove_from_graph_button, "shard_modal_remove_from_graph", "shard_modal", renderer, app, vertices);

    renderer.pop_parent();
}

fn render_insight_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let modal = &app.insight_modal;
    
    if modal.insight.is_none() {
        return;
    }
    
    let insight = modal.insight.as_ref().unwrap();
    
    // Validate and push "insight_modal" as parent for all components in this modal
    renderer.validate_component("insight_modal", Some("modals"), "InsightModal");
    renderer.push_parent("insight_modal".to_string());
    
    // Render modal container
    render_modal_container(modal.position, modal.size, renderer, vertices);
    
    const PADDING: f32 = 20.0;
    const HEADER_HEIGHT: f32 = 60.0;
    
    // Create container for vertical stacking
    let container = Rect::new(
        modal.position.x + PADDING,
        modal.position.y + PADDING,
        modal.size.x - PADDING * 2.0,
        modal.size.y - PADDING * 2.0 - 50.0, // Reserve space for footer
    );
    
    // Build vertical stack: header, content label, content area
    let mut section_heights = vec![HEADER_HEIGHT, 20.0]; // Header and "Content:" label
    
    // Add content area height
    if modal.is_editing_text {
        section_heights.push(modal.text_input.size.y); // Text input area
        section_heights.push(30.0); // Save button
    } else {
        section_heights.push(200.0); // Content display area
        section_heights.push(30.0); // Edit button
    }
    
    // Stack sections vertically
    let section_rects = layout::stack_vertical(&container, &section_heights, PADDING, 0.0);
    
    // Header with title and close button
    let header_rect = section_rects[0];
    if modal.is_editing_title {
        // Title input field - use standard text input rendering
        let mut title_input = modal.title_input.clone();
        title_input.text = modal.draft_title.clone();
        title_input.position = header_rect.position();
        title_input.size = Vec2::new(header_rect.width, header_rect.height);
        text_input_render::render_text_input(
            renderer,
            &title_input,
            app,
            vertices,
            Some(style::font_size::XLARGE),
            None,
            None,
            false,
        );
    } else {
        // Title display with edit button using Text component
        let title_text = if insight.title.is_empty() { "Insight" } else { &insight.title };
        let title_rect = Rect::new(
            header_rect.x,
            header_rect.y,
            header_rect.width,
            header_rect.height,
        );
        
        let mut title_text_component = crate::ui::text::Text::new_for_render(title_text)
            .with_font_size(style::font_size::XLARGE)
            .with_color(style::text::PRIMARY)
            .with_alignment(crate::ui::text::TextAlignment::Left);
        title_text_component.update_layout(title_rect, None, None);
        
        renderer.push_parent("insight_modal_title".to_string());
        renderer.validate_component("insight_modal_title", Some("modals"), "InsightModalTitle");
        title_text_component.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }
    
    // Close button
    render_button(&modal.close_button, "insight_modal_close_button", "insight_modal", renderer, app, vertices);
    
    // Content section label using Text component
    let content_label_rect = section_rects[1];
    let label_rect = Rect::from_pos_size(content_label_rect.position(), content_label_rect.size());
    
    let mut content_label = crate::ui::text::Text::new_for_render("Content:")
        .with_font_size(style::font_size::SMALL)
        .with_color(style::text::SECONDARY)
        .with_alignment(crate::ui::text::TextAlignment::Left);
    content_label.update_layout(label_rect, None, None);
    
    renderer.push_parent("insight_modal_content_label".to_string());
    renderer.validate_component("insight_modal_content_label", Some("modals"), "InsightModalContentLabel");
    content_label.render(renderer, app, vertices, None);
    renderer.pop_parent();
    
    if modal.is_editing_text {
        // Text input area - use standard text input rendering
        let content_rect = section_rects[2];
        let mut text_input_field = modal.text_input.clone();
        text_input_field.text = modal.draft_text.clone();
        text_input_field.position = content_rect.position();
        text_input_field.size = content_rect.size();
        text_input_render::render_text_input(
            renderer,
            &text_input_field,
            app,
            vertices,
            None,
            None,
            None,
            false,
        );
        
        // Save/Cancel buttons
        render_button(&modal.save_button, "insight_modal_save_button", "insight_modal", renderer, app, vertices);
    } else {
        // Display markdown content (simplified - just show text for now)
        let content_rect = section_rects[2];
        let content_bg = Quad {
            position: content_rect.position(),
            size: content_rect.size(),
            color: style::bg::SECONDARY,
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&content_bg.to_vertices());
        
        // Render text with word wrapping
        let text_pos = Vec2::new(content_rect.x + style::padding::SMALL, content_rect.y + style::padding::SMALL);
        let max_width = content_rect.width - style::padding::SMALL * 2.0;
        let words: Vec<&str> = insight.text.split_whitespace().collect();
        let mut line = String::new();
        let mut y_offset = 0.0;
        let mut line_index = 0;
        
        for word in words {
            let test_line = if line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", line, word)
            };
            
            let test_width = renderer.measure_text(&test_line, style::font_size::NORMAL).x;
            
            if test_width > max_width {
                if !line.is_empty() {
                    // Render line using Text component
                    let line_rect = Rect::new(
                        text_pos.x,
                        text_pos.y + y_offset,
                        max_width,
                        18.0,
                    );
                    
                    let mut line_text = crate::ui::text::Text::new_for_render(&line)
                        .with_font_size(style::font_size::NORMAL)
                        .with_color(style::text::PRIMARY)
                        .with_alignment(crate::ui::text::TextAlignment::Left);
                    line_text.update_layout(line_rect, None, None);
                    
                    let component_id = format!("insight_modal_content_line_{}", line_index);
                    renderer.push_parent(component_id.clone());
                    renderer.validate_component(&component_id, Some("modals"), "InsightModalContentLine");
                    line_text.render(renderer, app, vertices, None);
                    renderer.pop_parent();
                    
                    y_offset += 18.0;
                    line_index += 1;
                    line.clear();
                }
                line = word.to_string();
            } else {
                line = test_line;
            }
        }
        
        if !line.is_empty() {
            // Render line using Text component
            let line_rect = Rect::new(
                text_pos.x,
                text_pos.y + y_offset,
                max_width,
                18.0,
            );
            
            let mut line_text = crate::ui::text::Text::new_for_render(&line)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            line_text.update_layout(line_rect, None, None);
            
            let component_id = format!("insight_modal_content_line_{}", line_index);
            renderer.push_parent(component_id.clone());
            renderer.validate_component(&component_id, Some("modals"), "InsightModalContentLine");
            line_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
        
        // Edit button
        let edit_button_rect = section_rects[3];
        let edit_button = crate::ui::Button::new(
            edit_button_rect.position(),
            Vec2::new(60.0, 30.0),
            "Edit",
        );
        render_button(&edit_button, "insight_modal_edit_button", "insight_modal", renderer, app, vertices);
    }
    
    // Footer buttons
    let footer_y = modal.position.y + modal.size.y - 50.0;
    render_button(&modal.delete_button, "insight_modal_delete_button", "insight_modal", renderer, app, vertices);
    
    // Close button in footer (if not already rendered in header)
    let close_footer = crate::ui::Button::new(
        Vec2::new(modal.position.x + modal.size.x - 100.0, footer_y),
        Vec2::new(80.0, 30.0),
        "Close",
    );
    render_button(&close_footer, "insight_modal_close_footer", "insight_modal", renderer, app, vertices);
    
    // Pop "insight_modal" parent
    renderer.pop_parent();
}

fn render_pdf_modal(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
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
        color: style::bg::SECONDARY,
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
            .with_color(style::text::PRIMARY)
            .with_alignment(crate::ui::text::TextAlignment::Left);
        filename_text.update_layout(filename_rect, None, None);
        
        renderer.push_parent("pdf_modal_filename".to_string());
        renderer.validate_component("pdf_modal_filename", Some("modals"), "PdfModalFilename");
        filename_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }
    
    // Close button
    render_button(&modal.close_button, "pdf_modal_close_button", "pdf_modal", renderer, app, vertices);
    
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
            .with_color(style::text::SECONDARY)
            .with_alignment(crate::ui::text::TextAlignment::Center);
        loading_text_component.update_layout(loading_rect, None, None);
        
        renderer.push_parent("pdf_modal_loading".to_string());
        renderer.validate_component("pdf_modal_loading", Some("modals"), "PdfModalLoading");
        loading_text_component.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else if let Some(page) = modal.pdf_renderer.get_page(modal.current_page as usize) {
        // Render PDF page content
        let text_padding = 10.0;
        let mut y_offset = content_area.y + text_padding;
        let line_height = 18.0;
        let max_width = content_area.width - text_padding * 2.0;
        
        // Render text content with word wrapping
        let words: Vec<&str> = page.text_content.split_whitespace().collect();
        let mut current_line = String::new();
        let mut line_index = 0;
        
        for word in words {
            let test_line = if current_line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", current_line, word)
            };
            
            let test_width = renderer.measure_text(&test_line, style::font_size::NORMAL).x;
            
            if test_width > max_width && !current_line.is_empty() {
                // Render line using Text component
                let line_rect = Rect::new(
                    content_area.x + text_padding,
                    y_offset,
                    max_width,
                    line_height,
                );
                
                let mut line_text = crate::ui::text::Text::new_for_render(&current_line)
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(style::text::PRIMARY)
                    .with_alignment(crate::ui::text::TextAlignment::Left);
                line_text.update_layout(line_rect, None, None);
                
                let component_id = format!("pdf_modal_content_line_{}", line_index);
                renderer.push_parent(component_id.clone());
                renderer.validate_component(&component_id, Some("modals"), "PdfModalContentLine");
                line_text.render(renderer, app, vertices, None);
                renderer.pop_parent();
                
                y_offset += line_height;
                line_index += 1;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
            
            // Stop if we've run out of space
            if y_offset > content_area.bottom() - line_height {
                break;
            }
        }
        
        if !current_line.is_empty() && y_offset <= content_area.bottom() {
            // Render line using Text component
            let line_rect = Rect::new(
                content_area.x + text_padding,
                y_offset,
                max_width,
                line_height,
            );
            
            let mut line_text = crate::ui::text::Text::new_for_render(&current_line)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            line_text.update_layout(line_rect, None, None);
            
            let component_id = format!("pdf_modal_content_line_{}", line_index);
            renderer.push_parent(component_id.clone());
            renderer.validate_component(&component_id, Some("modals"), "PdfModalContentLine");
            line_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
        
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
                .with_color(style::text::SECONDARY)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            page_counter_text.update_layout(page_counter_rect, None, None);
            
            renderer.push_parent("pdf_modal_page_counter".to_string());
            renderer.validate_component("pdf_modal_page_counter", Some("modals"), "PdfModalPageCounter");
            page_counter_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
    } else if modal.filename.is_some() {
        // Show status message or error
        let (status_text, text_color) = if let Some(ref error) = modal.error {
            (error.as_str(), Vec4::new(0.9, 0.3, 0.3, 1.0))  // Red for errors
        } else if modal.loading {
            ("Loading PDF...", style::text::SECONDARY)
        } else {
            ("PDF content not available. Please wait for the file to load.", style::text::SECONDARY)
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
                .with_color(style::text::SECONDARY)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            page_counter_text.update_layout(page_counter_rect, None, None);
            
            renderer.push_parent("pdf_modal_page_counter".to_string());
            renderer.validate_component("pdf_modal_page_counter", Some("modals"), "PdfModalPageCounter");
            page_counter_text.render(renderer, app, vertices, None);
            renderer.pop_parent();
        }
    }
    
    // Navigation buttons
    let _footer_y = modal.position.y + modal.size.y - 50.0;
    render_button(&modal.prev_page_button, "pdf_modal_prev_page", "pdf_modal", renderer, app, vertices);
    render_button(&modal.next_page_button, "pdf_modal_next_page", "pdf_modal", renderer, app, vertices);
    
    // Pop "pdf_modal" parent
    renderer.pop_parent();
}

fn render_chat_info_dialog(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let modal = &app.chat_info_dialog;
    
    if modal.conversation_id.is_none() {
        return;
    }
    
    // Validate and push "chat_info_dialog" as parent for all components in this modal
    renderer.validate_component("chat_info_dialog", Some("modals"), "ChatInfoDialog");
    renderer.push_parent("chat_info_dialog".to_string());
    
    // Render modal container
    render_modal_container(modal.position, modal.size, renderer, vertices);
    
    const PADDING: f32 = 20.0;
    
    // Create container for vertical stacking
    let container = Rect::new(
        modal.position.x + PADDING,
        modal.position.y + PADDING,
        modal.size.x - PADDING * 2.0,
        modal.size.y - PADDING * 2.0 - 50.0, // Reserve space for footer
    );
    
    // Build vertical stack: header, citations label, citations list, insights label, insights list
    let header_height = 50.0;
    let label_height = 30.0;
    let citations_list_height = modal.citations_list.size.y;
    let insights_list_height = modal.insights_list.size.y;
    
    let section_heights = vec![
        header_height,
        label_height,
        citations_list_height,
        label_height,
        insights_list_height,
    ];
    
    // Stack sections vertically
    let section_rects = layout::stack_vertical(&container, &section_heights, PADDING, 0.0);
    
    // Header with title
    let header_rect = section_rects[0];
    if modal.is_editing_title {
        // Title input field - use standard text input rendering
        let mut title_input = modal.title_input.clone();
        title_input.text = modal.draft_title.clone();
        title_input.position = header_rect.position();
        title_input.size = Vec2::new(header_rect.width, header_rect.height);
        text_input_render::render_text_input(
            renderer,
            &title_input,
            app,
            vertices,
            Some(style::font_size::XLARGE),
            None,
            None,
            false,
        );
    } else {
        // Title display using Text component
        let title_rect = Rect::new(
            header_rect.x,
            header_rect.y,
            header_rect.width,
            header_rect.height,
        );
        
        let mut title_text = crate::ui::text::Text::new_for_render(&modal.draft_title)
            .with_font_size(style::font_size::XLARGE)
            .with_color(style::text::PRIMARY)
            .with_alignment(crate::ui::text::TextAlignment::Left);
        title_text.update_layout(title_rect, None, None);
        
        renderer.push_parent("chat_info_dialog_title".to_string());
        renderer.validate_component("chat_info_dialog_title", Some("chat_info_dialog"), "ChatInfoDialogTitle");
        title_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    }
    
    // Close button
    render_button(&modal.close_button, "chat_info_dialog_close_button", "chat_info_dialog", renderer, app, vertices);
    
    // Citations section label using Text component
    let citations_label_rect = section_rects[1];
    let label_rect = Rect::from_pos_size(citations_label_rect.position(), citations_label_rect.size());
    
    let mut citations_label = crate::ui::text::Text::new_for_render("Citations:")
        .with_font_size(style::font_size::MEDIUM)
        .with_color(style::text::PRIMARY)
        .with_alignment(crate::ui::text::TextAlignment::Left);
    citations_label.update_layout(label_rect, None, None);
    
    renderer.push_parent("chat_info_dialog_citations_label".to_string());
    renderer.validate_component("chat_info_dialog_citations_label", Some("chat_info_dialog"), "ChatInfoDialogCitationsLabel");
    citations_label.render(renderer, app, vertices, None);
    renderer.pop_parent();
    
    // Mode toggle button
    let mode_text = match modal.citation_mode {
        crate::ui::CitationMode::All => "Show Unique",
        crate::ui::CitationMode::Unique => "Show All",
    };
    // Create a temporary button with the correct label
    let mode_button = crate::ui::Button::new(
        modal.mode_toggle_button.position,
        modal.mode_toggle_button.size,
        mode_text,
    );
    render_button(&mode_button, "chat_info_dialog_mode_toggle", "chat_info_dialog", renderer, app, vertices);
    
    // Citations list area
    let citations_rect = section_rects[2];
    let citations_bg = Quad {
        position: citations_rect.position(),
        size: citations_rect.size(),
        color: style::bg::SECONDARY,
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&citations_bg.to_vertices());
    
    // Get citations from conversation
    let all_citations: Vec<crate::ui::chat_window::Citation> = if let Some(conv_id) = &modal.conversation_id {
        app.chat_state.conversations
            .iter()
            .find(|c| c.id == *conv_id)
            .map(|c| {
                // Get citations directly from shards
                c.shards
                    .iter()
                    .filter_map(|s| {
                        if matches!(s.metadata.role, crate::ui::chat_window::MessageRole::Assistant) {
                            Some(s.metadata.citations.iter().cloned())
                        } else {
                            None
                        }
                    })
                    .flatten()
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    
    // Filter citations based on mode
    let citations: Vec<crate::ui::chat_window::Citation> = match modal.citation_mode {
        crate::ui::CitationMode::All => all_citations,
        crate::ui::CitationMode::Unique => {
            let mut seen = std::collections::HashSet::new();
            all_citations.into_iter().filter(|cit| {
                let key = format!("{}:{}", cit.source, cit.title.as_ref().unwrap_or(&String::new()));
                seen.insert(key)
            }).collect()
        }
    };
    
    // Render citations with full details
    if citations.is_empty() {
        let no_citations_rect = Rect::new(
            citations_rect.x + PADDING,
            citations_rect.y + PADDING,
            citations_rect.width - PADDING * 2.0,
            30.0,
        );
        
        let mut no_citations_text = crate::ui::text::Text::new_for_render("No citations")
            .with_font_size(style::font_size::SMALL)
            .with_color(style::text::SECONDARY)
            .with_alignment(crate::ui::text::TextAlignment::Left);
        no_citations_text.update_layout(no_citations_rect, None, None);
        
        renderer.push_parent("chat_info_dialog_no_citations".to_string());
        renderer.validate_component("chat_info_dialog_no_citations", Some("chat_info_dialog"), "ChatInfoDialogNoCitations");
        no_citations_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else {
        // Render citations list
        let item_height = 25.0;
        let scroll_offset = modal.citations_list.scroll_offset;
        let mut y_offset = citations_rect.y + PADDING - scroll_offset;
        
        for (i, citation) in citations.iter().enumerate() {
            if y_offset + item_height < citations_rect.y {
                y_offset += item_height;
                continue;
            }
            if y_offset > citations_rect.y + citations_rect.height {
                break;
            }
            
            // Format citation: Title (Source, Year) – Section, p.Page
            let mut citation_text = String::new();
            if let Some(ref title) = citation.title {
                citation_text.push_str(title);
            }
            citation_text.push_str(" (");
            citation_text.push_str(&citation.source);
            if let Some(ref year) = citation.year {
                citation_text.push_str(", ");
                citation_text.push_str(year);
            }
            citation_text.push(')');
            if let Some(ref section) = citation.section {
                citation_text.push_str(" – ");
                citation_text.push_str(section);
            }
            if let Some(page) = citation.page {
                citation_text.push_str(&format!(", p.{}", page));
            }
            
            let citation_item_rect = Rect::new(
                citations_rect.x + PADDING,
                y_offset,
                citations_rect.width - PADDING * 2.0 - 25.0, // Space for magnify icon
                item_height,
            );
            
            let mut citation_text_component = crate::ui::text::Text::new_for_render(&citation_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::PRIMARY)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            citation_text_component.update_layout(citation_item_rect, None, None);
            
            let component_id = format!("chat_info_citation_{}", i);
            renderer.push_parent(component_id.clone());
            renderer.validate_component(&component_id, Some("modals"), "ChatInfoCitation");
            citation_text_component.render(renderer, app, vertices, None);
            renderer.pop_parent();
            
            // Render magnify icon
            use crate::ui::icons::icon_names;
            let icon_pos = Vec2::new(
                citations_rect.x + citations_rect.width - 30.0,
                y_offset + item_height / 2.0 - 7.0,
            );
            renderer.queue_icon(
                icon_names::MAGNIFY,
                icon_pos,
                14.0,
                style::text::SECONDARY,
            );
            
            y_offset += item_height;
        }
        
        // Update scroll content height
        let _total_height = citations.len() as f32 * item_height + PADDING * 2.0;
        // Note: We can't directly mutate modal here, but the scroll view should handle this
    }
    
    // Insights section label using Text component
    let insights_label_rect = section_rects[3];
    let label_rect = Rect::from_pos_size(insights_label_rect.position(), insights_label_rect.size());
    
    let mut insights_label = crate::ui::text::Text::new_for_render("Pinned Insights:")
        .with_font_size(style::font_size::MEDIUM)
        .with_color(style::text::PRIMARY)
        .with_alignment(crate::ui::text::TextAlignment::Left);
    insights_label.update_layout(label_rect, None, None);
    
    renderer.push_parent("chat_info_dialog_insights_label".to_string());
    renderer.validate_component("chat_info_dialog_insights_label", Some("chat_info_dialog"), "ChatInfoDialogInsightsLabel");
    insights_label.render(renderer, app, vertices, None);
    renderer.pop_parent();
    
    // Insights list area
    let insights_rect = Rect::new(
        section_rects[4].x,
        section_rects[4].y,
        section_rects[4].width,
        section_rects[4].height,
    );
    let insights_bg = Quad {
        position: insights_rect.position(),
        size: insights_rect.size(),
        color: style::bg::SECONDARY,
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&insights_bg.to_vertices());
    
    // Get insights for this conversation
    let conversation_insights: Vec<&Insight> = if let Some(conv_id) = &modal.conversation_id {
        // Match insights by checking if their text matches any message content
        if let Some(conv) = app.chat_state.conversations.iter().find(|c| c.id == *conv_id) {
            let message_texts: std::collections::HashSet<String> = conv.shards
                .iter()
                .map(|s| s.text.clone())
                .collect();
            
            app.insights_state.insights
                .iter()
                .filter(|insight| message_texts.contains(&insight.text))
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    
    // Render insights list
    if conversation_insights.is_empty() {
        let no_insights_rect = Rect::new(
            insights_rect.x + PADDING,
            insights_rect.y + PADDING,
            insights_rect.width - PADDING * 2.0,
            30.0,
        );
        
        let mut no_insights_text = crate::ui::text::Text::new_for_render("No pinned insights from this chat")
            .with_font_size(style::font_size::SMALL)
            .with_color(style::text::SECONDARY)
            .with_alignment(crate::ui::text::TextAlignment::Left);
        no_insights_text.update_layout(no_insights_rect, None, None);
        
        renderer.push_parent("chat_info_dialog_no_insights".to_string());
        renderer.validate_component("chat_info_dialog_no_insights", Some("chat_info_dialog"), "ChatInfoDialogNoInsights");
        no_insights_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else {
        let item_height = 30.0;
        let scroll_offset = modal.insights_list.scroll_offset;
        let mut y_offset = insights_rect.y + PADDING - scroll_offset;
        
        for (i, insight) in conversation_insights.iter().enumerate() {
            if y_offset + item_height < insights_rect.y {
                y_offset += item_height;
                continue;
            }
            if y_offset > insights_rect.y + insights_rect.height {
                break;
            }
            
            let display_text = if !insight.title.is_empty() {
                if insight.title.len() > 60 {
                    format!("{}...", &insight.title[..60])
                } else {
                    insight.title.clone()
                }
            } else if insight.text.len() > 60 {
                format!("{}...", &insight.text[..60])
            } else {
                insight.text.clone()
            };
            
            let insight_item_rect = Rect::new(
                insights_rect.x + PADDING,
                y_offset,
                insights_rect.width - PADDING * 2.0,
                item_height,
            );
            
            let mut insight_text_component = crate::ui::text::Text::new_for_render(&display_text)
                .with_font_size(style::font_size::SMALL)
                .with_color(style::text::PRIMARY)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            insight_text_component.update_layout(insight_item_rect, None, None);
            
            let component_id = format!("chat_info_insight_{}", i);
            renderer.push_parent(component_id.clone());
            renderer.validate_component(&component_id, Some("chat_info_dialog"), "ChatInfoInsight");
            insight_text_component.render(renderer, app, vertices, None);
            renderer.pop_parent();
            
            y_offset += item_height;
        }
    }
    
    // Footer buttons
    let footer_y = modal.position.y + modal.size.y - 50.0;
    render_button(&modal.delete_button, "chat_info_dialog_delete_button", "chat_info_dialog", renderer, app, vertices);
    
    let close_footer = crate::ui::Button::new(
        Vec2::new(modal.position.x + modal.size.x - 100.0, footer_y),
        Vec2::new(80.0, 30.0),
        "Close",
    );
    render_button(&close_footer, "chat_info_dialog_close_footer", "chat_info_dialog", renderer, app, vertices);
    
    // Pop "chat_info_dialog" parent
    renderer.pop_parent();
}

fn render_notepad_modal_from_window(renderer: &mut Renderer, notepad: &crate::ui::NotepadWindow, app: &App, vertices: &mut Vec<Vertex>) {
    let modal = &notepad.notepad_modal;
    
    // Validate and push "notepad_modal" as parent for all components in this modal
    renderer.validate_component("notepad_modal", Some("modals"), "NotepadModal");
    renderer.push_parent("notepad_modal".to_string());
    
    // Render modal container
    render_modal_container(modal.position, modal.size, renderer, vertices);
    
    const PADDING: f32 = 20.0;
    const HEADER_HEIGHT: f32 = 50.0;
    
    // Header
    let header_bg = Quad {
        position: Vec2::new(modal.position.x, modal.position.y),
        size: Vec2::new(modal.size.x, HEADER_HEIGHT),
        color: style::bg::SECONDARY,
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&header_bg.to_vertices());
    
    // Header title using Text component
    let header_title_rect = Rect::new(
        modal.position.x + PADDING,
        modal.position.y,
        modal.size.x - PADDING * 2.0,
        HEADER_HEIGHT,
    );
    
    let mut header_title = crate::ui::text::Text::new_for_render("Notepads")
        .with_font_size(style::font_size::LARGE)
        .with_color(style::text::PRIMARY)
        .with_alignment(crate::ui::text::TextAlignment::Left);
    header_title.update_layout(header_title_rect, None, None);
    
    renderer.push_parent("notepad_modal_title".to_string());
    renderer.validate_component("notepad_modal_title", Some("modals"), "NotepadModalTitle");
    header_title.render(renderer, app, vertices, None);
    renderer.pop_parent();
    
    // Close button
    render_button(&modal.close_button, "notepad_modal_close_button", "notepad_modal", renderer, app, vertices);
    
    // Notes list area
    let list_rect = Rect::from_pos_size(modal.papers_list.position, modal.papers_list.size);
    let list_bg = Quad {
        position: list_rect.position(),
        size: list_rect.size(),
        color: style::bg::SECONDARY,
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&list_bg.to_vertices());
    
    // Render notes
    if modal.filtered_papers.is_empty() {
        // Empty state using Text component
        let empty_rect = Rect::new(
            list_rect.x + PADDING,
            list_rect.y + PADDING,
            list_rect.width - PADDING * 2.0,
            30.0,
        );
        
        let mut empty_text = crate::ui::text::Text::new_for_render("No notepads found.")
            .with_font_size(style::font_size::SMALL)
            .with_color(style::text::SECONDARY)
            .with_alignment(crate::ui::text::TextAlignment::Left);
        empty_text.update_layout(empty_rect, None, None);
        
        renderer.push_parent("notepad_modal_empty".to_string());
        renderer.validate_component("notepad_modal_empty", Some("modals"), "NotepadModalEmpty");
        empty_text.render(renderer, app, vertices, None);
        renderer.pop_parent();
    } else {
        let item_height = 40.0;
        let scroll_offset = modal.papers_list.scroll_offset;
        let mut y_offset = PADDING - scroll_offset;
        
        for (idx, paper) in modal.filtered_papers.iter().enumerate() {
            if y_offset + item_height < 0.0 {
                y_offset += item_height;
                continue;
            }
            if y_offset > list_rect.height {
                break;
            }
            
            let item_rect = Rect::new(
                list_rect.x,
                list_rect.y + y_offset,
                list_rect.width,
                item_height,
            );
            
            // Highlight selected item
            if Some(idx) == modal.selected_paper_index {
                let highlight = Quad {
                    position: item_rect.position(),
                    size: item_rect.size(),
                    color: style::highlight::SELECTION,
                    corner_radius: 0.0,
                    bubble_effect: false,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&highlight.to_vertices());
            }
            
            // Paper title using Text component
            let display_name = paper.title.as_ref().unwrap_or(&paper.filename);
            let title_rect = Rect::new(
                item_rect.x + PADDING,
                item_rect.y,
                item_rect.width - PADDING * 2.0,
                item_rect.height,
            );
            
            let mut paper_title = crate::ui::text::Text::new_for_render(display_name)
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY)
                .with_alignment(crate::ui::text::TextAlignment::Left);
            paper_title.update_layout(title_rect, None, None);
            
            renderer.push_parent("notepad_modal_paper_title".to_string());
            renderer.validate_component("notepad_modal_paper_title", Some("modals"), "NotepadModalPaperTitle");
            paper_title.render(renderer, app, vertices, None);
            renderer.pop_parent();
            
            y_offset += item_height;
        }
    }
    
    // Footer buttons
    let footer_y = modal.position.y + modal.size.y - 50.0;
    render_button(&modal.import_button, "notepad_modal_import_button", "notepad_modal", renderer, app, vertices);
    render_button(&modal.delete_button, "notepad_modal_delete_button", "notepad_modal", renderer, app, vertices);
    
    // Pop "notepad_modal" parent
    renderer.pop_parent();
}

