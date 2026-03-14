use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::tab_bar::Tab;
use crate::ui::style;
use crate::ui::core::{Rect, text_input_render};
use crate::ui::{Text, TextAlignment};
use crate::ui::components::Renderable;
use crate::stylus::renderer::StylusRenderer;

pub fn render_notepad(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if app.ui_state.active_tab != Tab::Notepad {
        return;
    }
    
    // Parent "notepad" is already pushed by NotepadComponent::render
    if let Some(ref notepad) = app.notepad_window {
        // Background
        let bg = Quad {
            position: notepad.position,
            size: notepad.size,
            color: style::bg::PRIMARY,
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&bg.to_vertices());
        
        const PADDING: f32 = 20.0;
        
        // Render title input
        renderer.validate_component("notepad_title_input", Some("notepad"), "TitleInput");
        renderer.push_parent("notepad_title_input".to_string());
        text_input_render::render_text_input(renderer, &notepad.title_input, app, vertices, None, None, None, false);
        renderer.pop_parent();
        
        // Render CRUD buttons
        renderer.validate_component("notepad_buttons", Some("notepad"), "ButtonGroup");
        renderer.push_parent("notepad_buttons".to_string());
        render_button(&notepad.new_button, "new_button", "notepad_buttons", renderer, app, vertices);
        render_button(&notepad.save_button, "save_button", "notepad_buttons", renderer, app, vertices);
        render_button(&notepad.open_button, "open_button", "notepad_buttons", renderer, app, vertices);
        render_button(&notepad.delete_button, "delete_button", "notepad_buttons", renderer, app, vertices);
        renderer.pop_parent();
        
        // Render toolbar
        renderer.validate_component("notepad_toolbar", Some("notepad"), "Toolbar");
        renderer.push_parent("notepad_toolbar".to_string());
        render_toolbar(renderer, &notepad.toolbar, app, vertices);
        renderer.pop_parent();
        
        // Editor area - use the editor's position from update_layout (which accounts for title)
        // The editor position is already set correctly in NotepadWindow::update_layout()
        let editor_area = Rect::new(
            notepad.editor.position.x,
            notepad.editor.position.y,
            notepad.editor.size.x,
            notepad.editor.size.y,
        );
        
        // Render block backgrounds with focus highlighting
        let padding = PADDING;
        let block_spacing = 8.0;
        let mut y_offset = editor_area.y - notepad.editor.scroll_offset;
        
        for block in &notepad.editor.document.blocks {
            let block_height = StylusRenderer::get_block_height_static(block, notepad.editor.size.x - padding * 2.0);
            
            // Skip blocks that are off-screen
            if y_offset + block_height < notepad.editor.position.y {
                y_offset += block_height + block_spacing;
                continue;
            }
            if y_offset > notepad.editor.position.y + notepad.editor.size.y {
                break;
            }
            
            // Check if this block is focused
            let is_focused = notepad.editor.focused_block_id.as_ref()
                .map(|id| id == &block.id)
                .unwrap_or(false);
            
            // Render block background
            let block_bg = Quad {
                position: Vec2::new(notepad.editor.position.x, y_offset),
                size: Vec2::new(notepad.editor.size.x, block_height),
                color: if is_focused {
                    style::bg::SECONDARY
                } else {
                    Vec4::new(0.0, 0.0, 0.0, 0.0) // Transparent for unfocused blocks
                },
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
            slider_effect: false,
            };
            vertices.extend_from_slice(&block_bg.to_vertices());
            
            y_offset += block_height + block_spacing;
        }
        
        // Render editor blocks using StylusRenderer
        StylusRenderer::render_blocks(renderer, &notepad.editor, app.mouse_pos, app, vertices);
    }
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
    
    // Button text (centered) using Text component
    let text_rect = Rect::from_pos_size(button_rect.position(), button_rect.size());
    let mut button_text = Text::new_for_render(&button.label)
        .with_font_size(style::font_size::NORMAL)
        .with_color(style::text::PRIMARY)
        .with_alignment(TextAlignment::Center);
    button_text.update_layout(text_rect, None, None);
    
    renderer.validate_component(button_id, Some(parent_context), "NotepadButton");
    renderer.push_parent(button_id.to_string());
    button_text.render(renderer, app, vertices, None);
    renderer.pop_parent();
}

/// Render toolbar with all formatting buttons
fn render_toolbar(
    renderer: &mut Renderer,
    toolbar: &crate::ui::Toolbar,
    app: &App,
    vertices: &mut Vec<Vertex>,
) {
    // Toolbar background
    let toolbar_bg = Quad {
        position: toolbar.position,
        size: toolbar.size,
        color: style::bg::SECONDARY,
        corner_radius: style::corner_radius::SMALL,
        bubble_effect: false,
            slider_effect: false,
    };
    vertices.extend_from_slice(&toolbar_bg.to_vertices());
    
    // Render each toolbar button
    render_button(&toolbar.bold_button, "toolbar_bold", "notepad_toolbar", renderer, app, vertices);
    render_button(&toolbar.italic_button, "toolbar_italic", "notepad_toolbar", renderer, app, vertices);
    render_button(&toolbar.underline_button, "toolbar_underline", "notepad_toolbar", renderer, app, vertices);
    render_button(&toolbar.strikethrough_button, "toolbar_strikethrough", "notepad_toolbar", renderer, app, vertices);
    render_button(&toolbar.code_button, "toolbar_code", "notepad_toolbar", renderer, app, vertices);
    render_button(&toolbar.link_button, "toolbar_link", "notepad_toolbar", renderer, app, vertices);
}

