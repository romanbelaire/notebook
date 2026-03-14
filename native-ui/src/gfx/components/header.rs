use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::Vec4;
use crate::ui::style;
use crate::ui::core::{Rect, layout};
use crate::ui::{Text, TextAlignment};
use crate::ui::components::Renderable;

pub fn render_header(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    // Note: "header" is already validated and pushed by HeaderComponent before calling this function
    // We don't need to push/pop it here - it's already on the parent stack
    
    const FONT_SIZE: f32 = style::font_size::MEDIUM;
    
    // Header rect
    let header_rect = Rect::from_pos_size(app.header.position, app.header.size);
    
    // Header background
    let header_quad = Quad {
        position: header_rect.position(),
        size: header_rect.size(),
        color: style::bg::SECONDARY,
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&header_quad.to_vertices());

    // Tab bar rect (relative to screen, not header)
    let tab_bar_rect = Rect::from_pos_size(
        app.header.tab_bar.position + app.header.position,
        app.header.tab_bar.size
    );

    // Tab bar background (glow effect)
    let tab_bar_bg = Quad {
        position: tab_bar_rect.position(),
        size: tab_bar_rect.size(),
        color: Vec4::new(0.12, 0.12, 0.15, 0.6),
        corner_radius: 20.0,  // Half of height for glow effect
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&tab_bar_bg.to_vertices());

    // Tab bar slider (active tab highlight). Quad padded by full max_stretch on both sides so leading edge isn't clipped.
    const SLIDER_MAX_STRETCH: f32 = 48.0;
    let slider_x = tab_bar_rect.x + app.header.tab_bar.slider_position();
    let slider_width = app.header.tab_bar.slider_width();
    let slider_rect = Rect::new(
        slider_x - SLIDER_MAX_STRETCH,
        tab_bar_rect.y,
        slider_width + 2.0 * SLIDER_MAX_STRETCH,
        tab_bar_rect.height,
    );
    let slider_quad = Quad {
        position: slider_rect.position(),
        size: slider_rect.size(),
        color: style::highlight::HOVER,
        corner_radius: 18.0,  // Solid button shape (not half of height)
        bubble_effect: false,
        slider_effect: true,
    };
    vertices.extend_from_slice(&slider_quad.to_vertices());

    // Tab labels - use horizontal stack to position tabs
    let tab_width = tab_bar_rect.width / app.header.tab_bar.tabs.len() as f32;
    let tab_widths: Vec<f32> = (0..app.header.tab_bar.tabs.len()).map(|_| tab_width).collect();
    let tab_rects = layout::stack_horizontal(&tab_bar_rect, &tab_widths, 0.0, 0.0);
    
    // Set parent for tab bar components
    // Validate "header_tab_bar" BEFORE pushing it (so it's in hierarchy when children validate)
    renderer.validate_component("header_tab_bar", Some("header"), "TabBar");
    renderer.push_parent("header_tab_bar".to_string());
    for (i, tab) in app.header.tab_bar.tabs.iter().enumerate() {
        let tab_rect = tab_rects[i];
        
        // Tab label - use Text component with unique parent for each tab
        let tab_parent_id = format!("header_tab_{}", i);
        // Validate tab parent BEFORE pushing it (so it's in hierarchy when Text validates)
        renderer.validate_component(&tab_parent_id, Some("header_tab_bar"), "Tab");
        renderer.push_parent(tab_parent_id.clone());
        
        let text_color = if i == app.header.tab_bar.active_index {
            style::text::PRIMARY
        } else {
            style::text::SECONDARY
        };
        let mut tab_text = Text::new_for_render(tab.label())
            .with_font_size(FONT_SIZE)
            .with_color(text_color)
            .with_alignment(TextAlignment::Center);
        tab_text.update_layout(tab_rect, None, None);
        tab_text.render(renderer, app, vertices, None);
        
        renderer.pop_parent();
    }
    renderer.pop_parent();

    // Title "Notebook" - use Text component
    // Position to left of tab bar, vertically centered in header
    // Validate title BEFORE pushing it (so it's in hierarchy when Text validates)
    renderer.validate_component("header_title", Some("header"), "Title");
    renderer.push_parent("header_title".to_string());
    let title_rect = Rect::new(
        header_rect.x + style::padding::SMALL,
        header_rect.y + (header_rect.height - FONT_SIZE * 1.2) / 2.0,
        tab_bar_rect.x - header_rect.x - style::padding::SMALL * 2.0,
        FONT_SIZE * 1.2,
    );
    let mut title_text = Text::new_for_render("Notebook")
        .with_font_size(FONT_SIZE)
        .with_color(style::text::PRIMARY)
        .with_alignment(TextAlignment::Left);
    title_text.update_layout(title_rect, None, None);
    title_text.render(renderer, app, vertices, None);
    renderer.pop_parent();
    
    // Window control buttons (if enabled)
    if app.header.show_window_controls {
        // Close button
        let close_rect = Rect::from_pos_size(app.header.close_button.position, app.header.close_button.size);
        let close_color = match app.header.close_button.state {
            crate::ui::ButtonState::Pressed => style::button::DANGER_ACTIVE,
            crate::ui::ButtonState::Hover => style::button::DANGER_HOVER,
            crate::ui::ButtonState::Normal => style::button::DANGER,
        };
        let close_bg = Quad {
            position: close_rect.position(),
            size: close_rect.size(),
            color: close_color,
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&close_bg.to_vertices());
        // Window control buttons - set parent
        renderer.push_parent("header_window_controls".to_string());
        renderer.validate_component("header_window_controls", None, "WindowControls");
        
        // Close button text - use Text component
        let mut close_text = Text::new_for_render("×")
            .with_font_size(style::font_size::LARGE)
            .with_color(style::text::PRIMARY)
            .with_alignment(TextAlignment::Center);
        close_text.update_layout(close_rect, None, None);
        close_text.render(renderer, app, vertices, None);
        
        // Maximize button
        let max_rect = Rect::from_pos_size(app.header.maximize_button.position, app.header.maximize_button.size);
        let max_color = match app.header.maximize_button.state {
            crate::ui::ButtonState::Pressed => style::button::SECONDARY_ACTIVE,
            crate::ui::ButtonState::Hover => style::button::SECONDARY_HOVER,
            crate::ui::ButtonState::Normal => style::button::SECONDARY,
        };
        let max_bg = Quad {
            position: max_rect.position(),
            size: max_rect.size(),
            color: max_color,
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&max_bg.to_vertices());
        // Maximize button text - use Text component
        let mut max_text = Text::new_for_render("□")
            .with_font_size(FONT_SIZE)
            .with_color(style::text::PRIMARY)
            .with_alignment(TextAlignment::Center);
        max_text.update_layout(max_rect, None, None);
        max_text.render(renderer, app, vertices, None);
        
        // Minimize button
        let min_rect = Rect::from_pos_size(app.header.minimize_button.position, app.header.minimize_button.size);
        let min_color = match app.header.minimize_button.state {
            crate::ui::ButtonState::Pressed => style::button::SECONDARY_ACTIVE,
            crate::ui::ButtonState::Hover => style::button::SECONDARY_HOVER,
            crate::ui::ButtonState::Normal => style::button::SECONDARY,
        };
        let min_bg = Quad {
            position: min_rect.position(),
            size: min_rect.size(),
            color: min_color,
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&min_bg.to_vertices());
        // Minimize button text - use Text component
        let mut min_text = Text::new_for_render("−")
            .with_font_size(FONT_SIZE)
            .with_color(style::text::PRIMARY)
            .with_alignment(TextAlignment::Center);
        min_text.update_layout(min_rect, None, None);
        min_text.render(renderer, app, vertices, None);
        
        renderer.pop_parent();
    }
    
    // Note: "header" parent is popped by HeaderComponent after calling this function
}

