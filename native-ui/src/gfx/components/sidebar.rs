use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::style;

pub fn render_sidebar(renderer: &mut Renderer, app: &App, _vertices: &mut Vec<Vertex>) {
    // Sidebar background
    // Calculate translation offset to match content animation
    let open_width = crate::ui::sidebar::SidebarWindow::OPEN_WIDTH;
    let closed_width = 1.0; // CLOSED_WIDTH
    let width_delta = open_width - app.sidebar.current_width;
    let translation_offset = -width_delta; // Negative = move left as sidebar collapses
    
    let sidebar_position = app.sidebar.position + Vec2::new(translation_offset, 0.0);
    let sidebar_quad = Quad {
        position: sidebar_position,
        size: Vec2::new(app.sidebar.current_width, app.sidebar.height),
        color: style::bg::SECONDARY,
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    
    // Use explicit scissor rect to ensure proper rendering
    // Create a rect that matches the sidebar bounds
    use crate::ui::core::Rect;
    let sidebar_rect = Rect::new(
        sidebar_position.x,
        sidebar_position.y,
        app.sidebar.current_width,
        app.sidebar.height,
    );
    renderer.add_quad(&sidebar_quad, Some(&sidebar_rect));
}

