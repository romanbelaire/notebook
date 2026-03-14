use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::Vec2;

pub fn render_sidebar_glow(_renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if let Some(glow_y) = app.sidebar_edge_glow_position {
        if app.sidebar_edge_glow_intensity > 0.05 {
            let header_height = app.header.size.y;
            let glow_center_x = if app.sidebar.is_open {
                app.sidebar.current_width
            } else {
                0.0
            };
            
            // Glow ellipsoid - vertical orientation (tall and narrow)
            // Height is 1/3 of window height (excluding header)
            let window_content_height = app.viewport_size.y - header_height;
            let glow_width = 40.0 * app.sidebar_edge_glow_intensity;  // Narrow horizontal
            let glow_height = (window_content_height / 3.0) * app.sidebar_edge_glow_intensity; // 1/3 of window height
            // Allow glow to slip under header naturally (no clamping)
            let clamped_glow_y = glow_y;
            
            let glow_quad = Quad {
                position: Vec2::new(glow_center_x - glow_width / 2.0, clamped_glow_y - glow_height / 2.0),
                size: Vec2::new(glow_width, glow_height),
                color: glam::Vec4::new(0.4, 0.6, 0.9, app.sidebar_edge_glow_intensity * 0.3),
                corner_radius: glow_width / 2.0,  // Half the smaller dimension for ellipse effect
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&glow_quad.to_vertices());
        }
    }
}

