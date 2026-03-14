use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::style;

pub fn render_background(_renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    let header_height = app.header.size.y;
    let content_height = app.viewport_size.y - header_height;
    
    // Ultra-dark space background with subtle parallax glow
    // Layer 1: Base background (ultra-dark)
    let base_bg = Quad {
        position: Vec2::new(0.0, header_height),
        size: Vec2::new(app.viewport_size.x, content_height),
        color: style::bg::PRIMARY,
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&base_bg.to_vertices());
    
    // Layer 2: Subtle gradient overlay driven by mouse parallax
    let mouse_parallax_x = app.mouse_pos.x / app.viewport_size.x;
    let mouse_parallax_y = app.mouse_pos.y / app.viewport_size.y;
    let parallax_offset_x = (mouse_parallax_x - 0.5) * 20.0;  // Max 20px offset
    let parallax_offset_y = (mouse_parallax_y - 0.5) * 20.0;
    
    let gradient_layer = Quad {
        position: Vec2::new(parallax_offset_x, header_height + parallax_offset_y),
        size: Vec2::new(app.viewport_size.x + 40.0, content_height + 40.0),
        color: Vec4::new(1.0, 1.0, 1.0, 0.03),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&gradient_layer.to_vertices());
    
    // Layer 3: Subtle pattern overlay (fastest movement)
    let pattern_offset_x = parallax_offset_x * 1.5;
    let pattern_offset_y = parallax_offset_y * 1.5;
    let pattern_layer = Quad {
        position: Vec2::new(pattern_offset_x, header_height + pattern_offset_y),
        size: Vec2::new(app.viewport_size.x + 60.0, content_height + 60.0),
        color: Vec4::new(1.0, 1.0, 1.0, 0.02),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&pattern_layer.to_vertices());
}

