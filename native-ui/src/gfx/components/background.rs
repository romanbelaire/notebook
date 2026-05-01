use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::Vec2;
use crate::ui::style;
use crate::ui::core::Rect;
use crate::ui::components::Renderable;

pub fn render_background(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    renderer.set_composite_layer(crate::gfx::renderer::CompositeLayer::Background);
    let header_height = app.header.size.y;
    let content_height = app.viewport_size.y - header_height;
    
    // Ultra-dark space background with subtle parallax glow
    // Layer 1: Base background (ultra-dark)
    let base_bg = Quad {
        position: Vec2::new(0.0, header_height),
        size: Vec2::new(app.viewport_size.x, content_height),
        color: style::bg::PRIMARY(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&base_bg.to_vertices());

    // Static upper depth band (cockpit canopy read).
    let band_h = (content_height * 0.32).max(1.0);
    let depth_band = Quad {
        position: Vec2::new(0.0, header_height),
        size: Vec2::new(app.viewport_size.x, band_h),
        color: style::backdrop::DEPTH_BAND(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&depth_band.to_vertices());
    
    // Phosphor-tinted parallax haze (mouse-follow)
    let mouse_parallax_x = app.mouse_pos.x / app.viewport_size.x;
    let mouse_parallax_y = app.mouse_pos.y / app.viewport_size.y;
    let parallax_offset_x = (mouse_parallax_x - 0.5) * 20.0;  // Max 20px offset
    let parallax_offset_y = (mouse_parallax_y - 0.5) * 20.0;
    
    let gradient_layer = Quad {
        position: Vec2::new(parallax_offset_x, header_height + parallax_offset_y),
        size: Vec2::new(app.viewport_size.x + 40.0, content_height + 40.0),
        color: style::backdrop::PARALLAX_SLOW(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&gradient_layer.to_vertices());
    
    // Faster parallax layer
    let pattern_offset_x = parallax_offset_x * 1.5;
    let pattern_offset_y = parallax_offset_y * 1.5;
    let pattern_layer = Quad {
        position: Vec2::new(pattern_offset_x, header_height + pattern_offset_y),
        size: Vec2::new(app.viewport_size.x + 60.0, content_height + 60.0),
        color: style::backdrop::PARALLAX_FAST(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&pattern_layer.to_vertices());
}

/// Full-viewport background pass; z-order 0 (behind main UI).
pub struct BackgroundViewport;

pub const BACKGROUND_VIEWPORT: BackgroundViewport = BackgroundViewport;

/// Opt-in drop shadow for the background layer.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for BackgroundViewport {
    fn z_order(&self) -> i32 {
        0
    }

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
        render_background(renderer, app, vertices);
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        Some(Rect::new(0.0, 0.0, app.viewport_size.x, app.viewport_size.y))
    }

    fn update_layout(
        &mut self,
        _available_rect: Rect,
        _dirty_rect: Option<Rect>,
        _app: Option<&App>,
    ) {
    }
}

