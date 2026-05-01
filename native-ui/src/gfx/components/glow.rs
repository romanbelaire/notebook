use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::Renderer;
use crate::app::App;
use glam::Vec2;
use crate::ui::core::Rect;
use crate::ui::components::Renderable;
use crate::ui::style;

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
                color: glam::Vec4::new(
                    style::accent::PHOSPHOR_GLOW().x,
                    style::accent::PHOSPHOR_GLOW().y,
                    style::accent::PHOSPHOR_GLOW().z,
                    app.sidebar_edge_glow_intensity * 0.34,
                ),
                corner_radius: glow_width / 2.0,  // Half the smaller dimension for ellipse effect
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&glow_quad.to_vertices());
        }
    }
}

/// Sidebar edge glow; z-order 5.
pub struct GlowViewport;

pub const GLOW_VIEWPORT: GlowViewport = GlowViewport;

/// Opt-in drop shadow for the glow layer (rare — glow is already a soft light effect).
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for GlowViewport {
    fn z_order(&self) -> i32 {
        5
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
        render_sidebar_glow(renderer, app, vertices);
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

