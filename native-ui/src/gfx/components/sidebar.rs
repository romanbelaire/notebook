use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::app::App;
use glam::Vec2;
use crate::ui::style;
use crate::ui::core::Rect;
use crate::ui::components::Renderable;

/// Chrome quad position/size in screen space (matches collapse translation in [`render_sidebar`]).
pub fn sidebar_chrome_rect(app: &App) -> Rect {
    let open_width = crate::ui::sidebar::SidebarWindow::OPEN_WIDTH;
    let width_delta = open_width - app.sidebar.current_width;
    let translation_offset = -width_delta;
    let p = app.sidebar.position + Vec2::new(translation_offset, 0.0);
    Rect::new(p.x, p.y, app.sidebar.current_width, app.sidebar.height)
}

pub fn render_sidebar(renderer: &mut Renderer, app: &App, _vertices: &mut Vec<Vertex>) {
    renderer.set_composite_layer(CompositeLayer::SidebarChrome);
    let sidebar_rect = sidebar_chrome_rect(app);

    // Elevation: sidebar chrome reads as a vertical plinth lifted off the background.
    renderer.queue_shadow(&sidebar_rect, 0.0, &style::elevation::MEDIUM());

    let sidebar_quad = Quad {
        position: sidebar_rect.position(),
        size: sidebar_rect.size(),
        color: style::bg::SECONDARY(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    renderer.add_quad(&sidebar_quad, Some(&sidebar_rect));
}

/// Stateless [`Renderable`] for sidebar chrome; delegates to [`render_sidebar`].
pub struct SidebarViewport;

pub const SIDEBAR_VIEWPORT: SidebarViewport = SidebarViewport;

/// Opt-in drop shadow for the sidebar chrome. Call [`crate::ui::shadow::ViewportShadow::set`]
/// with `Some(ShadowSpec)` to enable.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for SidebarViewport {
    fn z_order(&self) -> i32 {
        20
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
        render_sidebar(renderer, app, vertices);
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        Some(sidebar_chrome_rect(app))
    }

    fn update_layout(
        &mut self,
        _available_rect: Rect,
        _dirty_rect: Option<Rect>,
        _app: Option<&App>,
    ) {
    }
}

