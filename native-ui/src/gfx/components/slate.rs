use crate::app::App;
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use crate::ui::style;
use crate::ui::text::Text;
use crate::ui::text::TextAlignment;
use glam::Vec2;

const PANEL_W: f32 = 200.0;
const ROW_H: f32 = 22.0;
const MARGIN: f32 = 8.0;
const MAX_VISIBLE: usize = 12;

/// Right-side paste board list (Slate).
pub struct SlateViewport;

pub const SLATE_VIEWPORT: SlateViewport = SlateViewport;

pub fn render_slate(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if !app.slate.visible || app.slate.entries.is_empty() {
        return;
    }
    renderer.validate_component("slate_viewport", Some("root"), "SlateViewport");
    renderer.push_parent("slate".to_string());
    renderer.set_composite_layer(CompositeLayer::HudChrome);

    let vy = app.viewport_size.y;
    let vx = app.viewport_size.x;
    let x0 = vx - PANEL_W - MARGIN;
    let n = app.slate.entries.len().min(MAX_VISIBLE);
    let h = n as f32 * ROW_H + 36.0;
    let y0 = 80.0_f32;
    if y0 + h > vy {
        // clip to viewport
    }

    let panel = Quad {
        position: Vec2::new(x0, y0),
        size: Vec2::new(PANEL_W, h.min(vy - y0 - MARGIN)),
        color: style::bg::PANEL_POPUP(),
        corner_radius: style::corner_radius::MEDIUM,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&panel.to_vertices());

    let title_rect = Rect::new(x0 + 8.0, y0 + 6.0, PANEL_W - 16.0, 20.0);
    let mut title = Text::new_for_render("Slate")
        .with_font_size(style::font_size::SMALL)
        .with_color(style::text::SECONDARY())
        .with_alignment(TextAlignment::Left);
    title.update_layout(title_rect, None, None);
    title.render(renderer, app, vertices, None);

    for (i, entry) in app.slate.entries.iter().take(MAX_VISIBLE).enumerate() {
        let preview = if entry.preview.len() > 42 {
            format!("{}…", &entry.preview[..42])
        } else {
            entry.preview.clone()
        };
        let row_y = y0 + 28.0 + i as f32 * ROW_H;
        let row_rect = Rect::new(x0 + 8.0, row_y, PANEL_W - 16.0, ROW_H - 2.0);
        let mut line = Text::new_for_render(&preview)
            .with_font_size(style::font_size::SMALL)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Left);
        line.update_layout(row_rect, None, None);
        line.render(renderer, app, vertices, None);
    }

    renderer.pop_parent();
}

impl Renderable for SlateViewport {
    fn z_order(&self) -> i32 {
        89
    }

    fn render(
        &self,
        renderer: &mut Renderer,
        app: &App,
        vertices: &mut Vec<Vertex>,
        _dirty_rect: Option<Rect>,
    ) {
        render_slate(renderer, app, vertices);
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
