use crate::app::App;
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use crate::ui::style;
use crate::ui::text::Text;
use crate::ui::toast::ToastType;
use glam::Vec2;

pub fn render_toasts(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if app.toast_manager.toasts.is_empty() {
        return;
    }

    renderer.validate_component("toasts_viewport", Some("root"), "ToastsViewport");
    renderer.push_parent("toasts".to_string());
    renderer.set_composite_layer(CompositeLayer::HudChrome);

    let font = style::font_size::TOOLTIP;
    let pad = style::toast::CARD_PADDING;
    let w = style::toast::CARD_WIDTH;
    let stripe = style::toast::STRIPE_W;

    for toast in &app.toast_manager.toasts {
        let op = toast.opacity;
        let pos = toast.position;
        let h = toast.height;
        let inner_w = w - 2.0 * pad - stripe;
        let draw_h = h.min(style::toast::CARD_MAX_HEIGHT);
        let text_rect = Rect::new(
            pos.x + pad + stripe + pad * 0.5,
            pos.y + pad,
            inner_w,
            draw_h - pad * 2.0,
        );
        let mut t = Text::new_for_render(toast.message.as_str())
            .with_font_size(font)
            .with_color({
                let mut c = style::text::PRIMARY();
                c.w *= op;
                c
            })
            .with_alignment(crate::ui::text::TextAlignment::Left)
            .with_scissor(Some(text_rect));

        let rim = style::toast::RIM_PAD;
        let mut rim_c = style::border::INSTRUMENT_RULE();
        rim_c.w *= op;
        let rim_quad = Quad {
            position: pos - Vec2::splat(rim),
            size: Vec2::new(w, draw_h) + Vec2::splat(rim * 2.0),
            color: rim_c,
            corner_radius: style::corner_radius::MEDIUM + rim,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&rim_quad.to_vertices());

        let mut panel_c = style::bg::PANEL_POPUP();
        panel_c.w *= op;
        let panel_quad = Quad {
            position: pos,
            size: Vec2::new(w, draw_h),
            color: panel_c,
            corner_radius: style::corner_radius::MEDIUM,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&panel_quad.to_vertices());

        let mut stripe_c = match toast.toast_type {
            ToastType::Error => style::accent::WARNING(),
            ToastType::Success => style::accent::PHOSPHOR(),
            ToastType::Info => style::accent::POP_COOL(),
        };
        stripe_c.w *= op;
        let stripe_quad = Quad {
            position: pos + Vec2::new(pad * 0.5, pad),
            size: Vec2::new(stripe, draw_h - pad * 2.0),
            color: stripe_c,
            corner_radius: style::corner_radius::SMALL * 0.5,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&stripe_quad.to_vertices());

        t.update_layout(text_rect, None, None);
        t.render(renderer, app, vertices, None);
    }

    renderer.pop_parent();
}

/// Toast layer; z-order 90 (below header at 100).
pub struct ToastsViewport;

pub const TOASTS_VIEWPORT: ToastsViewport = ToastsViewport;

/// Opt-in drop shadow for the toast overlay.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for ToastsViewport {
    fn z_order(&self) -> i32 {
        90
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
        render_toasts(renderer, app, vertices);
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
