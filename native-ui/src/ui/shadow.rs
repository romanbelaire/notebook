//! Lighting primitives for Renderable components (shadows + highlights).
//!
//! Every primitive shares one off-screen **key light at `-45°`** (upper-left). Each primitive
//! is rendered as **one quad** that runs through a dedicated sentinel branch of
//! [`ui_shader.wgsl`](../../gfx/shaders/ui_shader.wgsl) — no extra render passes, no textures.
//! The fragment shader evaluates closed-form analytical formulas (erf of a rounded-box SDF for
//! shadows; SDF-gradient Lambert for specular highlights) at O(1) per pixel.
//!
//! | Primitive | Branch | Renderer method | Where it paints |
//! |-----------|--------|-----------------|-----------------|
//! | Outer drop shadow       | `bubble == 3.0` | [`queue_shadow`](../../gfx/renderer.rs)           | Bottom-right *outside* the component |
//! | Inner drop shadow       | `bubble == 4.0` | [`queue_inner_shadow`](../../gfx/renderer.rs)     | Top-left *inside* the component      |
//! | Specular border rim     | `bubble == 5.0` | [`queue_border_highlight`](../../gfx/renderer.rs) | Top-left inner edge of the component |
//! | Specular surface sheen  | `bubble == 6.0` | [`queue_surface_highlight`](../../gfx/renderer.rs)| Top-left diagonal across the surface |
//!
//! See the `Elevation` row in [`docs/design/cassette-futurism.md`](../../docs/design/cassette-futurism.md).

use glam::{Vec2, Vec4};
use std::sync::RwLock;

/// Opt-in drop shadow for any [`crate::ui::components::Renderable`].
///
/// - `offset` — shift of the shadow quad relative to the component (positive y = shadow falls downward).
/// - `sigma` — gaussian blur standard deviation in pixels. The shadow quad is inflated by
///   `3 * sigma + spread` so the full falloff is visible.
/// - `color` — RGBA. Alpha scales the overall shadow opacity.
/// - `spread` — outward dilation of the shadow *before* blur (extra hard edge around the source shape).
#[derive(Debug, Clone, Copy)]
pub struct ShadowSpec {
    pub offset: Vec2,
    pub sigma: f32,
    pub color: Vec4,
    pub spread: f32,
}

impl ShadowSpec {
    /// Soft, centered shadow with no offset or spread.
    pub fn soft(sigma: f32, color: Vec4) -> Self {
        Self {
            offset: Vec2::ZERO,
            sigma,
            color,
            spread: 0.0,
        }
    }

    /// Shadow with a pixel offset (typical drop shadow).
    pub fn offset_xy(dx: f32, dy: f32, sigma: f32, color: Vec4) -> Self {
        Self {
            offset: Vec2::new(dx, dy),
            sigma,
            color,
            spread: 0.0,
        }
    }

    /// Builder: set spread (outward dilation before blur).
    pub fn with_spread(mut self, spread: f32) -> Self {
        self.spread = spread;
        self
    }

    /// Builder: override color.
    pub fn with_color(mut self, color: Vec4) -> Self {
        self.color = color;
        self
    }

    /// Builder: override offset.
    pub fn with_offset(mut self, offset: Vec2) -> Self {
        self.offset = offset;
        self
    }
}

/// Inner drop shadow: shadow cast **inside** the component, on the side *facing* the key light
/// (top-left interior). Rendered by [`crate::gfx::renderer::Renderer::queue_inner_shadow`] via
/// the `bubble == 4.0` branch of `ui_shader.wgsl`.
///
/// Physically this reads as the component being a *depression* in the surface: the inner lip on
/// the light-facing side blocks the key light, darkening the top-left interior. Same convention
/// as CSS `box-shadow: inset` with positive offsets.
#[derive(Debug, Clone, Copy)]
pub struct InnerShadowSpec {
    /// Offset magnitude in pixels at `+135°` (bottom-right). Controls how far the shadow
    /// penetrates inward from the edge before fully fading.
    pub offset_size: f32,
    /// Gaussian feather (σ) of the shadow falloff.
    pub sigma: f32,
    /// RGBA tint (alpha scales overall opacity).
    pub color: Vec4,
}

impl InnerShadowSpec {
    pub fn new(offset_size: f32, sigma: f32, color: Vec4) -> Self {
        Self { offset_size, sigma, color }
    }
}

/// Specular border highlight: a bright rim along the component's inner edge, brightest where
/// the edge faces the key light (top-left) and fading to zero at the bottom-right. Rendered by
/// [`crate::gfx::renderer::Renderer::queue_border_highlight`] via the `bubble == 5.0` branch.
///
/// The shader reconstructs the surface normal from the rounded-box SDF gradient and applies a
/// Lambert term `max(0, dot(-L, n))` with `L = (+√½, +√½)`, then modulates by a gaussian rim
/// pulse peaking on the inside of the boundary.
#[derive(Debug, Clone, Copy)]
pub struct BorderHighlightSpec {
    /// Thickness of the highlight band (pixels, measured inward from the edge).
    pub width: f32,
    /// Feather / anti-aliasing softness of the outer edge of the rim.
    pub sigma: f32,
    /// Highlight color. Typically a bright, low-alpha warm white.
    pub color: Vec4,
}

impl BorderHighlightSpec {
    pub fn new(width: f32, sigma: f32, color: Vec4) -> Self {
        Self { width, sigma, color }
    }
}

/// Specular surface highlight: a diagonal sheen across the component's interior, brightest at
/// the top-left and falling off toward the bottom-right — fakes a subtle bevel/curve under the
/// global key light. Rendered by [`crate::gfx::renderer::Renderer::queue_surface_highlight`]
/// via the `bubble == 6.0` branch.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceHighlightSpec {
    /// Falloff curve exponent. `1.0` = linear diagonal, `>1` concentrates the highlight near
    /// the top-left, `<1` spreads it. Typical values: 1.5–3.0.
    pub curve: f32,
    /// AA feather at the shape edge so the sheen hugs rounded corners cleanly.
    pub sigma: f32,
    /// Sheen color. Usually a low-alpha warm white.
    pub color: Vec4,
}

impl SurfaceHighlightSpec {
    pub fn new(curve: f32, sigma: f32, color: Vec4) -> Self {
        Self { curve, sigma, color }
    }
}

/// Interior-mutable shadow slot for stateless singleton viewports
/// (see `native-ui/src/gfx/components/*`). Provides the same "with_shadow" opt-in API
/// that stateful [`crate::ui::components::Renderable`] components get via a struct field.
pub struct ViewportShadow(RwLock<Option<ShadowSpec>>);

impl ViewportShadow {
    pub const fn new() -> Self {
        Self(RwLock::new(None))
    }

    /// Set or clear the shadow. Passing `None` disables the shadow for this viewport.
    pub fn set(&self, spec: Option<ShadowSpec>) {
        *self.0.write().unwrap() = spec;
    }

    /// Current shadow spec (copy, since [`ShadowSpec`] is [`Copy`]).
    pub fn get(&self) -> Option<ShadowSpec> {
        *self.0.read().unwrap()
    }
}
