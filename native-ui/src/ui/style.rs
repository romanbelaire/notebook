//! Centralized UI styling constants and utilities.
//!
//! # Design token roles (cassette futurism)
//!
//! - **Horizon** — Header chrome, tab folder, instrument rule (`stroke::INSTRUMENT_RULE_PX`, `chrome::TAB_*`, `border::INSTRUMENT_RULE`).
//! - **Bulkhead** — Deep shells and wells (`bg::PRIMARY`, `CHROME_DECK`, `PANEL_WELL`, `backdrop`).
//! - **Readout** — Body text and labels on panels (`text`, `font_size`, `bg::INPUT`).
//! - **Impulse** — Warm/cool emphasis (`accent::POP`, `POP_COOL`, `PHOSPHOR`).
//! - **Warning** — Alert / destructive (`accent::WARNING`, `button::DANGER_*`).
//!
//! See [`docs/design/cassette-futurism.md`](../../docs/design/cassette-futurism.md) for motivation and the hero layout reference.
use glam::Vec4;

// ===== SPACING & SIZING =====

/// Standard padding for UI elements
pub mod padding {
    pub const TINY: f32 = 5.0;
    pub const SMALL: f32 = 10.0;
    pub const MEDIUM: f32 = 14.0;
    pub const LARGE: f32 = 18.0;
    pub const XLARGE: f32 = 22.0;
    /// Extra inset between shard card border and message bubbles in constellation view.
    pub const SHARD_MESSAGE_INSET: f32 = SMALL;
}

// ===== HERO LAYOUT (header → viewport → composer) =====

/// Canonical margins and sizes for the main window’s reference surface (tab bar, chat viewport, composer).
pub mod hero {
    /// Horizontal and top “poster” gutter for the main chat viewport and composer deck.
    pub const MAIN_VIEWPORT_GUTTER: f32 = super::padding::LARGE;
    /// Centered folder tab strip width in the header instrument band.
    pub const TAB_BAR_WIDTH: f32 = 420.0;
    /// Tab bar height; keep in sync with [`crate::ui::header::HeaderWindow`] layout.
    pub const TAB_BAR_HEIGHT: f32 = 46.0;
}

/// Transient toast cards stacked from the bottom-right (HUD).
pub mod toast {
    pub const CARD_WIDTH: f32 = 280.0;
    pub const CARD_MIN_HEIGHT: f32 = 52.0;
    pub const CARD_MAX_HEIGHT: f32 = 160.0;
    pub const CARD_PADDING: f32 = super::padding::MEDIUM;
    pub const STACK_GAP: f32 = super::padding::SMALL;
    pub const MARGIN_X: f32 = super::hero::MAIN_VIEWPORT_GUTTER;
    pub const MARGIN_Y: f32 = super::hero::MAIN_VIEWPORT_GUTTER;
    pub const STRIPE_W: f32 = super::stroke::COMPOSER_RULE_PX.min(6.0);
    pub const RIM_PAD: f32 = 2.0;
    pub const HEURISTIC_CHARS_PER_LINE: f32 = 38.0;
}

/// Standard corner radius values
pub mod corner_radius {
    pub const SMALL: f32 = 5.0;
    pub const MEDIUM: f32 = 9.0;
    pub const LARGE: f32 = 14.0;
    pub const PILL: f32 = 9999.0;
}

/// Physical stroke widths (quads, SDF border alpha channel, line thickness).
///
/// Main chrome (header rule, composer deck, tab folder outer rim) uses **thick** instrument/composer
/// values for cassette CRT bezel read.
pub mod stroke {
    pub const INSTRUMENT_RULE_PX: f32 = 8.0;
    pub const COMPOSER_RULE_PX: f32 = 8.0;
    /// SDF ring on folder tab outer frame (`color.w` = line width); cap + shank share this for a continuous rim.
    pub const TAB_BAR_RING_PX: f32 = 8.0;
    /// Inset of folder cavity inside the outer frame (scale with `TAB_BAR_RING_PX`).
    pub const TAB_BAR_INNER_INSET: f32 = 8.0;
    /// Inset of nested dark track pill inside the cavity.
    pub const TAB_BAR_NEST_INSET: f32 = 10.0;
    /// Active tab pill outline.
    pub const TAB_SLIDER_RING_PX: f32 = 4.0;
    /// Inner nested pill outline between cavity and track.
    pub const TAB_TRACK_PILL_RING_PX: f32 = 3.0;
    pub const GRAPH_EDGE_PX: f32 = 6.0;
    /// Number of chord segments used to tessellate one bezier graph edge.
    pub const GRAPH_EDGE_STEPS: usize = 20;
    pub const BUBBLE_RIM_PX: f32 = 6.0;
    pub const SELECTION_OUTLINE_PX: f32 = 6.0;
}

/// Standard font sizes
pub mod font_size {
    pub const TINY: f32 = 10.0;
    pub const SMALL: f32 = 12.0;
    pub const NORMAL: f32 = 14.0;
    pub const MEDIUM: f32 = 16.0;
    pub const MESSAGE_BODY: f32 = MEDIUM;
    pub const LARGE: f32 = 18.0;
    pub const XLARGE: f32 = 20.0;
    pub const TITLE: f32 = 24.0;
    pub const TOOLTIP: f32 = 13.0;
    pub const LINE_HEIGHT_RATIO: f32 = 1.35;
}

/// Empty-state copy blocks (centered title + subtitle in large quiet fields).
pub mod empty_state {
    pub const TITLE_FONT: f32 = super::font_size::XLARGE;
    pub const SUBTITLE_FONT: f32 = super::font_size::NORMAL;
    pub const VERTICAL_GAP: f32 = super::padding::SMALL;
    pub const SIDE_INSET: f32 = super::hero::MAIN_VIEWPORT_GUTTER;
}

/// Standard button heights
pub mod button_height {
    pub const SMALL: f32 = 28.0;
    pub const NORMAL: f32 = 36.0;
    pub const LARGE: f32 = 44.0;
}

/// Standard input field heights
pub mod input_height {
    pub const SMALL: f32 = 32.0;
    pub const NORMAL: f32 = 40.0;
    pub const LARGE: f32 = 48.0;
}

/// Notepad formatting toolbar (instrument strip).
pub mod toolbar_chrome {
    pub const BAR_HEIGHT: f32 = super::input_height::NORMAL;
    pub const BUTTON_EXTENT: f32 = super::input_height::SMALL;
    pub const BUTTON_SPACING: f32 = super::padding::TINY;
}

/// Sidebar section lists (conversations, documents): row rhythm and spacing.
pub mod sidebar_layout {
    pub const SECTION_TITLE_HEIGHT: f32 = 40.0;
    pub const ROW_HEIGHT: f32 = 40.0;
    pub const SECTION_SPACING: f32 = super::padding::LARGE;
    pub const TITLE_AREA_PADDING: f32 = super::padding::MEDIUM;
}

/// Earth-tone palette for procedural graph edge coloring.
///
/// Eight stops that span the full hue wheel at ~45° increments, each desaturated and value-damped
/// into the earth-tone band (S ≈ 40–60 %, V ≈ 55–80 %).  Sampled by ribbon color lookup in the constellation renderer.
pub mod edge_palette {
    use glam::Vec4;

    pub const STOPS: [Vec4; 8] = [
        Vec4::new(0.769, 0.439, 0.290, 1.0), // terracotta  #C4704A  hue ~18°
        Vec4::new(0.769, 0.635, 0.290, 1.0), // ochre/gold  #C4A24A  hue ~40°
        Vec4::new(0.549, 0.651, 0.353, 1.0), // olive       #8CA65A  hue ~80°
        Vec4::new(0.353, 0.580, 0.471, 1.0), // sage-teal   #5A9478  hue ~155°
        Vec4::new(0.290, 0.486, 0.627, 1.0), // slate-blue  #4A7CA0  hue ~205°
        Vec4::new(0.478, 0.420, 0.659, 1.0), // dusty viol  #7A6BA8  hue ~258°
        Vec4::new(0.627, 0.376, 0.502, 1.0), // mauve       #A06080  hue ~320°
        Vec4::new(0.769, 0.408, 0.376, 1.0), // brick       #C46860  hue ~5°
    ];

    /// Sample the palette at a continuous position in [0, 1] using linear interpolation.
    /// Values outside [0, 1] wrap around via modulo so the wheel is seamlessly cyclic.
    pub fn sample(t: f32) -> Vec4 {
        let n = STOPS.len() as f32;
        let scaled = t.rem_euclid(1.0) * n;
        let lo = scaled as usize % STOPS.len();
        let hi = (lo + 1) % STOPS.len();
        let frac = scaled.fract();
        STOPS[lo] * (1.0 - frac) + STOPS[hi] * frac
    }
}

/// Constellation (graph chat) tunable layout constants.
pub mod constellation {
    pub const MIN_FONT: f32 = 4.0;
    pub const MACRO_MIN_TEXT_PT: f32 = 4.0;
    pub const MACRO_NODE_RADIUS_PX: f32 = 3.2;
    pub const MACRO_NODE_HIT_RADIUS_PAD_PX: f32 = 3.0;
    pub const MACRO_SELECTED_RING_PX: f32 = 1.6;
    pub const MACRO_JIGGLE_AMPLITUDE_PX: f32 = 2.2;
    pub const MACRO_JIGGLE_RATE_HZ: f32 = 0.8;
    pub const MACRO_TOOLTIP_WIDTH: f32 = 340.0;
    pub const FAR_SIMPLIFY_THRESHOLD: f32 = 2.5;
    pub const EDGE_INSET: f32 = 1.35;
    pub const EDGE_INSET_MIN: f32 = 0.65;
    /// Corner radius (screen pixels at scale=1) for the rounded-elbow graph connectors.
    pub const EDGE_CORNER_RADIUS: f32 = 30.0;
    /// Gap between adjacent bundled connector lines, in unscaled pixels.
    /// Lines are spaced at `edge_thickness + EDGE_BUNDLE_GAP_PX` so they are tight but never touch.
    pub const EDGE_BUNDLE_GAP_PX: f32 = 3.0;
    pub const SCALE_BUCKET_MULTIPLIER: f32 = 4.0;

    pub const MOVE_HANDLE_HEIGHT: f32 = 14.0;
    pub const RESIZE_HANDLE_SIZE: f32 = 14.0;
    pub const BUBBLE_SPACING: f32 = 8.0;
    pub const BUBBLE_MAX_WIDTH_RATIO: f32 = 0.7;
    pub const BUBBLE_MIN_CONTENT_WIDTH: f32 = 80.0;
    pub const HIDDEN_PLACEHOLDER_HEIGHT: f32 = 20.0;

    pub const ACTION_BUTTON_SIZE: f32 = 18.0;
    pub const ACTION_BUTTON_SPACING: f32 = 4.0;
    pub const ACTION_ROW_PADDING: f32 = 6.0;
    pub const MESSAGE_ACTION_BUTTON_SIZE: f32 = 14.0;
    pub const MESSAGE_ACTION_ICON_SIZE: f32 = 12.0;
    pub const ACTION_ICON_SIZE: f32 = 14.0;
    pub const MESSAGE_ACTION_INSET: f32 = 4.0;
    pub const MESSAGE_HIT_EXPAND: f32 = 2.0;
    pub const BUTTON_ROW_RESERVE: f32 = 22.0;
    /// Vertical gap between agent/user message text (clipped bubble area) and the shard bottom action row.
    pub const SHARD_ACTION_TEXT_CLEARANCE: f32 = 12.0;

    pub const CITATION_GAP: f32 = 4.0;
    pub const NOTE_LINE_HEIGHT: f32 = 18.0;
    pub const NOTE_ICON_WIDTH: f32 = 20.0;
    pub const NOTE_EDIT_RIGHT_OFFSET: f32 = 48.0;
    pub const NOTE_REMOVE_RIGHT_OFFSET: f32 = 24.0;
    pub const NOTE_TEXT_EXTRA_RIGHT_PAD: f32 = 52.0;

    /// Fit four bottom action buttons in `action_area_width`; shrink size and spacing so icons never overlap.
    pub fn fit_shard_action_row(action_area_width: f32, button_size: f32, spacing: f32) -> (f32, f32) {
        let w = action_area_width.max(0.0);
        let mut s = spacing;
        let mut b = button_size.min((w - 3.0 * s).max(0.0) / 4.0).max(1.0);
        if 4.0 * b + 3.0 * s > w {
            s = 0.0;
            b = (w / 4.0).max(1.0).min(button_size);
        }
        (b, s)
    }
}

// ===== COLORS =====
// Values come from [`crate::ui::theme::ThemePalette`], installed per frame via [`install_theme_for_id`].

use std::cell::RefCell;

thread_local! {
    static ACTIVE_PALETTE: RefCell<crate::ui::theme::ThemePalette> =
        RefCell::new(crate::ui::theme::ThemePalette::standard());
}

/// Sync active palette from persisted `SettingsState.theme` before drawing UI for this frame.
pub fn install_theme_for_id(theme_id: &str) {
    let p = crate::ui::theme::ThemePalette::resolve(theme_id);
    ACTIVE_PALETTE.with(|cell| {
        *cell.borrow_mut() = p;
    });
}

/// Accent — bridge readout trim (cream), impulse/warning red, cool blue-violet UI pop (TNG-style deck).
pub mod accent {
    use super::Vec4;

    pub fn PHOSPHOR() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().accent.phosphor)
    }

    pub fn PHOSPHOR_GLOW() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().accent.phosphor_glow)
    }

    pub fn POP() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().accent.pop)
    }

    pub fn POP_COOL() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().accent.pop_cool)
    }

    pub fn WARNING() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().accent.warning)
    }
}

/// Viewport backdrop overlays (parallax haze, static depth band).
pub mod backdrop {
    use super::Vec4;

    pub fn PARALLAX_SLOW() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().backdrop.parallax_slow)
    }

    pub fn PARALLAX_FAST() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().backdrop.parallax_fast)
    }

    pub fn DEPTH_BAND() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().backdrop.depth_band)
    }
}

/// Header tab bar fills (semi-transparent track vs active pill).
pub mod chrome {
    use super::Vec4;

    pub const TAB_FOLDER_TOP_RADIUS: f32 = 14.0;

    pub fn TAB_BAR_BG() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().chrome.tab_bar_bg)
    }

    pub fn TAB_BAR_SLIDER() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().chrome.tab_bar_slider)
    }

    pub fn COMPOSER_BACKPLATE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().chrome.composer_backplate)
    }

    /// Darker RGB than [`COMPOSER_BACKPLATE`] for tab track outline (`Vec4::w` = line width).
    pub fn COMPOSER_BACKPLATE_DIM_RIM() -> Vec4 {
        let c = COMPOSER_BACKPLATE();
        const DIM: f32 = 0.72;
        Vec4::new(
            c.x * DIM,
            c.y * DIM,
            c.z * DIM,
            super::stroke::TAB_TRACK_PILL_RING_PX,
        )
    }
}

/// Background colors
pub mod bg {
    use super::Vec4;

    pub fn PRIMARY() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.primary)
    }

    pub fn CHROME_DECK() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.chrome_deck)
    }

    pub fn SECONDARY() -> Vec4 {
        CHROME_DECK()
    }

    pub fn PANEL_WELL() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.panel_well)
    }

    pub fn TERTIARY() -> Vec4 {
        PANEL_WELL()
    }

    pub fn INPUT() -> Vec4 {
        PANEL_WELL()
    }

    pub fn INPUT_FOCUSED() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.input_focused)
    }

    pub fn PANEL_POPUP() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.panel_popup)
    }

    pub fn USER_MESSAGE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.user_message)
    }

    pub fn ASSISTANT_MESSAGE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.assistant_message)
    }

    pub fn MUTED_MESSAGE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.muted_message)
    }

    pub fn SHARD_BACKPLATE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().bg.shard_backplate)
    }
}

/// Text colors
pub mod text {
    use super::Vec4;

    pub fn PRIMARY() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().text.primary)
    }

    pub fn SECONDARY() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().text.secondary)
    }

    pub fn TERTIARY() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().text.tertiary)
    }

    pub fn PLACEHOLDER() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().text.placeholder)
    }

    pub fn GHOST() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().text.ghost)
    }

    pub fn ACCENT() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().text.accent)
    }
}

/// Button colors
pub mod button {
    use super::Vec4;

    pub fn PRIMARY() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.primary)
    }

    pub fn PRIMARY_HOVER() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.primary_hover)
    }

    pub fn PRIMARY_ACTIVE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.primary_active)
    }

    pub fn SECONDARY() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.secondary)
    }

    pub fn SECONDARY_HOVER() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.secondary_hover)
    }

    pub fn SECONDARY_ACTIVE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.secondary_active)
    }

    pub fn DANGER() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.danger)
    }

    pub fn DANGER_HOVER() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.danger_hover)
    }

    pub fn DANGER_ACTIVE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().button.danger_active)
    }
}

/// Border colors
pub mod border {
    use super::Vec4;

    pub fn SUBTLE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().border.subtle)
    }

    pub fn DEFAULT() -> Vec4 {
        SUBTLE()
    }

    pub fn FOCUSED() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().border.focused)
    }

    pub fn HOVER() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().border.hover)
    }

    pub fn ACCENT() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().border.accent)
    }

    pub fn PHOSPHOR() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().border.phosphor)
    }

    pub fn INSTRUMENT_RULE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().border.instrument_rule)
    }
}

/// Selection and highlight colors
pub mod highlight {
    use super::Vec4;

    pub fn SELECTION() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().highlight.selection)
    }

    pub fn HOVER() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().highlight.hover)
    }

    pub fn ACTIVE() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().highlight.active)
    }
}

/// Markdown readout (inline code, fenced bodies) on dark panels.
pub mod markdown {
    use super::Vec4;

    pub fn CODE_FOREGROUND() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().markdown.code_foreground)
    }
}

/// Stylus / rich editor accents (highlights, selection bands).
pub mod editor {
    use super::Vec4;

    pub fn MATCH_HIGHLIGHT_TEXT() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().editor.match_highlight_text)
    }

    pub fn SELECTION_BAND() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().editor.selection_band)
    }
}

/// Graph / constellation emphasis (pins, alerts).
pub mod graph {
    use super::Vec4;

    pub fn PIN_ICON_TINT() -> Vec4 {
        super::ACTIVE_PALETTE.with(|c| c.borrow().graph.pin_icon_tint)
    }
}

// ===== LAYOUT UTILITIES =====

/// Calculate centered position
pub fn center_x(container_width: f32, element_width: f32) -> f32 {
    (container_width - element_width) / 2.0
}

/// Calculate centered position
pub fn center_y(container_height: f32, element_height: f32) -> f32 {
    (container_height - element_height) / 2.0
}

/// Vertically center text within a container
/// Text rendering uses baseline positioning, so we need to account for ascent
pub fn center_text_y(container_y: f32, container_height: f32, font_size: f32) -> f32 {
    // Center of container minus half the font size to get proper baseline position
    container_y + (container_height - font_size) / 2.0 + font_size * 0.75
}

/// Align element to bottom of container with padding
pub fn align_bottom(container_y: f32, container_height: f32, element_height: f32, padding: f32) -> f32 {
    container_y + container_height - element_height - padding
}

/// Align element to right of container with padding
pub fn align_right(container_x: f32, container_width: f32, element_width: f32, padding: f32) -> f32 {
    container_x + container_width - element_width - padding
}

// ===== ELEVATION / DROP SHADOW PRESETS =====

/// Cassette-futurism elevation tiers. Directional, feathered drop shadows tinted from [`bg::PRIMARY`]
/// (no harsh pure-black drops) so overlays read as floating plates, not glassy cards.
///
/// **Lighting model.** A single key light sits off-screen at angle `-45°` (upper-left). Every
/// drop shadow is therefore offset toward `+135°` (bottom-right) by [`SHADOW_SIZE_PX`] — the
/// same vector for every component keeps the scene consistent. The gaussian sigma in the
/// fragment shader feathers the shadow into a gradient; tiers differ in blur and opacity,
/// not in direction.
///
/// See the `Elevation` row in [`docs/design/cassette-futurism.md`](../../docs/design/cassette-futurism.md).
pub mod elevation {
    use super::Vec4;
    use crate::ui::shadow::{BorderHighlightSpec, InnerShadowSpec, ShadowSpec, SurfaceHighlightSpec};
    use glam::Vec2;

    /// Offset magnitude (in pixels) of every drop shadow. Projects the key light at `-45°`
    /// into a shadow vector at `+135°`. Bump this globally to push the whole scene further
    /// off its background plane.
    pub const SHADOW_SIZE_PX: f32 = 4.0;

    /// Unit component of the 135° shadow direction: `cos(45°) = sin(45°) = 1/√2`. Multiplying
    /// [`SHADOW_SIZE_PX`] by this gives the per-axis offset so the displacement vector has
    /// length [`SHADOW_SIZE_PX`] (not [`SHADOW_SIZE_PX`] on each axis).
    const LIGHT_OFFSET_UNIT: f32 = 0.70710678;

    /// Tint a shadow color from the active `bg::PRIMARY`: darker RGB, variable alpha. Lower the
    /// RGB multipliers to push the shadow closer to pure black; raise them for a softer,
    /// theme-tinted shadow. `alpha` is the main per-tier darkness dimmer.
    fn shadow_tint(alpha: f32) -> Vec4 {
        let base = super::bg::PRIMARY();
        Vec4::new(base.x * 0.20, base.y * 0.20, base.z * 0.25, alpha)
    }

    /// Build a directional drop shadow at `+135°` (bottom-right) with the global
    /// [`SHADOW_SIZE_PX`] offset, given gaussian sigma (feather width) and opacity.
    /// Tiers only vary `sigma` and `alpha` — direction and magnitude are shared.
    fn directional(sigma: f32, alpha: f32) -> ShadowSpec {
        let d = SHADOW_SIZE_PX * LIGHT_OFFSET_UNIT;
        ShadowSpec {
            offset: Vec2::new(d, d),
            sigma,
            color: shadow_tint(alpha),
            spread: 0.0,
        }
    }

    /// Resting raised element (cards, rows, grouped lists). Tight feather.
    pub fn LOW() -> ShadowSpec {
        directional(4.0, 0.55)
    }

    /// Floating chrome (sidebar edge, composer deck, popovers). Wider feather, stronger gradient.
    pub fn MEDIUM() -> ShadowSpec {
        directional(6.0, 0.68)
    }

    /// Modals, dialogs, toasts (things lifted off the page). Wide feather, maximum gradient
    /// spread off the bottom-right edge.
    pub fn HIGH() -> ShadowSpec {
        directional(10.0, 0.80)
    }

    // ===== INNER DROP SHADOW =====
    //
    // Same lighting model, but the shadow lives *inside* the component on the top-left
    // (the light-facing inner wall). Tiers differ in offset penetration + feather.

    /// Subtle inset shadow for pressed / recessed controls (text inputs, wells).
    pub fn INNER_LOW() -> InnerShadowSpec {
        InnerShadowSpec {
            offset_size: SHADOW_SIZE_PX,
            sigma: 5.0,
            color: shadow_tint(0.45),
        }
    }

    /// Default inset shadow for depressed panels and recessed chrome (search fields, chips).
    pub fn INNER_MEDIUM() -> InnerShadowSpec {
        InnerShadowSpec {
            offset_size: SHADOW_SIZE_PX * 1.5,
            sigma: 8.0,
            color: shadow_tint(0.55),
        }
    }

    /// Heavy inset for deeply-recessed wells (inline code blocks, embedded consoles).
    pub fn INNER_HIGH() -> InnerShadowSpec {
        InnerShadowSpec {
            offset_size: SHADOW_SIZE_PX * 2.0,
            sigma: 12.0,
            color: shadow_tint(0.65),
        }
    }

    // ===== SPECULAR HIGHLIGHTS =====

    /// Warm-white tint for specular highlights. Pulled slightly toward the active theme's
    /// `text::PRIMARY` so highlights feel part of the palette, not pure white.
    fn highlight_tint(alpha: f32) -> Vec4 {
        let base = super::text::PRIMARY();
        Vec4::new(
            (base.x * 0.4 + 0.6).min(1.0),
            (base.y * 0.4 + 0.6).min(1.0),
            (base.z * 0.4 + 0.6).min(1.0),
            alpha,
        )
    }

    /// Thin, bright rim along the top-left inner edge of the component. Used to simulate
    /// a catch-light on raised chrome (buttons, floating panels).
    pub fn BORDER_HIGHLIGHT() -> BorderHighlightSpec {
        BorderHighlightSpec {
            width: 1.5,
            sigma: 0.75,
            color: highlight_tint(0.55),
        }
    }

    /// Wider, softer rim for more emphatic raised elements (active states, primary buttons).
    pub fn BORDER_HIGHLIGHT_STRONG() -> BorderHighlightSpec {
        BorderHighlightSpec {
            width: 2.5,
            sigma: 1.25,
            color: highlight_tint(0.75),
        }
    }

    /// Subtle diagonal sheen across the surface — fakes a gentle bevel under the key light.
    /// Good default for cards and plates.
    pub fn SURFACE_HIGHLIGHT() -> SurfaceHighlightSpec {
        SurfaceHighlightSpec {
            curve: 2.0,
            sigma: 0.75,
            color: highlight_tint(0.10),
        }
    }

    /// More visible surface sheen for hero chrome (top of the header, composer chassis).
    pub fn SURFACE_HIGHLIGHT_STRONG() -> SurfaceHighlightSpec {
        SurfaceHighlightSpec {
            curve: 1.5,
            sigma: 0.75,
            color: highlight_tint(0.18),
        }
    }
}

