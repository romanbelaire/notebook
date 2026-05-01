use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::app::App;
use glam::Vec2;
use crate::ui::style;
use crate::ui::core::{Rect, layout};
use crate::ui::{Text, TextAlignment};
use crate::ui::components::Renderable;

pub fn render_header(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    renderer.set_composite_layer(CompositeLayer::HudChrome);
    // Note: "header" is already validated and pushed by HeaderComponent before calling this function
    // We don't need to push/pop it here - it's already on the parent stack
    
    const FONT_SIZE: f32 = style::font_size::MEDIUM;
    
    let header_rect = Rect::from_pos_size(app.header.position, app.header.size);

    // Elevation: cassette-futurism horizon. Lift the header chrome off the viewport with a soft drop shadow.
    renderer.queue_shadow(&header_rect, 0.0, &style::elevation::MEDIUM());

    // Header background
    let header_quad = Quad {
        position: header_rect.position(),
        size: header_rect.size(),
        color: style::bg::SECONDARY(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&header_quad.to_vertices());
    renderer.add_vertices(vertices, None);
    vertices.clear();

    // Tab bar rect (relative to screen, not header)
    let tab_bar_rect = Rect::from_pos_size(
        app.header.tab_bar.position + app.header.position,
        app.header.tab_bar.size
    );
    let tab_inset = style::stroke::TAB_BAR_INNER_INSET;
    let track_inner = tab_bar_rect.inset(tab_inset);
    // Vertical padding only: track uses full cavity width; height follows tab label line box.
    let tab_label_line_h = FONT_SIZE * style::font_size::LINE_HEIGHT_RATIO;
    const TAB_LABEL_V_PAD: f32 = 4.0;
    let track_pill_h = (tab_label_line_h + TAB_LABEL_V_PAD * 2.0).min(track_inner.height);
    let nest_y = (track_inner.height - track_pill_h) * 0.5;
    // Track spans full cavity width so its outer edges align with the well (horizontal nest only squeezed text before).
    let track_pill = track_inner.inset_by(0.0, nest_y, 0.0, nest_y);
    let folder_r = style::chrome::TAB_FOLDER_TOP_RADIUS;
    let tab_pill_h = (folder_r * 2.0).min(tab_bar_rect.height);
    let tab_equator = tab_pill_h * 0.5;
    let tab_shank_h = (tab_bar_rect.height - tab_equator).max(0.0);
    let ring_w = style::stroke::TAB_BAR_RING_PX;
    const CLIP_PAD: f32 = 2.0;

    let clip_tab_open = Rect::new(
        tab_bar_rect.x,
        tab_bar_rect.y,
        tab_bar_rect.width,
        tab_equator + CLIP_PAD,
    );
    // Inner surface of the folder shell (drawn first so corner wedges are cream, then bezel on top).
    let cream_cap_r = (folder_r - ring_w * 0.5)
        .max(2.0)
        .min(tab_pill_h * 0.5);

    let tab_cap_cavity = Quad {
        position: tab_bar_rect.position(),
        size: Vec2::new(tab_bar_rect.width, tab_pill_h),
        color: style::chrome::COMPOSER_BACKPLATE(),
        corner_radius: cream_cap_r,
        bubble_effect: false,
        slider_effect: false,
    };
    let tab_shank_cavity = Quad {
        position: Vec2::new(tab_bar_rect.x, tab_bar_rect.y + tab_equator),
        size: Vec2::new(tab_bar_rect.width, tab_shank_h),
        color: style::chrome::COMPOSER_BACKPLATE(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };

    // Stay below min_dim/2 − epsilon so ui_shader never treats the fill as the elliptical-glow path.
    let nest_r = (track_pill.height * 0.42)
        .min(18.0)
        .max(9.0)
        .min(track_pill.height * 0.5 - 1.0);
    let track_pill_fill = Quad {
        position: track_pill.position(),
        size: track_pill.size(),
        color: style::chrome::TAB_BAR_BG(),
        corner_radius: nest_r,
        bubble_effect: false,
        slider_effect: false,
    };

    let nest_ring = style::chrome::COMPOSER_BACKPLATE_DIM_RIM();
    let track_ring_w = style::stroke::TAB_TRACK_PILL_RING_PX;
    let track_outline_r = nest_r + track_ring_w;
    let track_pill_outline = Quad {
        position: Vec2::new(
            track_pill.x - track_ring_w,
            track_pill.y - track_ring_w,
        ),
        size: Vec2::new(
            track_pill.width + track_ring_w * 2.0,
            track_pill.height + track_ring_w * 2.0,
        ),
        color: nest_ring,
        corner_radius: -track_outline_r,
        bubble_effect: false,
        slider_effect: false,
    };

    let tb = &app.header.tab_bar;
    let tab_logical_w = tb.size.x.max(1.0);
    let frac_lo = (tb.slider_position() / tab_logical_w).clamp(0.0, 1.0);
    let frac_w = (tb.slider_width() / tab_logical_w).clamp(0.0, 1.0);
    let slider_w_screen = (frac_w * track_pill.width).max(1.0);
    let slider_rect = Rect::new(
        track_pill.x + frac_lo * track_pill.width,
        track_pill.y,
        slider_w_screen,
        track_pill.height,
    );
    let mut slider_ring = style::accent::PHOSPHOR();
    slider_ring.w = style::stroke::TAB_SLIDER_RING_PX;
    let slider_outline = Quad {
        position: slider_rect.position(),
        size: slider_rect.size(),
        color: slider_ring,
        corner_radius: -nest_r,
        bubble_effect: false,
        slider_effect: false,
    };
    let slider_quad = Quad {
        position: slider_rect.position(),
        size: slider_rect.size(),
        color: style::chrome::TAB_BAR_SLIDER(),
        corner_radius: nest_r,
        bubble_effect: false,
        slider_effect: false,
    };

    let mut tab_track_ring = style::chrome::COMPOSER_BACKPLATE();
    tab_track_ring.w = ring_w;
    let tab_cap_border = Quad {
        position: tab_bar_rect.position(),
        size: Vec2::new(tab_bar_rect.width, tab_pill_h),
        color: tab_track_ring,
        corner_radius: -folder_r,
        bubble_effect: false,
        slider_effect: false,
    };
    let tab_shank_border = Quad {
        position: Vec2::new(tab_bar_rect.x, tab_bar_rect.y + tab_equator),
        size: Vec2::new(tab_bar_rect.width, tab_shank_h),
        color: tab_track_ring,
        // Negative triggers border SDF; magnitude ~0 keeps corners square (-0.0 is not < 0 in IEEE).
        corner_radius: -1.0e-4,
        bubble_effect: false,
        slider_effect: false,
    };

    // Z: cream well (under bezel) → folder rim → inset track (fill + rim) → active pill → text (last)
    // Folder tab sits on top of the header surface; queue one shadow spanning cap+shank so the
    // rounded-top / square-bottom folder reads as lifted chrome. Corner radius of the tab cap covers
    // the visible rounded edges; the shank bottom is flush with the header bottom so bottom rounding
    // gets clipped naturally by the header/background boundary.
    renderer.queue_shadow(&tab_bar_rect, cream_cap_r, &style::elevation::LOW());
    renderer.add_quad(&tab_cap_cavity, Some(&clip_tab_open));
    vertices.extend_from_slice(&tab_shank_cavity.to_vertices());
    renderer.add_vertices(vertices, None);
    vertices.clear();
    renderer.add_quad(&tab_cap_border, Some(&clip_tab_open));
    renderer.add_quad(&tab_shank_border, None);
    vertices.extend_from_slice(&track_pill_fill.to_vertices());
    vertices.extend_from_slice(&track_pill_outline.to_vertices());
    renderer.add_vertices(vertices, None);
    vertices.clear();
    vertices.extend_from_slice(&slider_outline.to_vertices());
    vertices.extend_from_slice(&slider_quad.to_vertices());
    renderer.add_vertices(vertices, None);
    vertices.clear();

    // Tab labels — inside nested track pill
    let tab_width = track_pill.width / app.header.tab_bar.tabs.len() as f32;
    let tab_widths: Vec<f32> = (0..app.header.tab_bar.tabs.len()).map(|_| tab_width).collect();
    let tab_rects = layout::stack_horizontal(&track_pill, &tab_widths, 0.0, 0.0);
    
    // Set parent for tab bar components
    // Validate "header_tab_bar" BEFORE pushing it (so it's in hierarchy when children validate)
    renderer.validate_component("header_tab_bar", Some("header"), "TabBar");
    renderer.push_parent("header_tab_bar".to_string());
    for (i, tab) in app.header.tab_bar.tabs.iter().enumerate() {
        let tab_rect = tab_rects[i];
        
        // Tab label - use Text component with unique parent for each tab
        let tab_parent_id = format!("header_tab_{}", i);
        // Validate tab parent BEFORE pushing it (so it's in hierarchy when Text validates)
        renderer.validate_component(&tab_parent_id, Some("header_tab_bar"), "Tab");
        renderer.push_parent(tab_parent_id.clone());
        
        let text_color = if i == app.header.tab_bar.active_index {
            style::text::PRIMARY()
        } else {
            style::text::SECONDARY()
        };
        let mut tab_text = Text::new_for_render(tab.label())
            .with_font_size(FONT_SIZE)
            .with_color(text_color)
            .with_alignment(TextAlignment::Center)
            .with_scissor(None);
        tab_text.update_layout(tab_rect, None, None);
        tab_text.render(renderer, app, vertices, None);
        
        renderer.pop_parent();
    }
    renderer.pop_parent();

    // Title — use Text component
    // Position to left of tab bar, vertically centered in header
    // Validate title BEFORE pushing it (so it's in hierarchy when Text validates)
    renderer.validate_component("header_title", Some("header"), "Title");
    renderer.push_parent("header_title".to_string());
    let title_rect = Rect::new(
        header_rect.x + style::padding::MEDIUM,
        header_rect.y + (header_rect.height - FONT_SIZE * 1.2) / 2.0,
        tab_bar_rect.x - header_rect.x - style::padding::MEDIUM * 2.0,
        FONT_SIZE * 1.2,
    );
    let mut title_text = Text::new_for_render("Constellar")
        .with_font_size(FONT_SIZE)
        .with_color(style::text::PRIMARY())
        .with_alignment(TextAlignment::Left)
        .with_scissor(None);
    title_text.update_layout(title_rect, None, None);
    title_text.render(renderer, app, vertices, None);
    renderer.pop_parent();
    
    // Window control buttons (if enabled)
    if app.header.show_window_controls {
        // Close button
        let close_rect = Rect::from_pos_size(app.header.close_button.position, app.header.close_button.size);
        let close_color = match app.header.close_button.state {
            crate::ui::ButtonState::Pressed => style::button::DANGER_ACTIVE(),
            crate::ui::ButtonState::Hover => style::button::DANGER_HOVER(),
            crate::ui::ButtonState::Normal => style::button::DANGER(),
        };
        let close_bg = Quad {
            position: close_rect.position(),
            size: close_rect.size(),
            color: close_color,
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&close_bg.to_vertices());
        // Window control buttons - set parent
        renderer.push_parent("header_window_controls".to_string());
        renderer.validate_component("header_window_controls", None, "WindowControls");
        
        // Close button text - use Text component
        let mut close_text = Text::new_for_render("×")
            .with_font_size(style::font_size::LARGE)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Center)
            .with_scissor(None);
        close_text.update_layout(close_rect, None, None);
        close_text.render(renderer, app, vertices, None);
        
        // Maximize button
        let max_rect = Rect::from_pos_size(app.header.maximize_button.position, app.header.maximize_button.size);
        let max_color = match app.header.maximize_button.state {
            crate::ui::ButtonState::Pressed => style::button::SECONDARY_ACTIVE(),
            crate::ui::ButtonState::Hover => style::button::SECONDARY_HOVER(),
            crate::ui::ButtonState::Normal => style::button::SECONDARY(),
        };
        let max_bg = Quad {
            position: max_rect.position(),
            size: max_rect.size(),
            color: max_color,
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&max_bg.to_vertices());
        // Maximize button text - use Text component
        let mut max_text = Text::new_for_render("□")
            .with_font_size(FONT_SIZE)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Center)
            .with_scissor(None);
        max_text.update_layout(max_rect, None, None);
        max_text.render(renderer, app, vertices, None);
        
        // Minimize button
        let min_rect = Rect::from_pos_size(app.header.minimize_button.position, app.header.minimize_button.size);
        let min_color = match app.header.minimize_button.state {
            crate::ui::ButtonState::Pressed => style::button::SECONDARY_ACTIVE(),
            crate::ui::ButtonState::Hover => style::button::SECONDARY_HOVER(),
            crate::ui::ButtonState::Normal => style::button::SECONDARY(),
        };
        let min_bg = Quad {
            position: min_rect.position(),
            size: min_rect.size(),
            color: min_color,
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&min_bg.to_vertices());
        // Minimize button text - use Text component
        let mut min_text = Text::new_for_render("−")
            .with_font_size(FONT_SIZE)
            .with_color(style::text::PRIMARY())
            .with_alignment(TextAlignment::Center)
            .with_scissor(None);
        min_text.update_layout(min_rect, None, None);
        min_text.render(renderer, app, vertices, None);
        
        renderer.pop_parent();
    }

    let rule_h = style::stroke::INSTRUMENT_RULE_PX;
    let instrument_rule = Quad {
        position: Vec2::new(0.0, header_rect.y + header_rect.height - rule_h),
        size: Vec2::new(app.viewport_size.x, rule_h),
        color: style::chrome::COMPOSER_BACKPLATE(),
        corner_radius: 0.0,
        bubble_effect: false,
        slider_effect: false,
    };
    vertices.extend_from_slice(&instrument_rule.to_vertices());
    
    // Note: "header" parent is popped by HeaderComponent after calling this function
}

/// Stateless [`Renderable`] for header chrome; delegates to [`render_header`].
pub struct HeaderViewport;

pub const HEADER_VIEWPORT: HeaderViewport = HeaderViewport;

/// Opt-in drop shadow for the header chrome.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for HeaderViewport {
    fn z_order(&self) -> i32 {
        100
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
        render_header(renderer, app, vertices);
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        Some(Rect::from_pos_size(app.header.position, app.header.size))
    }

    fn update_layout(
        &mut self,
        _available_rect: Rect,
        _dirty_rect: Option<Rect>,
        _app: Option<&App>,
    ) {
    }
}

