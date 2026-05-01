//! Named color themes; [`ThemePalette::resolve`] maps persisted `SettingsState.theme` to UI colors.

use glam::Vec4;

/// `(settings id, label shown in Settings dropdown)`
pub const THEME_CHOICES: &[(&str, &str)] = &[
    ("standard", "Cassette (dark blue)"),
    ("cassette-sage", "Cassette — sage & rust"),
    ("cassette-sunburst", "Cassette — sunburst warm"),
    ("cassette-vector", "Cassette — vector HUD"),
    (
        "cassette-field",
        "Cassette — field (#242623 / #CAC977 / #DC4C0D / #71BDBD)",
    ),
    ("sakura-light", "Sakura light"),
    ("springtime-light", "Springtime light"),
    ("forest-dark", "Forest dark"),
    ("toadstool-light", "Toadstool light"),
    ("acorn-dark", "Acorn dark"),
    ("light", "Basic light"),
    ("dark", "Dark high contrast"),
];

#[derive(Clone)]
pub struct Accent {
    pub phosphor: Vec4,
    pub phosphor_glow: Vec4,
    pub pop: Vec4,
    pub pop_cool: Vec4,
    pub warning: Vec4,
}

#[derive(Clone)]
pub struct Backdrop {
    pub parallax_slow: Vec4,
    pub parallax_fast: Vec4,
    pub depth_band: Vec4,
}

#[derive(Clone)]
pub struct Chrome {
    pub tab_bar_bg: Vec4,
    pub tab_bar_slider: Vec4,
    pub composer_backplate: Vec4,
}

#[derive(Clone)]
pub struct Bg {
    pub primary: Vec4,
    pub chrome_deck: Vec4,
    pub panel_well: Vec4,
    pub input_focused: Vec4,
    pub panel_popup: Vec4,
    pub user_message: Vec4,
    pub assistant_message: Vec4,
    pub muted_message: Vec4,
    pub shard_backplate: Vec4,
}

#[derive(Clone)]
pub struct Text {
    pub primary: Vec4,
    pub secondary: Vec4,
    pub tertiary: Vec4,
    pub placeholder: Vec4,
    pub ghost: Vec4,
    pub accent: Vec4,
}

#[derive(Clone)]
pub struct Button {
    pub primary: Vec4,
    pub primary_hover: Vec4,
    pub primary_active: Vec4,
    pub secondary: Vec4,
    pub secondary_hover: Vec4,
    pub secondary_active: Vec4,
    pub danger: Vec4,
    pub danger_hover: Vec4,
    pub danger_active: Vec4,
}

#[derive(Clone)]
pub struct Border {
    pub subtle: Vec4,
    pub focused: Vec4,
    pub hover: Vec4,
    pub accent: Vec4,
    pub phosphor: Vec4,
    pub instrument_rule: Vec4,
}

#[derive(Clone)]
pub struct Highlight {
    pub selection: Vec4,
    pub hover: Vec4,
    pub active: Vec4,
}

#[derive(Clone)]
pub struct Markdown {
    pub code_foreground: Vec4,
}

#[derive(Clone)]
pub struct Editor {
    pub match_highlight_text: Vec4,
    pub selection_band: Vec4,
}

#[derive(Clone)]
pub struct Graph {
    pub pin_icon_tint: Vec4,
}

#[derive(Clone)]
pub struct ThemePalette {
    pub accent: Accent,
    pub backdrop: Backdrop,
    pub chrome: Chrome,
    pub bg: Bg,
    pub text: Text,
    pub button: Button,
    pub border: Border,
    pub highlight: Highlight,
    pub markdown: Markdown,
    pub editor: Editor,
    pub graph: Graph,
}

impl ThemePalette {
    /// Constellation shard card fill: between bulkhead deck and panel well so the card reads as a discrete plate.
    fn interpolate_shard_backplate(chrome_deck: Vec4, panel_well: Vec4) -> Vec4 {
        const T: f32 = 0.55;
        Vec4::new(
            chrome_deck.x + (panel_well.x - chrome_deck.x) * T,
            chrome_deck.y + (panel_well.y - chrome_deck.y) * T,
            chrome_deck.z + (panel_well.z - chrome_deck.z) * T,
            1.0,
        )
    }

    /// Default “cassette futurism” palette (former hard-coded `style` colors).
    pub fn standard() -> Self {
        let accent = Accent {
            phosphor: Vec4::new(0.90, 0.87, 0.80, 1.0),
            phosphor_glow: Vec4::new(0.16, 0.22, 0.40, 1.0),
            pop: Vec4::new(0.68, 0.35, 0.30, 1.0),
            pop_cool: Vec4::new(0.44, 0.50, 0.74, 1.0),
            warning: Vec4::new(0.78, 0.32, 0.28, 1.0),
        };
        let pop = accent.pop;
        let chrome_deck = Vec4::new(0.11, 0.14, 0.26, 1.0);
        let panel_well = Vec4::new(0.20, 0.24, 0.38, 1.0);
        Self {
            accent,
            backdrop: Backdrop {
                parallax_slow: Vec4::new(0.22, 0.28, 0.46, 0.038),
                parallax_fast: Vec4::new(0.22, 0.28, 0.46, 0.026),
                depth_band: Vec4::new(0.10, 0.13, 0.26, 0.08),
            },
            chrome: Chrome {
                tab_bar_bg: Vec4::new(0.08, 0.11, 0.22, 0.96),
                tab_bar_slider: Vec4::new(0.28, 0.34, 0.52, 1.0),
                composer_backplate: Vec4::new(0.93, 0.89, 0.80, 1.0),
            },
            bg: Bg {
                primary: Vec4::new(0.08, 0.10, 0.18, 1.0),
                chrome_deck,
                panel_well,
                input_focused: Vec4::new(0.24, 0.28, 0.44, 1.0),
                panel_popup: Vec4::new(0.09, 0.11, 0.22, 0.97),
                user_message: Vec4::new(0.62, 0.32, 0.28, 1.0),
                assistant_message: Vec4::new(0.30, 0.34, 0.48, 1.0),
                muted_message: Vec4::new(0.10, 0.12, 0.22, 0.74),
                shard_backplate: Self::interpolate_shard_backplate(chrome_deck, panel_well),
            },
            text: Text {
                primary: Vec4::new(0.93, 0.91, 0.84, 1.0),
                secondary: Vec4::new(0.66, 0.72, 0.84, 1.0),
                tertiary: Vec4::new(0.46, 0.52, 0.64, 1.0),
                placeholder: Vec4::new(0.46, 0.52, 0.64, 0.72),
                ghost: Vec4::new(0.46, 0.52, 0.64, 0.45),
                accent: Vec4::new(0.96, 0.93, 0.84, 1.0),
            },
            button: Button {
                primary: Vec4::new(0.08, 0.10, 0.18, 1.0),
                primary_hover: Vec4::new(0.14, 0.18, 0.32, 1.0),
                primary_active: Vec4::new(0.06, 0.08, 0.14, 1.0),
                secondary: Vec4::new(0.26, 0.30, 0.46, 1.0),
                secondary_hover: Vec4::new(0.32, 0.36, 0.52, 1.0),
                secondary_active: Vec4::new(0.20, 0.24, 0.38, 1.0),
                danger: Vec4::new(0.7, 0.2, 0.2, 1.0),
                danger_hover: Vec4::new(0.8, 0.25, 0.25, 1.0),
                danger_active: Vec4::new(0.6, 0.15, 0.15, 1.0),
            },
            border: Border {
                subtle: Vec4::new(0.88, 0.85, 0.78, 0.38),
                focused: Vec4::new(0.48, 0.62, 0.94, 0.78),
                hover: Vec4::new(0.80, 0.78, 0.72, 0.45),
                accent: Vec4::new(0.70, 0.42, 0.36, 0.52),
                phosphor: Vec4::new(0.82, 0.80, 0.74, 0.68),
                instrument_rule: Vec4::new(0.91, 0.88, 0.81, 0.88),
            },
            highlight: Highlight {
                selection: Vec4::new(0.90, 0.86, 0.78, 0.14),
                hover: Vec4::new(0.24, 0.28, 0.44, 1.0),
                active: Vec4::new(0.30, 0.34, 0.50, 1.0),
            },
            markdown: Markdown {
                code_foreground: Vec4::new(0.82, 0.79, 0.72, 1.0),
            },
            editor: Editor {
                match_highlight_text: pop,
                selection_band: Vec4::new(0.44, 0.50, 0.74, 0.28),
            },
            graph: Graph { pin_icon_tint: pop },
        }
    }

    pub fn resolve(theme_id: &str) -> Self {
        let id = theme_id.trim();
        let id = if id.is_empty() { "standard" } else { id };
        let mut p = Self::standard();
        match id {
            "standard" => {}
            "cassette-sage" => p.apply_cassette_sage(),
            "cassette-sunburst" => p.apply_cassette_sunburst(),
            "cassette-vector" => p.apply_cassette_vector(),
            "cassette-field" => p.apply_cassette_field(),
            "sakura-light" => p.apply_sakura_light(),
            "springtime-light" => p.apply_springtime_light(),
            "forest-dark" => p.apply_forest_dark(),
            "toadstool-light" => p.apply_toadstool_light(),
            "acorn-dark" => p.apply_acorn_dark(),
            "light" => p.apply_basic_light(),
            "dark" => p.apply_dark_contrast(),
            _ => return Self::standard(),
        }
        p
    }

    /// Moodboard: charcoal bulkheads, muted sage wells, **burnt orange** impulse, **teal** data accent, taupe readouts.
    fn apply_cassette_sage(&mut self) {
        let pop = Vec4::new(0.78, 0.42, 0.24, 1.0);
        self.bg.primary = Vec4::new(0.09, 0.10, 0.08, 1.0);
        self.bg.chrome_deck = Vec4::new(0.11, 0.13, 0.10, 1.0);
        self.bg.panel_well = Vec4::new(0.16, 0.20, 0.15, 1.0);
        self.bg.input_focused = Vec4::new(0.20, 0.26, 0.18, 1.0);
        self.bg.panel_popup = Vec4::new(0.08, 0.10, 0.08, 0.97);
        self.bg.user_message = Vec4::new(0.72, 0.38, 0.22, 1.0);
        self.bg.assistant_message = Vec4::new(0.28, 0.36, 0.30, 1.0);
        self.bg.muted_message = Vec4::new(0.10, 0.12, 0.09, 0.74);
        self.text.primary = Vec4::new(0.93, 0.91, 0.84, 1.0);
        self.text.secondary = Vec4::new(0.72, 0.68, 0.58, 1.0);
        self.text.tertiary = Vec4::new(0.55, 0.52, 0.45, 1.0);
        self.text.placeholder = Vec4::new(0.55, 0.52, 0.45, 0.72);
        self.text.ghost = Vec4::new(0.55, 0.52, 0.45, 0.45);
        self.text.accent = Vec4::new(0.95, 0.86, 0.68, 1.0);
        self.accent.phosphor = Vec4::new(0.88, 0.90, 0.80, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.18, 0.32, 0.26, 1.0);
        self.accent.pop = pop;
        self.accent.pop_cool = Vec4::new(0.22, 0.55, 0.52, 1.0);
        self.accent.warning = Vec4::new(0.82, 0.32, 0.20, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.08, 0.10, 0.09, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.26, 0.38, 0.32, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.91, 0.88, 0.78, 1.0);
        self.button.primary = Vec4::new(0.10, 0.12, 0.09, 1.0);
        self.button.primary_hover = Vec4::new(0.16, 0.20, 0.14, 1.0);
        self.button.primary_active = Vec4::new(0.07, 0.08, 0.06, 1.0);
        self.button.secondary = Vec4::new(0.24, 0.34, 0.28, 1.0);
        self.button.secondary_hover = Vec4::new(0.30, 0.42, 0.34, 1.0);
        self.button.secondary_active = Vec4::new(0.18, 0.26, 0.22, 1.0);
        self.border.subtle = Vec4::new(0.55, 0.52, 0.45, 0.38);
        self.border.focused = Vec4::new(0.22, 0.55, 0.52, 0.78);
        self.border.hover = Vec4::new(0.75, 0.62, 0.48, 0.45);
        self.border.accent = Vec4::new(0.78, 0.45, 0.30, 0.52);
        self.border.phosphor = Vec4::new(0.78, 0.76, 0.65, 0.68);
        self.border.instrument_rule = Vec4::new(0.90, 0.86, 0.74, 0.88);
        self.highlight.selection = Vec4::new(0.88, 0.72, 0.45, 0.16);
        self.highlight.hover = Vec4::new(0.20, 0.28, 0.22, 1.0);
        self.highlight.active = Vec4::new(0.26, 0.36, 0.28, 1.0);
        self.backdrop.parallax_slow = Vec4::new(0.62, 0.38, 0.20, 0.042);
        self.backdrop.parallax_fast = Vec4::new(0.18, 0.48, 0.44, 0.028);
        self.backdrop.depth_band = Vec4::new(0.14, 0.22, 0.16, 0.09);
        self.markdown.code_foreground = Vec4::new(0.84, 0.80, 0.68, 1.0);
        self.editor.selection_band = Vec4::new(0.22, 0.52, 0.48, 0.28);
        self.editor.match_highlight_text = pop;
        self.graph.pin_icon_tint = pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    /// Moodboard: umber black, **mustard / cream** chrome, stripe-energy haze (orange + teal).
    fn apply_cassette_sunburst(&mut self) {
        let pop = Vec4::new(0.82, 0.48, 0.22, 1.0);
        self.bg.primary = Vec4::new(0.12, 0.09, 0.06, 1.0);
        self.bg.chrome_deck = Vec4::new(0.16, 0.11, 0.08, 1.0);
        self.bg.panel_well = Vec4::new(0.24, 0.18, 0.12, 1.0);
        self.bg.input_focused = Vec4::new(0.30, 0.22, 0.14, 1.0);
        self.bg.panel_popup = Vec4::new(0.14, 0.10, 0.07, 0.97);
        self.bg.user_message = Vec4::new(0.75, 0.40, 0.18, 1.0);
        self.bg.assistant_message = Vec4::new(0.35, 0.40, 0.45, 1.0);
        self.bg.muted_message = Vec4::new(0.14, 0.11, 0.08, 0.74);
        self.text.primary = Vec4::new(0.96, 0.92, 0.82, 1.0);
        self.text.secondary = Vec4::new(0.78, 0.70, 0.55, 1.0);
        self.text.tertiary = Vec4::new(0.58, 0.52, 0.42, 1.0);
        self.text.placeholder = Vec4::new(0.58, 0.52, 0.42, 0.72);
        self.text.ghost = Vec4::new(0.58, 0.52, 0.42, 0.45);
        self.text.accent = Vec4::new(0.98, 0.88, 0.45, 1.0);
        self.accent.phosphor = Vec4::new(0.95, 0.85, 0.55, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.42, 0.28, 0.12, 1.0);
        self.accent.pop = pop;
        self.accent.pop_cool = Vec4::new(0.28, 0.55, 0.62, 1.0);
        self.accent.warning = Vec4::new(0.88, 0.32, 0.22, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.14, 0.10, 0.07, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.48, 0.38, 0.22, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.94, 0.88, 0.72, 1.0);
        self.button.primary = Vec4::new(0.14, 0.10, 0.07, 1.0);
        self.button.primary_hover = Vec4::new(0.22, 0.16, 0.10, 1.0);
        self.button.primary_active = Vec4::new(0.10, 0.07, 0.05, 1.0);
        self.button.secondary = Vec4::new(0.38, 0.32, 0.24, 1.0);
        self.button.secondary_hover = Vec4::new(0.46, 0.38, 0.28, 1.0);
        self.button.secondary_active = Vec4::new(0.30, 0.25, 0.18, 1.0);
        self.border.subtle = Vec4::new(0.72, 0.62, 0.48, 0.38);
        self.border.focused = Vec4::new(0.30, 0.58, 0.65, 0.78);
        self.border.hover = Vec4::new(0.85, 0.65, 0.35, 0.48);
        self.border.accent = Vec4::new(0.80, 0.50, 0.25, 0.52);
        self.border.phosphor = Vec4::new(0.88, 0.78, 0.55, 0.68);
        self.border.instrument_rule = Vec4::new(0.94, 0.88, 0.68, 0.90);
        self.highlight.selection = Vec4::new(0.95, 0.78, 0.35, 0.18);
        self.highlight.hover = Vec4::new(0.30, 0.22, 0.14, 1.0);
        self.highlight.active = Vec4::new(0.38, 0.28, 0.18, 1.0);
        self.backdrop.parallax_slow = Vec4::new(0.72, 0.42, 0.18, 0.048);
        self.backdrop.parallax_fast = Vec4::new(0.22, 0.52, 0.58, 0.032);
        self.backdrop.depth_band = Vec4::new(0.35, 0.22, 0.10, 0.10);
        self.markdown.code_foreground = Vec4::new(0.90, 0.82, 0.60, 1.0);
        self.editor.selection_band = Vec4::new(0.28, 0.55, 0.58, 0.30);
        self.editor.match_highlight_text = pop;
        self.graph.pin_icon_tint = pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    /// Moodboard: near-black deck, **green phosphor** readouts, **amber** impulse, **cyan** data traces.
    fn apply_cassette_vector(&mut self) {
        let pop = Vec4::new(0.95, 0.68, 0.28, 1.0);
        self.bg.primary = Vec4::new(0.04, 0.05, 0.05, 1.0);
        self.bg.chrome_deck = Vec4::new(0.06, 0.07, 0.07, 1.0);
        self.bg.panel_well = Vec4::new(0.10, 0.14, 0.11, 1.0);
        self.bg.input_focused = Vec4::new(0.12, 0.18, 0.14, 1.0);
        self.bg.panel_popup = Vec4::new(0.05, 0.07, 0.06, 0.97);
        self.bg.user_message = Vec4::new(0.55, 0.32, 0.22, 1.0);
        self.bg.assistant_message = Vec4::new(0.18, 0.28, 0.24, 1.0);
        self.bg.muted_message = Vec4::new(0.06, 0.08, 0.07, 0.74);
        self.text.primary = Vec4::new(0.86, 0.93, 0.80, 1.0);
        self.text.secondary = Vec4::new(0.58, 0.72, 0.62, 1.0);
        self.text.tertiary = Vec4::new(0.42, 0.55, 0.48, 1.0);
        self.text.placeholder = Vec4::new(0.42, 0.55, 0.48, 0.72);
        self.text.ghost = Vec4::new(0.42, 0.55, 0.48, 0.45);
        self.text.accent = Vec4::new(0.95, 0.88, 0.55, 1.0);
        self.accent.phosphor = Vec4::new(0.78, 0.92, 0.75, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.22, 0.52, 0.38, 1.0);
        self.accent.pop = pop;
        self.accent.pop_cool = Vec4::new(0.35, 0.72, 0.82, 1.0);
        self.accent.warning = Vec4::new(0.92, 0.40, 0.28, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.05, 0.07, 0.06, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.20, 0.38, 0.30, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.82, 0.90, 0.78, 1.0);
        self.button.primary = Vec4::new(0.06, 0.08, 0.07, 1.0);
        self.button.primary_hover = Vec4::new(0.10, 0.14, 0.12, 1.0);
        self.button.primary_active = Vec4::new(0.04, 0.05, 0.05, 1.0);
        self.button.secondary = Vec4::new(0.16, 0.30, 0.26, 1.0);
        self.button.secondary_hover = Vec4::new(0.20, 0.38, 0.32, 1.0);
        self.button.secondary_active = Vec4::new(0.12, 0.24, 0.20, 1.0);
        self.border.subtle = Vec4::new(0.50, 0.65, 0.55, 0.38);
        self.border.focused = Vec4::new(0.35, 0.72, 0.78, 0.78);
        self.border.hover = Vec4::new(0.65, 0.78, 0.55, 0.45);
        self.border.accent = Vec4::new(0.85, 0.55, 0.30, 0.52);
        self.border.phosphor = Vec4::new(0.62, 0.78, 0.65, 0.68);
        self.border.instrument_rule = Vec4::new(0.80, 0.88, 0.75, 0.88);
        self.highlight.selection = Vec4::new(0.45, 0.78, 0.52, 0.16);
        self.highlight.hover = Vec4::new(0.14, 0.22, 0.18, 1.0);
        self.highlight.active = Vec4::new(0.18, 0.28, 0.22, 1.0);
        self.backdrop.parallax_slow = Vec4::new(0.22, 0.48, 0.35, 0.04);
        self.backdrop.parallax_fast = Vec4::new(0.30, 0.62, 0.72, 0.028);
        self.backdrop.depth_band = Vec4::new(0.08, 0.18, 0.12, 0.10);
        self.markdown.code_foreground = Vec4::new(0.75, 0.88, 0.72, 1.0);
        self.editor.selection_band = Vec4::new(0.35, 0.68, 0.75, 0.28);
        self.editor.match_highlight_text = pop;
        self.graph.pin_icon_tint = pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    /// User swatches: background `#242623`, header `#525B60`, main text `#CAC977`, accent1 user bubble `#DC4C0D`,
    /// accent2 trim / highlight `#71BDBD`.
    fn apply_cassette_field(&mut self) {
        let bg = Vec4::new(36.0 / 255.0, 38.0 / 255.0, 35.0 / 255.0, 1.0);
        let hdr = Vec4::new(32.0 / 255.0, 32.0 / 255.0, 35.0 / 255.0, 1.0);
        let txt = Vec4::new(202.0 / 255.0, 201.0 / 255.0, 119.0 / 255.0, 1.0);
        let a1 = Vec4::new(35.0 / 255.0, 56.0 / 255.0, 59.0 / 255.0, 1.0);
        let a2 = Vec4::new(113.0 / 255.0, 189.0 / 255.0, 189.0 / 255.0, 1.0);
        let well = Vec4::new(46.0 / 255.0, 49.0 / 255.0, 46.0 / 255.0, 1.0);
        let pop = a1;
        let plastic = Vec4::new(188.7 / 255.0, 186.15 / 255.0, 147.9 / 255.0, 1.0);

        self.bg.primary = bg;
        self.bg.chrome_deck = hdr;
        self.bg.panel_well = well;
        self.bg.input_focused = Vec4::new(56.0 / 255.0, 60.0 / 255.0, 56.0 / 255.0, 1.0);
        self.bg.panel_popup = Vec4::new(bg.x, bg.y, bg.z, 0.97);
        self.bg.user_message = a1;
        self.bg.assistant_message =
            Vec4::new(38.0 / 255.0, 49.0 / 255.0, 41.0 / 255.0, 1.0);
        self.bg.muted_message = Vec4::new(bg.x + 0.02, bg.y + 0.02, bg.z + 0.02, 0.74);

        self.text.primary = txt;
        self.text.secondary = Vec4::new(
            txt.x * 0.88,
            txt.y * 0.88,
            txt.z * 0.88,
            1.0,
        );
        self.text.tertiary = Vec4::new(
            txt.x * 0.68,
            txt.y * 0.68,
            txt.z * 0.68,
            1.0,
        );
        self.text.placeholder = Vec4::new(
            txt.x * 0.55,
            txt.y * 0.55,
            txt.z * 0.55,
            0.72,
        );
        self.text.ghost = Vec4::new(
            txt.x * 0.55,
            txt.y * 0.55,
            txt.z * 0.55,
            0.45,
        );
        self.text.accent = a2;

        self.accent.phosphor = Vec4::new(
            (txt.x + 0.08).min(1.0),
            (txt.y + 0.08).min(1.0),
            (txt.z + 0.12).min(1.0),
            1.0,
        );
        self.accent.phosphor_glow = Vec4::new(26.0 / 255.0, 48.0 / 255.0, 46.0 / 255.0, 1.0);
        self.accent.pop = pop;
        self.accent.pop_cool = a2;
        self.accent.warning = Vec4::new(0.92, 0.28, 0.18, 1.0);

        self.chrome.tab_bar_bg = Vec4::new(hdr.x, hdr.y, hdr.z, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(
            (hdr.x + a2.x) * 0.5,
            (hdr.y + a2.y) * 0.5,
            (hdr.z + a2.z) * 0.5,
            1.0,
        );
        self.chrome.composer_backplate = plastic;

        self.button.primary = Vec4::new(bg.x + 0.04, bg.y + 0.04, bg.z + 0.04, 1.0);
        self.button.primary_hover = Vec4::new(bg.x + 0.08, bg.y + 0.08, bg.z + 0.08, 1.0);
        self.button.primary_active = bg;
        self.button.secondary = hdr;
        self.button.secondary_hover = Vec4::new(
            hdr.x + 0.06,
            hdr.y + 0.06,
            hdr.z + 0.06,
            1.0,
        );
        self.button.secondary_active =
            Vec4::new(hdr.x - 0.04, hdr.y - 0.04, hdr.z - 0.04, 1.0);

        self.border.subtle = Vec4::new(txt.x, txt.y, txt.z, 0.35);
        self.border.focused = Vec4::new(a2.x, a2.y, a2.z, 0.85);
        self.border.hover = Vec4::new(a2.x, a2.y, a2.z, 0.45);
        self.border.accent = Vec4::new(a1.x, a1.y, a1.z, 0.58);
        self.border.phosphor = Vec4::new(txt.x, txt.y, txt.z, 0.65);
        self.border.instrument_rule = Vec4::new(txt.x, txt.y, txt.z, 0.88);

        self.highlight.selection = Vec4::new(a2.x, a2.y, a2.z, 0.20);
        self.highlight.hover = well;
        self.highlight.active =
            Vec4::new((hdr.x + a2.x) * 0.5, (hdr.y + a2.y) * 0.5, (hdr.z + a2.z) * 0.5, 1.0);

        self.backdrop.parallax_slow = Vec4::new(a1.x, a1.y, a1.z, 0.038);
        self.backdrop.parallax_fast = Vec4::new(a2.x, a2.y, a2.z, 0.028);
        self.backdrop.depth_band = Vec4::new(0.12, 0.14, 0.12, 0.09);

        self.markdown.code_foreground =
            Vec4::new((txt.x + 0.06).min(1.0), (txt.y + 0.06).min(1.0), txt.z, 1.0);
        self.editor.selection_band = Vec4::new(a2.x, a2.y, a2.z, 0.30);
        self.editor.match_highlight_text = pop;
        self.graph.pin_icon_tint = pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    fn apply_sakura_light(&mut self) {
        self.bg.primary = Vec4::new(0.98, 0.94, 0.96, 1.0);
        self.bg.chrome_deck = Vec4::new(0.96, 0.90, 0.93, 1.0);
        self.bg.panel_well = Vec4::new(0.92, 0.86, 0.90, 1.0);
        self.bg.input_focused = Vec4::new(0.90, 0.82, 0.88, 1.0);
        self.bg.panel_popup = Vec4::new(0.99, 0.96, 0.97, 0.97);
        self.bg.user_message = Vec4::new(0.82, 0.42, 0.55, 1.0);
        self.bg.assistant_message = Vec4::new(0.72, 0.58, 0.78, 1.0);
        self.bg.muted_message = Vec4::new(0.88, 0.82, 0.86, 0.85);
        self.text.primary = Vec4::new(0.20, 0.12, 0.18, 1.0);
        self.text.secondary = Vec4::new(0.38, 0.28, 0.36, 1.0);
        self.text.tertiary = Vec4::new(0.50, 0.42, 0.48, 1.0);
        self.text.placeholder = Vec4::new(0.50, 0.42, 0.48, 0.72);
        self.text.ghost = Vec4::new(0.50, 0.42, 0.48, 0.45);
        self.text.accent = Vec4::new(0.35, 0.15, 0.28, 1.0);
        self.accent.phosphor = Vec4::new(0.95, 0.80, 0.88, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.65, 0.35, 0.50, 1.0);
        self.accent.pop = Vec4::new(0.78, 0.30, 0.45, 1.0);
        self.accent.pop_cool = Vec4::new(0.55, 0.38, 0.72, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.90, 0.82, 0.88, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.82, 0.68, 0.78, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.99, 0.94, 0.97, 1.0);
        self.button.primary = Vec4::new(0.72, 0.40, 0.58, 1.0);
        self.button.primary_hover = Vec4::new(0.78, 0.48, 0.64, 1.0);
        self.button.primary_active = Vec4::new(0.62, 0.34, 0.50, 1.0);
        self.button.secondary = Vec4::new(0.88, 0.78, 0.86, 1.0);
        self.button.secondary_hover = Vec4::new(0.92, 0.84, 0.90, 1.0);
        self.button.secondary_active = Vec4::new(0.82, 0.72, 0.80, 1.0);
        self.border.subtle = Vec4::new(0.55, 0.35, 0.45, 0.35);
        self.border.phosphor = Vec4::new(0.45, 0.22, 0.35, 0.55);
        self.border.instrument_rule = Vec4::new(0.75, 0.55, 0.65, 0.85);
        self.border.hover = Vec4::new(0.75, 0.45, 0.58, 0.45);
        self.border.focused = Vec4::new(0.72, 0.40, 0.82, 0.78);
        self.border.accent = Vec4::new(0.78, 0.38, 0.50, 0.55);
        self.highlight.hover = Vec4::new(0.88, 0.78, 0.86, 1.0);
        self.highlight.active = Vec4::new(0.84, 0.72, 0.80, 1.0);
        self.highlight.selection = Vec4::new(0.92, 0.65, 0.78, 0.22);
        self.backdrop.parallax_slow = Vec4::new(0.75, 0.45, 0.58, 0.045);
        self.backdrop.parallax_fast = Vec4::new(0.75, 0.45, 0.58, 0.032);
        self.backdrop.depth_band = Vec4::new(0.60, 0.35, 0.48, 0.08);
        self.markdown.code_foreground = Vec4::new(0.45, 0.22, 0.35, 1.0);
        self.editor.selection_band = Vec4::new(0.72, 0.48, 0.82, 0.28);
        self.editor.match_highlight_text = self.accent.pop;
        self.graph.pin_icon_tint = self.accent.pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    fn apply_springtime_light(&mut self) {
        self.bg.primary = Vec4::new(0.94, 0.98, 0.95, 1.0);
        self.bg.chrome_deck = Vec4::new(0.90, 0.96, 0.92, 1.0);
        self.bg.panel_well = Vec4::new(0.84, 0.92, 0.88, 1.0);
        self.bg.input_focused = Vec4::new(0.78, 0.90, 0.84, 1.0);
        self.bg.panel_popup = Vec4::new(0.96, 0.99, 0.97, 0.97);
        self.bg.user_message = Vec4::new(0.35, 0.62, 0.45, 1.0);
        self.bg.assistant_message = Vec4::new(0.48, 0.68, 0.58, 1.0);
        self.bg.muted_message = Vec4::new(0.82, 0.90, 0.86, 0.85);
        self.text.primary = Vec4::new(0.12, 0.22, 0.16, 1.0);
        self.text.secondary = Vec4::new(0.28, 0.42, 0.34, 1.0);
        self.text.tertiary = Vec4::new(0.36, 0.50, 0.42, 1.0);
        self.text.placeholder = Vec4::new(0.36, 0.50, 0.42, 0.72);
        self.text.ghost = Vec4::new(0.36, 0.50, 0.42, 0.45);
        self.text.accent = Vec4::new(0.14, 0.38, 0.28, 1.0);
        self.accent.phosphor = Vec4::new(0.80, 0.92, 0.85, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.22, 0.52, 0.38, 1.0);
        self.accent.pop = Vec4::new(0.28, 0.58, 0.42, 1.0);
        self.accent.pop_cool = Vec4::new(0.30, 0.48, 0.62, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.82, 0.92, 0.88, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.58, 0.78, 0.68, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.94, 0.99, 0.96, 1.0);
        self.button.primary = Vec4::new(0.28, 0.52, 0.40, 1.0);
        self.button.primary_hover = Vec4::new(0.34, 0.60, 0.48, 1.0);
        self.button.primary_active = Vec4::new(0.22, 0.44, 0.32, 1.0);
        self.button.secondary = Vec4::new(0.78, 0.88, 0.82, 1.0);
        self.button.secondary_hover = Vec4::new(0.84, 0.92, 0.86, 1.0);
        self.button.secondary_active = Vec4::new(0.72, 0.82, 0.76, 1.0);
        self.border.subtle = Vec4::new(0.32, 0.48, 0.40, 0.35);
        self.border.phosphor = Vec4::new(0.28, 0.45, 0.36, 0.55);
        self.border.instrument_rule = Vec4::new(0.55, 0.72, 0.62, 0.85);
        self.border.hover = Vec4::new(0.45, 0.65, 0.52, 0.45);
        self.border.focused = Vec4::new(0.35, 0.62, 0.72, 0.78);
        self.border.accent = Vec4::new(0.38, 0.62, 0.48, 0.55);
        self.highlight.hover = Vec4::new(0.78, 0.90, 0.84, 1.0);
        self.highlight.active = Vec4::new(0.72, 0.86, 0.78, 1.0);
        self.highlight.selection = Vec4::new(0.55, 0.82, 0.65, 0.22);
        self.backdrop.parallax_slow = Vec4::new(0.35, 0.62, 0.48, 0.045);
        self.backdrop.parallax_fast = Vec4::new(0.35, 0.62, 0.48, 0.032);
        self.backdrop.depth_band = Vec4::new(0.28, 0.48, 0.38, 0.08);
        self.markdown.code_foreground = Vec4::new(0.18, 0.38, 0.28, 1.0);
        self.editor.selection_band = Vec4::new(0.35, 0.58, 0.65, 0.28);
        self.editor.match_highlight_text = self.accent.pop;
        self.graph.pin_icon_tint = self.accent.pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    fn apply_forest_dark(&mut self) {
        self.bg.primary = Vec4::new(0.06, 0.12, 0.10, 1.0);
        self.bg.chrome_deck = Vec4::new(0.08, 0.16, 0.12, 1.0);
        self.bg.panel_well = Vec4::new(0.12, 0.24, 0.18, 1.0);
        self.bg.input_focused = Vec4::new(0.14, 0.30, 0.22, 1.0);
        self.bg.panel_popup = Vec4::new(0.06, 0.14, 0.10, 0.97);
        self.bg.user_message = Vec4::new(0.45, 0.62, 0.38, 1.0);
        self.bg.assistant_message = Vec4::new(0.22, 0.40, 0.32, 1.0);
        self.bg.muted_message = Vec4::new(0.08, 0.14, 0.11, 0.74);
        self.text.primary = Vec4::new(0.90, 0.94, 0.88, 1.0);
        self.text.secondary = Vec4::new(0.62, 0.76, 0.68, 1.0);
        self.text.tertiary = Vec4::new(0.45, 0.58, 0.50, 1.0);
        self.text.placeholder = Vec4::new(0.45, 0.58, 0.50, 0.72);
        self.text.ghost = Vec4::new(0.45, 0.58, 0.50, 0.45);
        self.text.accent = Vec4::new(0.82, 0.94, 0.78, 1.0);
        self.accent.phosphor = Vec4::new(0.78, 0.90, 0.72, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.18, 0.42, 0.30, 1.0);
        self.accent.pop = Vec4::new(0.55, 0.78, 0.42, 1.0);
        self.accent.pop_cool = Vec4::new(0.38, 0.62, 0.58, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.06, 0.14, 0.10, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.22, 0.42, 0.32, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.82, 0.90, 0.78, 1.0);
        self.button.primary = Vec4::new(0.10, 0.22, 0.16, 1.0);
        self.button.primary_hover = Vec4::new(0.14, 0.30, 0.22, 1.0);
        self.button.primary_active = Vec4::new(0.06, 0.14, 0.10, 1.0);
        self.button.secondary = Vec4::new(0.22, 0.40, 0.30, 1.0);
        self.button.secondary_hover = Vec4::new(0.28, 0.48, 0.36, 1.0);
        self.button.secondary_active = Vec4::new(0.18, 0.34, 0.25, 1.0);
        self.border.subtle = Vec4::new(0.72, 0.82, 0.74, 0.38);
        self.border.phosphor = Vec4::new(0.75, 0.86, 0.78, 0.68);
        self.border.instrument_rule = Vec4::new(0.78, 0.88, 0.80, 0.88);
        self.border.hover = Vec4::new(0.65, 0.78, 0.68, 0.45);
        self.border.focused = Vec4::new(0.48, 0.72, 0.58, 0.78);
        self.border.accent = Vec4::new(0.52, 0.72, 0.45, 0.52);
        self.highlight.hover = Vec4::new(0.18, 0.32, 0.24, 1.0);
        self.highlight.active = Vec4::new(0.22, 0.38, 0.28, 1.0);
        self.highlight.selection = Vec4::new(0.70, 0.85, 0.65, 0.14);
        self.backdrop.parallax_slow = Vec4::new(0.22, 0.45, 0.34, 0.038);
        self.backdrop.parallax_fast = Vec4::new(0.22, 0.45, 0.34, 0.026);
        self.backdrop.depth_band = Vec4::new(0.10, 0.24, 0.16, 0.08);
        self.markdown.code_foreground = Vec4::new(0.78, 0.88, 0.72, 1.0);
        self.editor.selection_band = Vec4::new(0.38, 0.62, 0.55, 0.28);
        self.editor.match_highlight_text = self.accent.pop;
        self.graph.pin_icon_tint = self.accent.pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    fn apply_toadstool_light(&mut self) {
        self.bg.primary = Vec4::new(0.97, 0.93, 0.89, 1.0);
        self.bg.chrome_deck = Vec4::new(0.94, 0.88, 0.82, 1.0);
        self.bg.panel_well = Vec4::new(0.90, 0.82, 0.74, 1.0);
        self.bg.input_focused = Vec4::new(0.88, 0.78, 0.68, 1.0);
        self.bg.panel_popup = Vec4::new(0.99, 0.96, 0.92, 0.97);
        self.bg.user_message = Vec4::new(0.75, 0.35, 0.30, 1.0);
        self.bg.assistant_message = Vec4::new(0.82, 0.55, 0.38, 1.0);
        self.bg.muted_message = Vec4::new(0.92, 0.84, 0.76, 0.85);
        self.text.primary = Vec4::new(0.22, 0.14, 0.10, 1.0);
        self.text.secondary = Vec4::new(0.42, 0.32, 0.26, 1.0);
        self.text.tertiary = Vec4::new(0.55, 0.45, 0.38, 1.0);
        self.text.placeholder = Vec4::new(0.55, 0.45, 0.38, 0.72);
        self.text.ghost = Vec4::new(0.55, 0.45, 0.38, 0.45);
        self.text.accent = Vec4::new(0.55, 0.22, 0.15, 1.0);
        self.accent.phosphor = Vec4::new(0.96, 0.88, 0.78, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.55, 0.28, 0.18, 1.0);
        self.accent.pop = Vec4::new(0.72, 0.32, 0.26, 1.0);
        self.accent.pop_cool = Vec4::new(0.42, 0.48, 0.65, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.90, 0.82, 0.74, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.85, 0.65, 0.48, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.98, 0.94, 0.88, 1.0);
        self.button.primary = Vec4::new(0.62, 0.32, 0.24, 1.0);
        self.button.primary_hover = Vec4::new(0.72, 0.38, 0.28, 1.0);
        self.button.primary_active = Vec4::new(0.52, 0.26, 0.20, 1.0);
        self.button.secondary = Vec4::new(0.88, 0.76, 0.65, 1.0);
        self.button.secondary_hover = Vec4::new(0.92, 0.82, 0.72, 1.0);
        self.button.secondary_active = Vec4::new(0.80, 0.70, 0.58, 1.0);
        self.border.subtle = Vec4::new(0.55, 0.38, 0.28, 0.35);
        self.border.phosphor = Vec4::new(0.50, 0.32, 0.22, 0.55);
        self.border.instrument_rule = Vec4::new(0.82, 0.68, 0.52, 0.85);
        self.border.hover = Vec4::new(0.72, 0.52, 0.38, 0.45);
        self.border.focused = Vec4::new(0.55, 0.48, 0.88, 0.78);
        self.border.accent = Vec4::new(0.72, 0.42, 0.32, 0.55);
        self.highlight.hover = Vec4::new(0.92, 0.84, 0.74, 1.0);
        self.highlight.active = Vec4::new(0.88, 0.78, 0.66, 1.0);
        self.highlight.selection = Vec4::new(0.95, 0.72, 0.55, 0.22);
        self.backdrop.parallax_slow = Vec4::new(0.62, 0.38, 0.28, 0.045);
        self.backdrop.parallax_fast = Vec4::new(0.62, 0.38, 0.28, 0.032);
        self.backdrop.depth_band = Vec4::new(0.48, 0.28, 0.20, 0.08);
        self.markdown.code_foreground = Vec4::new(0.42, 0.25, 0.18, 1.0);
        self.editor.selection_band = Vec4::new(0.65, 0.48, 0.38, 0.28);
        self.editor.match_highlight_text = self.accent.pop;
        self.graph.pin_icon_tint = self.accent.pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    fn apply_acorn_dark(&mut self) {
        self.bg.primary = Vec4::new(0.14, 0.10, 0.08, 1.0);
        self.bg.chrome_deck = Vec4::new(0.18, 0.13, 0.10, 1.0);
        self.bg.panel_well = Vec4::new(0.28, 0.20, 0.14, 1.0);
        self.bg.input_focused = Vec4::new(0.34, 0.24, 0.16, 1.0);
        self.bg.panel_popup = Vec4::new(0.16, 0.11, 0.08, 0.97);
        self.bg.user_message = Vec4::new(0.65, 0.42, 0.28, 1.0);
        self.bg.assistant_message = Vec4::new(0.42, 0.34, 0.26, 1.0);
        self.bg.muted_message = Vec4::new(0.20, 0.15, 0.11, 0.74);
        self.text.primary = Vec4::new(0.94, 0.88, 0.78, 1.0);
        self.text.secondary = Vec4::new(0.78, 0.68, 0.55, 1.0);
        self.text.tertiary = Vec4::new(0.58, 0.50, 0.42, 1.0);
        self.text.placeholder = Vec4::new(0.58, 0.50, 0.42, 0.72);
        self.text.ghost = Vec4::new(0.58, 0.50, 0.42, 0.45);
        self.text.accent = Vec4::new(0.98, 0.92, 0.72, 1.0);
        self.accent.phosphor = Vec4::new(0.92, 0.84, 0.68, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.45, 0.32, 0.18, 1.0);
        self.accent.pop = Vec4::new(0.82, 0.52, 0.28, 1.0);
        self.accent.pop_cool = Vec4::new(0.55, 0.60, 0.75, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.12, 0.09, 0.07, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.42, 0.30, 0.20, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.92, 0.86, 0.76, 1.0);
        self.button.primary = Vec4::new(0.16, 0.11, 0.08, 1.0);
        self.button.primary_hover = Vec4::new(0.24, 0.17, 0.12, 1.0);
        self.button.primary_active = Vec4::new(0.10, 0.07, 0.05, 1.0);
        self.button.secondary = Vec4::new(0.38, 0.28, 0.20, 1.0);
        self.button.secondary_hover = Vec4::new(0.46, 0.34, 0.24, 1.0);
        self.button.secondary_active = Vec4::new(0.30, 0.22, 0.16, 1.0);
        self.border.subtle = Vec4::new(0.88, 0.80, 0.68, 0.38);
        self.border.phosphor = Vec4::new(0.85, 0.78, 0.65, 0.68);
        self.border.instrument_rule = Vec4::new(0.90, 0.84, 0.70, 0.88);
        self.border.hover = Vec4::new(0.78, 0.68, 0.55, 0.45);
        self.border.focused = Vec4::new(0.65, 0.55, 0.88, 0.78);
        self.border.accent = Vec4::new(0.75, 0.52, 0.38, 0.52);
        self.highlight.hover = Vec4::new(0.32, 0.24, 0.18, 1.0);
        self.highlight.active = Vec4::new(0.38, 0.28, 0.20, 1.0);
        self.highlight.selection = Vec4::new(0.88, 0.76, 0.45, 0.18);
        self.backdrop.parallax_slow = Vec4::new(0.45, 0.32, 0.22, 0.038);
        self.backdrop.parallax_fast = Vec4::new(0.45, 0.32, 0.22, 0.026);
        self.backdrop.depth_band = Vec4::new(0.28, 0.18, 0.12, 0.08);
        self.markdown.code_foreground = Vec4::new(0.86, 0.78, 0.65, 1.0);
        self.editor.selection_band = Vec4::new(0.60, 0.55, 0.72, 0.28);
        self.editor.match_highlight_text = self.accent.pop;
        self.graph.pin_icon_tint = self.accent.pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    fn apply_basic_light(&mut self) {
        self.bg.primary = Vec4::new(0.92, 0.92, 0.94, 1.0);
        self.bg.chrome_deck = Vec4::new(0.88, 0.88, 0.90, 1.0);
        self.bg.panel_well = Vec4::new(0.82, 0.82, 0.86, 1.0);
        self.bg.input_focused = Vec4::new(0.78, 0.78, 0.84, 1.0);
        self.bg.panel_popup = Vec4::new(0.98, 0.98, 0.99, 0.97);
        self.bg.user_message = Vec4::new(0.35, 0.48, 0.72, 1.0);
        self.bg.assistant_message = Vec4::new(0.55, 0.58, 0.65, 1.0);
        self.bg.muted_message = Vec4::new(0.86, 0.86, 0.90, 0.85);
        self.text.primary = Vec4::new(0.12, 0.12, 0.14, 1.0);
        self.text.secondary = Vec4::new(0.38, 0.38, 0.42, 1.0);
        self.text.tertiary = Vec4::new(0.50, 0.50, 0.55, 1.0);
        self.text.placeholder = Vec4::new(0.50, 0.50, 0.55, 0.72);
        self.text.ghost = Vec4::new(0.50, 0.50, 0.55, 0.45);
        self.text.accent = Vec4::new(0.18, 0.22, 0.32, 1.0);
        self.accent.phosphor = Vec4::new(0.90, 0.90, 0.92, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.28, 0.32, 0.45, 1.0);
        self.accent.pop = Vec4::new(0.45, 0.38, 0.78, 1.0);
        self.accent.pop_cool = Vec4::new(0.22, 0.48, 0.62, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.82, 0.82, 0.86, 0.96);
        self.chrome.tab_bar_slider = Vec4::new(0.58, 0.60, 0.68, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.96, 0.96, 0.98, 1.0);
        self.button.primary = Vec4::new(0.25, 0.28, 0.38, 1.0);
        self.button.primary_hover = Vec4::new(0.32, 0.36, 0.48, 1.0);
        self.button.primary_active = Vec4::new(0.18, 0.20, 0.28, 1.0);
        self.button.secondary = Vec4::new(0.72, 0.72, 0.78, 1.0);
        self.button.secondary_hover = Vec4::new(0.78, 0.78, 0.84, 1.0);
        self.button.secondary_active = Vec4::new(0.66, 0.66, 0.72, 1.0);
        self.border.subtle = Vec4::new(0.30, 0.30, 0.35, 0.35);
        self.border.phosphor = Vec4::new(0.35, 0.35, 0.40, 0.55);
        self.border.instrument_rule = Vec4::new(0.65, 0.65, 0.70, 0.85);
        self.border.hover = Vec4::new(0.55, 0.55, 0.62, 0.45);
        self.border.focused = Vec4::new(0.35, 0.48, 0.78, 0.78);
        self.border.accent = Vec4::new(0.45, 0.50, 0.62, 0.55);
        self.highlight.hover = Vec4::new(0.85, 0.86, 0.92, 1.0);
        self.highlight.active = Vec4::new(0.78, 0.80, 0.88, 1.0);
        self.highlight.selection = Vec4::new(0.55, 0.62, 0.88, 0.20);
        self.backdrop.parallax_slow = Vec4::new(0.35, 0.38, 0.48, 0.04);
        self.backdrop.parallax_fast = Vec4::new(0.35, 0.38, 0.48, 0.03);
        self.backdrop.depth_band = Vec4::new(0.25, 0.28, 0.38, 0.07);
        self.markdown.code_foreground = Vec4::new(0.22, 0.24, 0.32, 1.0);
        self.editor.selection_band = Vec4::new(0.35, 0.45, 0.65, 0.28);
        self.editor.match_highlight_text = self.accent.pop;
        self.graph.pin_icon_tint = self.accent.pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }

    fn apply_dark_contrast(&mut self) {
        self.bg.primary = Vec4::new(0.03, 0.03, 0.05, 1.0);
        self.bg.chrome_deck = Vec4::new(0.05, 0.05, 0.08, 1.0);
        self.bg.panel_well = Vec4::new(0.10, 0.10, 0.14, 1.0);
        self.bg.input_focused = Vec4::new(0.14, 0.14, 0.20, 1.0);
        self.bg.panel_popup = Vec4::new(0.04, 0.04, 0.07, 0.97);
        self.bg.user_message = Vec4::new(0.85, 0.35, 0.30, 1.0);
        self.bg.assistant_message = Vec4::new(0.35, 0.40, 0.55, 1.0);
        self.bg.muted_message = Vec4::new(0.06, 0.06, 0.10, 0.82);
        self.text.primary = Vec4::new(0.98, 0.98, 1.0, 1.0);
        self.text.secondary = Vec4::new(0.78, 0.80, 0.88, 1.0);
        self.text.tertiary = Vec4::new(0.58, 0.60, 0.70, 1.0);
        self.text.placeholder = Vec4::new(0.58, 0.60, 0.70, 0.72);
        self.text.ghost = Vec4::new(0.58, 0.60, 0.70, 0.45);
        self.text.accent = Vec4::new(1.0, 0.98, 0.92, 1.0);
        self.accent.phosphor = Vec4::new(0.95, 0.95, 0.98, 1.0);
        self.accent.phosphor_glow = Vec4::new(0.20, 0.25, 0.45, 1.0);
        self.accent.pop = Vec4::new(0.95, 0.45, 0.40, 1.0);
        self.accent.pop_cool = Vec4::new(0.50, 0.60, 0.95, 1.0);
        self.chrome.tab_bar_bg = Vec4::new(0.04, 0.04, 0.08, 0.98);
        self.chrome.tab_bar_slider = Vec4::new(0.35, 0.38, 0.55, 1.0);
        self.chrome.composer_backplate = Vec4::new(0.88, 0.88, 0.92, 1.0);
        self.button.primary = Vec4::new(0.08, 0.10, 0.18, 1.0);
        self.button.primary_hover = Vec4::new(0.14, 0.16, 0.28, 1.0);
        self.button.primary_active = Vec4::new(0.05, 0.06, 0.12, 1.0);
        self.button.secondary = Vec4::new(0.28, 0.30, 0.45, 1.0);
        self.button.secondary_hover = Vec4::new(0.36, 0.38, 0.55, 1.0);
        self.button.secondary_active = Vec4::new(0.22, 0.24, 0.38, 1.0);
        self.border.subtle = Vec4::new(0.90, 0.90, 0.95, 0.45);
        self.border.phosphor = Vec4::new(0.92, 0.92, 0.96, 0.75);
        self.border.instrument_rule = Vec4::new(0.95, 0.95, 0.98, 0.92);
        self.border.hover = Vec4::new(0.75, 0.78, 0.90, 0.55);
        self.border.focused = Vec4::new(0.45, 0.58, 0.98, 0.85);
        self.border.accent = Vec4::new(0.85, 0.45, 0.40, 0.6);
        self.highlight.hover = Vec4::new(0.18, 0.18, 0.28, 1.0);
        self.highlight.active = Vec4::new(0.24, 0.24, 0.38, 1.0);
        self.highlight.selection = Vec4::new(0.45, 0.55, 0.95, 0.22);
        self.backdrop.parallax_slow = Vec4::new(0.25, 0.30, 0.50, 0.05);
        self.backdrop.parallax_fast = Vec4::new(0.25, 0.30, 0.50, 0.035);
        self.backdrop.depth_band = Vec4::new(0.08, 0.10, 0.20, 0.10);
        self.markdown.code_foreground = Vec4::new(0.90, 0.88, 0.95, 1.0);
        self.editor.selection_band = Vec4::new(0.45, 0.55, 0.92, 0.32);
        self.editor.match_highlight_text = self.accent.pop;
        self.graph.pin_icon_tint = self.accent.pop;
        self.bg.shard_backplate =
            Self::interpolate_shard_backplate(self.bg.chrome_deck, self.bg.panel_well);
    }
}

pub fn theme_index_for_id(theme_id: &str) -> usize {
    let id = theme_id.trim();
    THEME_CHOICES
        .iter()
        .position(|(s, _)| *s == id)
        .unwrap_or(0)
}

pub fn theme_id_for_index(index: usize) -> &'static str {
    THEME_CHOICES
        .get(index)
        .map(|(id, _)| *id)
        .unwrap_or("standard")
}
