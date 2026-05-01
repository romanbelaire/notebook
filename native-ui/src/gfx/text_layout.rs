//! Parley layout built once per cache key; measurement, glyph positions, and Vello draw reuse it.

use parley::layout::{Alignment, Layout, PositionedLayoutItem};
use parley::style::{FontStyle, FontWeight, StyleProperty};
use parley::{FontContext, LayoutContext};
use vello::peniko::{Brush, Color as VelloColor, Fill};
use vello::Scene;
use vello::{peniko::kurbo::Affine, Glyph};

/// Brush fingerprint for cache keys when only measurement / wrapped layout is needed (no draw).
pub const PARAGRAPH_MEASURE_BRUSH_BITS: u64 = 0;

pub const MAX_PARAGRAPH_CACHE_ENTRIES: usize = 512;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ParagraphCacheKey {
    pub text: String,
    pub size_bits: u32,
    pub max_width_bits: u32,
    pub brush_bits: u64,
    /// bit0 bold, bit1 italic (matches Parley font resolution).
    pub style_bits: u8,
}

impl ParagraphCacheKey {
    pub fn new(
        text: &str,
        size: f32,
        max_width: Option<f32>,
        brush_bits: u64,
        bold: bool,
        italic: bool,
    ) -> Self {
        let style_bits = (bold as u8) | ((italic as u8) << 1);
        Self {
            text: text.to_string(),
            size_bits: (size * 1000.0) as u32,
            max_width_bits: max_width
                .map(|w| (w * 1000.0).round() as u32)
                .unwrap_or(u32::MAX),
            brush_bits,
            style_bits,
        }
    }

    #[inline]
    pub fn bold(&self) -> bool {
        (self.style_bits & 1) != 0
    }

    #[inline]
    pub fn italic(&self) -> bool {
        (self.style_bits & 2) != 0
    }
}

pub fn brush_bits_from_rgba_f32(color: [f32; 4]) -> u64 {
    let r = (color[0] * 255.0).clamp(0.0, 255.0) as u64;
    let g = (color[1] * 255.0).clamp(0.0, 255.0) as u64;
    let b = (color[2] * 255.0).clamp(0.0, 255.0) as u64;
    let a = (color[3] * 255.0).clamp(0.0, 255.0) as u64;
    (r << 24) | (g << 16) | (b << 8) | a
}

pub fn measure_default_brush() -> Brush {
    Brush::Solid(VelloColor::WHITE)
}

/// Wrapped segment: last line top (`min_coord`), last line advance, `layout.width()`, and total line height sum.
#[derive(Clone, Copy, Default)]
pub struct ParagraphWrappedFlow {
    pub last_line_top: f32,
    pub last_line_advance: f32,
    pub last_line_height: f32,
    pub layout_width: f32,
    pub content_height: f32,
}

pub fn paragraph_wrapped_flow(layout: &Layout<Brush>) -> ParagraphWrappedFlow {
    let mut flow = ParagraphWrappedFlow {
        layout_width: layout.width(),
        ..Default::default()
    };
    for line in layout.lines() {
        let m = line.metrics();
        flow.last_line_top = m.min_coord;
        flow.last_line_advance = m.advance;
        flow.last_line_height = m.line_height;
        flow.content_height += m.line_height;
    }
    flow
}

pub struct CachedParagraph {
    pub layout: Layout<Brush>,
    pub first_width: f32,
    pub first_height: f32,
    pub first_baseline: f32,
    pub content_height: f32,
    /// Character-boundary x positions for `break_all_lines(None)` layout; relative to line start (add `start_x` when returning).
    pub glyph_x_unbounded: Vec<f32>,
    /// From `break_all_lines(Some(w))`: line height factor × font size, lines of x positions relative to line origin.
    pub wrapped_glyph_lines: Option<(f32, Vec<Vec<f32>>)>,
}

pub fn build_cached_paragraph(
    layout_context: &mut LayoutContext<Brush>,
    font_context: &mut FontContext,
    text: &str,
    font_size: f32,
    max_width: Option<f32>,
    brush: Brush,
    line_height_ratio: f32,
    bold: bool,
    italic: bool,
) -> CachedParagraph {
    let lh_default = font_size * line_height_ratio;

    if text.is_empty() {
        return CachedParagraph {
            layout: Layout::new(),
            first_width: 0.0,
            first_height: lh_default,
            first_baseline: font_size * 0.75,
            content_height: lh_default,
            glyph_x_unbounded: vec![0.0],
            wrapped_glyph_lines: None,
        };
    }

    let mut builder = layout_context.ranged_builder(font_context, text, 1.0);
    builder.push_default(StyleProperty::FontSize(font_size));
    builder.push_default(StyleProperty::FontWeight(if bold {
        FontWeight::BOLD
    } else {
        FontWeight::NORMAL
    }));
    builder.push_default(StyleProperty::FontStyle(if italic {
        FontStyle::Italic
    } else {
        FontStyle::Normal
    }));
    builder.push_default(StyleProperty::LineHeight(line_height_ratio));
    builder.push_default(StyleProperty::Brush(brush));

    let mut layout = builder.build(text);
    layout.break_all_lines(max_width);
    layout.align(max_width, Alignment::Start);

    let mut first_width = 0.0f32;
    let mut first_height = lh_default;
    let mut first_baseline = font_size * 0.75;

    for line in layout.lines() {
        let m = line.metrics();
        first_width = m.advance;
        first_height = m.size();
        first_baseline = m.baseline;
        break;
    }

    let mut content_height = 0.0f32;
    for line in layout.lines() {
        content_height += line.metrics().line_height;
    }
    if content_height == 0.0 {
        content_height = lh_default;
    }

    let glyph_x_unbounded = glyph_positions_unbounded(&layout, 0.0);

    let wrapped_glyph_lines = max_width.map(|_| {
        let line_h = lh_default;
        let mut lines = Vec::new();
        for line in layout.lines() {
            let mut positions = Vec::new();
            positions.push(0.0);
            let mut x = 0.0f32;
            for item in line.items() {
                if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                    x = glyph_run.offset();
                    for glyph in glyph_run.glyphs() {
                        x += glyph.advance;
                        positions.push(x);
                    }
                }
            }
            lines.push(positions);
        }
        if lines.is_empty() {
            lines.push(vec![0.0]);
        }
        (line_h, lines)
    });

    CachedParagraph {
        layout,
        first_width,
        first_height,
        first_baseline,
        content_height,
        glyph_x_unbounded,
        wrapped_glyph_lines,
    }
}

fn glyph_positions_unbounded(layout: &Layout<Brush>, start_x: f32) -> Vec<f32> {
    let mut positions = Vec::new();
    positions.push(start_x);
    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                let mut x = start_x + glyph_run.offset();
                for glyph in glyph_run.glyphs() {
                    x += glyph.advance;
                    positions.push(x);
                }
            }
        }
    }
    positions
}

/// Draw glyphs from a prepared Parley layout into a Vello scene. Returns total content height.
pub fn vello_draw_paragraph_layout(
    scene: &mut Scene,
    layout: &Layout<Brush>,
    x: f32,
    y: f32,
    font_size_fallback: f32,
    line_height_ratio: f32,
) -> f32 {
    let transform = Affine::translate((x as f64, y as f64));
    let mut total_height = 0.0f32;
    for line in layout.lines() {
        let lm = line.metrics();
        total_height += lm.line_height;
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                let run = glyph_run.run();
                let font = run.font();
                let font_size_val = run.font_size();
                let style = glyph_run.style();

                scene
                    .draw_glyphs(font)
                    .brush(&style.brush)
                    .transform(transform)
                    .font_size(font_size_val)
                    .draw(
                        Fill::NonZero,
                        glyph_run.positioned_glyphs().map(|g| Glyph {
                            id: g.id as u32,
                            x: g.x,
                            y: g.y,
                        }),
                    );
            }
        }
    }
    if total_height == 0.0 {
        font_size_fallback * line_height_ratio
    } else {
        total_height
    }
}
