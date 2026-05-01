use crate::app::App;
use crate::gfx::renderer::Renderer;
use crate::gfx::text_layout::{measure_default_brush, ParagraphCacheKey, PARAGRAPH_MEASURE_BRUSH_BITS};
use crate::gfx::types::Vertex;
use crate::ui::style;
use glam::{Vec2, Vec4};
use pulldown_cmark::{Event, Parser, Tag, TagEnd};

#[inline]
fn markdown_segment_color(is_code: bool, base: Vec4) -> Vec4 {
    if is_code {
        Vec4::new(0.86, 0.84, 0.86, 1.0)
    } else {
        base
    }
}

#[inline]
fn effective_message_font_size(font_size: f32, _bold: bool, _italic: bool, _code: bool) -> f32 {
    font_size
}

/// Layout width and total height of markdown body matching [`render_markdown_text`] line breaks and wraps.
pub(crate) fn measure_message_markdown(
    renderer: &mut Renderer,
    markdown: &str,
    max_width: f32,
    font_size: f32,
) -> Vec2 {
    if markdown.is_empty() {
        return Vec2::new(
            0.0,
            font_size * style::font_size::LINE_HEIGHT_RATIO,
        );
    }
    walk_markdown(
        renderer,
        markdown,
        Vec2::ZERO,
        max_width,
        font_size,
        Vec4::ONE,
        None,
        "",
    )
}

struct RenderContext<'a> {
    app: &'a App,
    vertices: &'a mut Vec<Vertex>,
    component_prefix: &'a str,
    record: Option<&'a mut Vec<(String, Vec2, f32, Vec4)>>,
}

fn advance_segment_width(
    renderer: &mut Renderer,
    max_content_w: &mut f32,
    current_x: &mut f32,
    start_x: f32,
    text: &str,
    size: f32,
    code: bool,
    text_color: Vec4,
) {
    let w = renderer.segment_width_unbounded_queue(
        text,
        size,
        markdown_segment_color(code, text_color),
    );
    *max_content_w = (*max_content_w).max(*current_x - start_x + w);
    *current_x += w;
}

/// Shared pulldown walk: measure-only when `render` is None; else queue draws. Returns (max_line_width, content_height).
fn walk_markdown(
    renderer: &mut Renderer,
    markdown: &str,
    start_pos: Vec2,
    max_width: f32,
    font_size: f32,
    text_color: Vec4,
    mut render: Option<RenderContext<'_>>,
    fallback_prefix: &str,
) -> Vec2 {
    let parser = Parser::new(markdown);
    let mut current_x = start_pos.x;
    let mut current_y = start_pos.y;
    let mut current_text = String::new();
    let mut is_code = false;
    let mut is_bold = false;
    let mut is_italic = false;
    let mut line_max_font = font_size;
    let mut max_content_w = 0.0f32;

    let component_prefix = render
        .as_ref()
        .map(|r| r.component_prefix)
        .unwrap_or(fallback_prefix);

    #[derive(Clone, Copy)]
    struct ListFrame {
        ordered: bool,
        next_n: u64,
    }
    let mut list_stack: Vec<ListFrame> = Vec::new();

    for event in parser {
        match event {
            Event::Start(Tag::List(first)) => {
                list_stack.push(ListFrame {
                    ordered: first.is_some(),
                    next_n: first.unwrap_or(1),
                });
            }
            Event::End(TagEnd::List(_)) => {
                list_stack.pop();
            }
            Event::Start(Tag::Item) => {
                let prefix = list_stack.last().map(|f| {
                    if f.ordered {
                        format!("{}. ", f.next_n)
                    } else {
                        "• ".to_string()
                    }
                });
                if let Some(p) = prefix {
                    current_text.push_str(&p);
                }
            }
            Event::End(TagEnd::Item) => {
                if let Some(f) = list_stack.last_mut() {
                    if f.ordered {
                        f.next_n += 1;
                    }
                }
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    flush_geometry_line(
                        renderer,
                        &mut max_content_w,
                        current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                    current_y += line_max_font * style::font_size::LINE_HEIGHT_RATIO;
                }
                current_x = start_pos.x;
                line_max_font = font_size;
            }
            Event::End(TagEnd::Paragraph) => {
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    flush_geometry_line(
                        renderer,
                        &mut max_content_w,
                        current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                }
                current_y += line_max_font * style::font_size::LINE_HEIGHT_RATIO;
                current_x = start_pos.x;
                line_max_font = font_size;
            }
            Event::Start(Tag::Strong) => {
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    advance_segment_width(
                        renderer,
                        &mut max_content_w,
                        &mut current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                }
                is_bold = true;
            }
            Event::End(pulldown_cmark::TagEnd::Strong) => {
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    advance_segment_width(
                        renderer,
                        &mut max_content_w,
                        &mut current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                }
                is_bold = false;
            }
            Event::Start(Tag::Emphasis) => {
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    advance_segment_width(
                        renderer,
                        &mut max_content_w,
                        &mut current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                }
                is_italic = true;
            }
            Event::End(pulldown_cmark::TagEnd::Emphasis) => {
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    advance_segment_width(
                        renderer,
                        &mut max_content_w,
                        &mut current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                }
                is_italic = false;
            }
            Event::Start(Tag::CodeBlock(_)) => {
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    advance_segment_width(
                        renderer,
                        &mut max_content_w,
                        &mut current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                }
                is_code = true;
            }
            Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    advance_segment_width(
                        renderer,
                        &mut max_content_w,
                        &mut current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                }
                is_code = false;
            }
            Event::Text(text) => {
                let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                let words: Vec<&str> = text.split_whitespace().collect();
                for word in words {
                    let word_with_space = if current_text.is_empty() {
                        word.to_string()
                    } else {
                        format!(" {}", word)
                    };
                    let test_text = format!("{}{}", current_text, word_with_space);
                    let line_used = current_x - start_pos.x;
                    let rem = (max_width - line_used).max(1.0);
                    let cand_key = ParagraphCacheKey::new(
                        &test_text,
                        size,
                        Some(rem),
                        PARAGRAPH_MEASURE_BRUSH_BITS,
                        is_bold,
                        is_italic,
                    );
                    let n_lines = renderer
                        .paragraph_cache_get_or_insert(
                            cand_key,
                            &test_text,
                            size,
                            Some(rem),
                            measure_default_brush(),
                        )
                        .layout
                        .len();
                    if n_lines > 1 && !current_text.is_empty() {
                        if let Some(ref mut r) = render {
                            render_text_segment(
                                renderer,
                                r.app,
                                &current_text,
                                Vec2::new(current_x, current_y),
                                size,
                                text_color,
                                is_code,
                                is_bold,
                                is_italic,
                                r.vertices,
                                component_prefix,
                                start_pos,
                                r.record.as_deref_mut(),
                            );
                        }
                        flush_geometry_line(
                            renderer,
                            &mut max_content_w,
                            current_x,
                            start_pos.x,
                            &current_text,
                            size,
                            is_code,
                            text_color,
                        );
                        line_max_font = line_max_font.max(size);
                        current_y += line_max_font * style::font_size::LINE_HEIGHT_RATIO;
                        current_x = start_pos.x;
                        line_max_font = font_size;
                        current_text = word.to_string();
                    } else {
                        current_text = test_text;
                    }
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if !current_text.is_empty() {
                    let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
                    if let Some(ref mut r) = render {
                        render_text_segment(
                            renderer,
                            r.app,
                            &current_text,
                            Vec2::new(current_x, current_y),
                            size,
                            text_color,
                            is_code,
                            is_bold,
                            is_italic,
                            r.vertices,
                            component_prefix,
                            start_pos,
                            r.record.as_deref_mut(),
                        );
                    }
                    flush_geometry_line(
                        renderer,
                        &mut max_content_w,
                        current_x,
                        start_pos.x,
                        &current_text,
                        size,
                        is_code,
                        text_color,
                    );
                    line_max_font = line_max_font.max(size);
                    current_text.clear();
                }
                current_y += line_max_font * style::font_size::LINE_HEIGHT_RATIO;
                current_x = start_pos.x;
                line_max_font = font_size;
            }
            _ => {}
        }
    }
    if !current_text.is_empty() {
        let size = effective_message_font_size(font_size, is_bold, is_italic, is_code);
        if let Some(ref mut r) = render {
            render_text_segment(
                renderer,
                r.app,
                &current_text,
                Vec2::new(current_x, current_y),
                size,
                text_color,
                is_code,
                is_bold,
                is_italic,
                r.vertices,
                component_prefix,
                start_pos,
                r.record.as_deref_mut(),
            );
        }
        flush_geometry_line(
            renderer,
            &mut max_content_w,
            current_x,
            start_pos.x,
            &current_text,
            size,
            is_code,
            text_color,
        );
        line_max_font = line_max_font.max(size);
    }

    let height = current_y - start_pos.y + line_max_font * style::font_size::LINE_HEIGHT_RATIO;
    Vec2::new(max_content_w.max(1.0), height.max(1.0))
}

fn flush_geometry_line(
    renderer: &mut Renderer,
    max_content_w: &mut f32,
    seg_x: f32,
    origin_x: f32,
    text: &str,
    size: f32,
    code: bool,
    text_color: Vec4,
) {
    let w = renderer.segment_width_unbounded_queue(
        text,
        size,
        markdown_segment_color(code, text_color),
    );
    *max_content_w = (*max_content_w).max((seg_x - origin_x) + w);
}

/// Render markdown text with basic formatting (bold, italic, code)
/// Returns the final Y position after rendering
pub(crate) fn render_markdown_text(
    renderer: &mut Renderer,
    app: &App,
    markdown: &str,
    start_pos: Vec2,
    max_width: f32,
    font_size: f32,
    text_color: Vec4,
    vertices: &mut Vec<Vertex>,
    component_prefix: &str,
    record: Option<&mut Vec<(String, Vec2, f32, Vec4)>>,
) -> f32 {
    let ctx = RenderContext {
        app,
        vertices,
        component_prefix,
        record,
    };
    let size = walk_markdown(
        renderer,
        markdown,
        start_pos,
        max_width,
        font_size,
        text_color,
        Some(ctx),
        component_prefix,
    );
    start_pos.y + size.y
}

/// Render a text segment with optional formatting (code, bold, italic).
/// font_size is already the effective size (scaled for bold/italic by the caller).
pub(crate) fn render_text_segment(
    renderer: &mut Renderer,
    _app: &App,
    text: &str,
    position: Vec2,
    font_size: f32,
    base_color: Vec4,
    is_code: bool,
    _is_bold: bool,
    _is_italic: bool,
    vertices: &mut Vec<Vertex>,
    component_prefix: &str,
    start_pos: Vec2,
    record: Option<&mut Vec<(String, Vec2, f32, Vec4)>>,
) {
    let color = if is_code {
        Vec4::new(0.86, 0.84, 0.86, 1.0)
    } else {
        base_color
    };
    let text_hash =
        format!("{:x}", text.chars().take(20).map(|c| c as u32).sum::<u32>());
    let component_id = format!(
        "{}_text_{}_{}_{}",
        component_prefix,
        text_hash,
        position.x as u32,
        position.y as u32
    );
    if !renderer.validate_component(&component_id, None, "MarkdownText") {
        return;
    }
    if let Some(rec) = record {
        rec.push((
            text.to_string(),
            position - start_pos,
            font_size,
            color,
        ));
    }
    renderer.queue_text(text, position, color, font_size, None);
}
