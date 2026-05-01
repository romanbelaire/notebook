use glam::{Vec2, Vec4};
use crate::stylus::{StylusEditor, Block, BlockType, BlockContent};
use crate::stylus::formatting::TextFormat;
use crate::gfx::renderer::Renderer;
use crate::gfx::types::{Quad, Vertex};
use crate::ui::components::Renderable;
use crate::ui::core::Rect;
use crate::ui::style;

pub struct StylusRenderer;

impl StylusRenderer {
    pub fn render_blocks(
        renderer: &mut Renderer,
        editor: &StylusEditor,
        _mouse_pos: Vec2,
        app: &crate::app::App,
        vertices: &mut Vec<Vertex>,
    ) {
        // Push stylus parent (inherits from current stack, typically "notepad")
        renderer.push_parent("stylus".to_string());
        renderer.validate_component("stylus", None, "Stylus");
        
        const PADDING: f32 = 20.0; // Match notepad component padding
        let mut y_offset = editor.position.y + PADDING - editor.scroll_offset;
        let block_spacing = 8.0;

        for block in &editor.document.blocks {
            let block_height = Self::get_block_height(block, editor.size.x - PADDING * 2.0);
            // Cull using full block extent (same as NotepadEditorViewport); using a fixed
            // line_height skips tall blocks whose top is above the viewport but body remains visible.
            if y_offset + block_height < editor.position.y {
                y_offset += block_height + block_spacing;
                continue;
            }
            if y_offset > editor.position.y + editor.size.y {
                break;
            }

            let block_rect = Vec2::new(editor.position.x + PADDING, y_offset);

            // Check if this block is focused
            let is_focused = editor.focused_block_id.as_ref()
                .map(|id| id == &block.id)
                .unwrap_or(false);

            // Render block content (backgrounds are rendered in main renderer)
            Self::render_block_content(renderer, block, block_rect, is_focused, editor, app, vertices);

            y_offset += block_height + block_spacing;
        }
        
        // Pop stylus parent
        renderer.pop_parent();
    }

    pub fn get_block_height_static(block: &Block, width: f32) -> f32 {
        Self::get_block_height(block, width)
    }

    fn get_block_height(block: &Block, width: f32) -> f32 {
        /// Count visual lines, respecting both explicit `\n` breaks and character-based word-wrap.
        fn count_lines(text: &str, chars_per_line: usize) -> f32 {
            let mut total = 0usize;
            for paragraph in text.split('\n') {
                total += (paragraph.len() as f32 / chars_per_line as f32).ceil() as usize;
                total = total.max(1); // blank lines still occupy one visual line
            }
            total.max(1) as f32
        }

        match &block.content {
            BlockContent::Text { text, .. } => {
                let font_size = Self::get_font_size(&block.block_type);
                let line_height = font_size * style::font_size::LINE_HEIGHT_RATIO;
                let chars_per_line = (width / (font_size * 0.6)).max(1.0) as usize;
                count_lines(text, chars_per_line) * line_height
            }
            BlockContent::Code { code, .. } => {
                let font_size = style::font_size::NORMAL;
                let line_height = font_size * style::font_size::LINE_HEIGHT_RATIO;
                let chars_per_line = (width / (font_size * 0.6)).max(1.0) as usize;
                count_lines(code, chars_per_line) * line_height
            }
            BlockContent::Divider => 20.0,
            BlockContent::Image { height, .. } => height.unwrap_or(200.0),
            _ => 40.0,  // Default height for other block types
        }
    }

    pub fn get_font_size_static(block_type: &BlockType) -> f32 {
        Self::get_font_size(block_type)
    }
    
    fn get_font_size(block_type: &BlockType) -> f32 {
        match block_type {
            BlockType::Heading1 => style::font_size::TITLE + 8.0,
            BlockType::Heading2 => style::font_size::TITLE + 4.0,
            BlockType::Heading3 => style::font_size::TITLE,
            BlockType::Heading4 => style::font_size::XLARGE,
            BlockType::Heading5 => style::font_size::LARGE,
            BlockType::Code => style::font_size::NORMAL,
            BlockType::Quote => style::font_size::MEDIUM,
            _ => style::font_size::MEDIUM,
        }
    }

    fn render_block_content(
        renderer: &mut Renderer,
        block: &Block,
        position: Vec2,
        is_focused: bool,
        editor: &StylusEditor,
        app: &crate::app::App,
        vertices: &mut Vec<Vertex>,
    ) {
        let font_size = Self::get_font_size(&block.block_type);
        let text_color = style::text::PRIMARY();

        match &block.content {
            BlockContent::Text { text, formats } => {
                // Render selection if present
                if let Some(ref selection) = editor.selection {
                    if selection.start.block_id == block.id && selection.end.block_id == block.id {
                        let start = selection.start.position.min(selection.end.position);
                        let end = selection.start.position.max(selection.end.position);
                        if let Some(text_ref) = block.content.get_text() {
                            if end <= text_ref.len() {
                                let positions =
                                    renderer.compute_glyph_positions(text_ref, font_size, position.x);
                                let c_start = text_ref[..start.min(text_ref.len())].chars().count();
                                let c_end = text_ref[..end.min(text_ref.len())].chars().count();
                                let ix = |i: usize| i.min(positions.len().saturating_sub(1));
                                let start_x = positions[ix(c_start)];
                                let end_x = positions[ix(c_end)];
                                let selection_rect = Quad {
                                    position: Vec2::new(start_x, position.y),
                                    size: Vec2::new(end_x - start_x, font_size + 4.0),
                                    color: style::editor::SELECTION_BAND(),
                                    corner_radius: 0.0,
                                    bubble_effect: false,
                                    slider_effect: false,
                                };
                                vertices.extend_from_slice(&selection_rect.to_vertices());
                            }
                        }
                    }
                }
                
                // Render text with formatting
                if !text.is_empty() {
                    // Sort formats by start position
                    let mut sorted_formats = formats.clone();
                    sorted_formats.sort_by_key(|f| f.start);
                    
                    // Render text in segments with formatting
                    let mut current_pos = 0;
                    let mut x_offset = 0.0;
                    
                    for format_span in &sorted_formats {
                        // Render text before this format using Text component
                        if current_pos < format_span.start {
                            let segment = &text[current_pos..format_span.start];
                            if !segment.is_empty() {
                                for (span_index, (sub, is_highlight)) in Self::segment_highlight_spans(segment, editor.highlight_term.as_deref()).into_iter().enumerate() {
                                    if sub.is_empty() { continue; }
                                    let color = if is_highlight {
                                        style::editor::MATCH_HIGHLIGHT_TEXT()
                                    } else {
                                        text_color
                                    };
                                    let segment_rect = Rect::new(
                                        position.x + x_offset,
                                        position.y,
                                        1000.0,
                                        font_size * 1.5,
                                    );
                                    let mut segment_text = crate::ui::text::Text::new_for_render(sub)
                                        .with_font_size(font_size)
                                        .with_color(color)
                                        .with_alignment(crate::ui::text::TextAlignment::Left);
                                    segment_text.update_layout(segment_rect, None, None);
                                    let segment_id = format!("stylus_text_segment_{}_{}", current_pos, span_index);
                                    renderer.push_parent(segment_id.clone());
                                    renderer.validate_component(&segment_id, None, "StylusTextSegment");
                                    segment_text.render(renderer, app, &mut Vec::new(), None);
                                    renderer.pop_parent();
                                    x_offset += renderer.measure_text(sub, font_size).x;
                                }
                            }
                        }
                        
                        // Render formatted text segment
                        let format_end = format_span.end.min(text.len());
                        if format_span.start < format_end {
                            let segment = &text[format_span.start..format_end];
                            if !segment.is_empty() {
                                let (format_color, format_size) = match &format_span.format {
                                    TextFormat::Bold => (Vec4::new(1.0, 1.0, 1.0, 1.0), font_size),
                                    TextFormat::Italic => (Vec4::new(0.80, 0.88, 1.0, 1.0), font_size),
                                    TextFormat::Underline => (text_color, font_size),
                                    TextFormat::Strikethrough => (Vec4::new(0.6, 0.6, 0.6, 0.85), font_size),
                                    TextFormat::Code => (Vec4::new(0.55, 1.0, 0.65, 1.0), font_size * 0.9),
                                    TextFormat::Link { .. } => (Vec4::new(0.45, 0.70, 1.0, 1.0), font_size),
                                };
                                let seg_x_start = position.x + x_offset;
                                for (span_index, (sub, is_highlight)) in Self::segment_highlight_spans(segment, editor.highlight_term.as_deref()).into_iter().enumerate() {
                                    if sub.is_empty() { continue; }
                                    let color = if is_highlight {
                                        style::editor::MATCH_HIGHLIGHT_TEXT()
                                    } else {
                                        format_color
                                    };
                                    let segment_rect = Rect::new(
                                        position.x + x_offset,
                                        position.y,
                                        1000.0,
                                        format_size * 1.5,
                                    );
                                    let mut segment_text = crate::ui::text::Text::new_for_render(sub)
                                        .with_font_size(format_size)
                                        .with_color(color)
                                        .with_alignment(crate::ui::text::TextAlignment::Left);
                                    segment_text.update_layout(segment_rect, None, None);
                                    let formatted_segment_id = format!("stylus_formatted_text_segment_{}_{}", current_pos, span_index);
                                    renderer.push_parent(formatted_segment_id.clone());
                                    renderer.validate_component(&formatted_segment_id, None, "StylusFormattedTextSegment");
                                    segment_text.render(renderer, app, &mut Vec::new(), None);
                                    renderer.pop_parent();
                                    x_offset += renderer.measure_text(sub, format_size).x;
                                }
                                let seg_x_end = position.x + x_offset;
                                let seg_width = (seg_x_end - seg_x_start).max(0.0);

                                // Bold: bright left-edge accent line
                                if matches!(format_span.format, TextFormat::Bold) {
                                    let accent = Quad {
                                        position: Vec2::new(seg_x_start - 2.0, position.y),
                                        size: Vec2::new(2.0, font_size),
                                        color: Vec4::new(1.0, 0.85, 0.3, 0.8),
                                        corner_radius: 1.0,
                                        bubble_effect: false,
                                        slider_effect: false,
                                    };
                                    vertices.extend_from_slice(&accent.to_vertices());
                                }

                                // Underline: thin line 2px below the text baseline
                                if matches!(format_span.format, TextFormat::Underline) && seg_width > 0.0 {
                                    let underline = Quad {
                                        position: Vec2::new(seg_x_start, position.y + font_size + 1.0),
                                        size: Vec2::new(seg_width, 1.5),
                                        color: text_color,
                                        corner_radius: 0.0,
                                        bubble_effect: false,
                                        slider_effect: false,
                                    };
                                    vertices.extend_from_slice(&underline.to_vertices());
                                }

                                // Strikethrough: line at mid-height
                                if matches!(format_span.format, TextFormat::Strikethrough) && seg_width > 0.0 {
                                    let strike = Quad {
                                        position: Vec2::new(seg_x_start, position.y + font_size * 0.45),
                                        size: Vec2::new(seg_width, 1.5),
                                        color: Vec4::new(0.6, 0.6, 0.6, 0.85),
                                        corner_radius: 0.0,
                                        bubble_effect: false,
                                        slider_effect: false,
                                    };
                                    vertices.extend_from_slice(&strike.to_vertices());
                                }

                                // Code: background highlight chip
                                if matches!(format_span.format, TextFormat::Code) && seg_width > 0.0 {
                                    let chip = Quad {
                                        position: Vec2::new(seg_x_start - 2.0, position.y - 1.0),
                                        size: Vec2::new(seg_width + 4.0, font_size + 4.0),
                                        color: Vec4::new(0.0, 0.18, 0.08, 0.45),
                                        corner_radius: 3.0,
                                        bubble_effect: false,
                                        slider_effect: false,
                                    };
                                    vertices.extend_from_slice(&chip.to_vertices());
                                }

                                // Link: underline in blue
                                if matches!(format_span.format, TextFormat::Link { .. }) && seg_width > 0.0 {
                                    let underline = Quad {
                                        position: Vec2::new(seg_x_start, position.y + font_size + 1.0),
                                        size: Vec2::new(seg_width, 1.5),
                                        color: Vec4::new(0.45, 0.70, 1.0, 0.9),
                                        corner_radius: 0.0,
                                        bubble_effect: false,
                                        slider_effect: false,
                                    };
                                    vertices.extend_from_slice(&underline.to_vertices());
                                }
                            }
                        }
                        
                        current_pos = format_end;
                    }
                    
                    // Render remaining text using Text component
                    if current_pos < text.len() {
                        let segment = &text[current_pos..];
                        if !segment.is_empty() {
                            for (span_index, (sub, is_highlight)) in Self::segment_highlight_spans(segment, editor.highlight_term.as_deref()).into_iter().enumerate() {
                                if sub.is_empty() { continue; }
                                let color = if is_highlight {
                                    style::editor::MATCH_HIGHLIGHT_TEXT()
                                } else {
                                    text_color
                                };
                                let segment_rect = Rect::new(
                                    position.x + x_offset,
                                    position.y,
                                    1000.0,
                                    font_size * 1.5,
                                );
                                let mut segment_text = crate::ui::text::Text::new_for_render(sub)
                                    .with_font_size(font_size)
                                    .with_color(color)
                                    .with_alignment(crate::ui::text::TextAlignment::Left);
                                segment_text.update_layout(segment_rect, None, None);
                                let remaining_segment_id = format!("stylus_text_segment_remaining_{}_{}", current_pos, span_index);
                                renderer.push_parent(remaining_segment_id.clone());
                                renderer.validate_component(&remaining_segment_id, None, "StylusTextSegment");
                                segment_text.render(renderer, app, &mut Vec::new(), None);
                                renderer.pop_parent();
                                x_offset += renderer.measure_text(sub, font_size).x;
                            }
                        }
                    }
                }

                // Render cursor if this block is focused - use unified cursor system
                if is_focused && app.cursor_visible {
                    if let Some(ref cursor) = editor.cursor {
                        if cursor.block_id == block.id {
                            // Calculate cursor position using glyph positions for proper alignment
                            let text_start_x = position.x;
                            
                            // Compute glyph positions for the entire text to support smooth interpolation
                            // We need positions for at least up to cursor.position, but compute for entire text
                            // to handle edge cases during animation
                            let max_pos = cursor.position.max((app.cursor_position_animation.value.ceil() as usize).min(text.len()));
                            let prefix = &text[..max_pos.min(text.len())];
                            let positions = renderer.compute_glyph_positions(prefix, font_size, text_start_x);
                            
                            // Use smoothly interpolated cursor position from app animation
                            let cursor_pos_float = app.cursor_position_animation.value;
                            let cursor_pos_floor = cursor_pos_float.floor() as usize;
                            let cursor_pos_ceil = cursor_pos_float.ceil() as usize;
                            let t = cursor_pos_float - cursor_pos_float.floor();
                            
                            let cursor_x = if cursor_pos_ceil < positions.len() && cursor_pos_floor < positions.len() {
                                // Interpolate between two glyph positions for smooth movement
                                let x1 = positions[cursor_pos_floor];
                                let x2 = positions[cursor_pos_ceil];
                                x1 + (x2 - x1) * t
                            } else if cursor_pos_floor < positions.len() {
                                positions[cursor_pos_floor]
                            } else {
                                // If at end or beyond, use last position or starting position
                                positions.last().copied().unwrap_or(text_start_x)
                            };
                            
                            // Use cursor helper from core for proper alignment
                            use crate::ui::core::cursor;
                            let block_rect = crate::ui::core::Rect::new(
                                position.x,
                                position.y,
                                1000.0, // Large width for calculation
                                font_size * 1.5, // Approximate line height
                            );
                            let cursor_rect = cursor::rect_at_position(&block_rect, cursor_x, font_size);
                            
                            // Render cursor as a quad (not text) for proper alignment
                            let cursor_quad = Quad {
                                position: cursor_rect.position(),
                                size: cursor_rect.size(),
                                color: style::text::PRIMARY(),
                                corner_radius: 0.0,
                                bubble_effect: false,
                                slider_effect: false,
                            };
                            vertices.extend_from_slice(&cursor_quad.to_vertices());
                        }
                    } else if text.is_empty() {
                        // Show cursor for empty focused block
                        use crate::ui::core::cursor;
                        let block_rect = crate::ui::core::Rect::new(
                            position.x,
                            position.y,
                            1000.0,
                            font_size * 1.5,
                        );
                        let cursor_rect = cursor::rect_at_position(&block_rect, position.x, font_size);
                        let cursor_quad = Quad {
                            position: cursor_rect.position(),
                            size: cursor_rect.size(),
                            color: style::text::PRIMARY(),
                            corner_radius: 0.0,
                            bubble_effect: false,
                            slider_effect: false,
                        };
                        vertices.extend_from_slice(&cursor_quad.to_vertices());
                    }
                }
            }
            BlockContent::Code { code, language: _ } => {
                // Code block text using Text component
                let code_color = style::markdown::CODE_FOREGROUND();
                let code_rect = Rect::new(
                    position.x,
                    position.y + 16.0,
                    1000.0,
                    20.0,
                );
                
                let mut code_text = crate::ui::text::Text::new_for_render(code)
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(code_color)
                    .with_alignment(crate::ui::text::TextAlignment::Left);
                code_text.update_layout(code_rect, None, None);
                
                let code_block_id = format!("stylus_code_block_{}", block.id);
                renderer.push_parent(code_block_id.clone());
                renderer.validate_component(&code_block_id, None, "StylusCodeBlock");
                code_text.render(renderer, app, &mut Vec::new(), None);
                renderer.pop_parent();
            }
            BlockContent::Divider => {
                // Render a horizontal line
                // This will be handled by the quad rendering
            }
            BlockContent::Image { url, alt, .. } => {
                // Image alt/URL text using Text component
                let display_text = alt.as_ref().unwrap_or(url);
                let image_text_rect = Rect::new(
                    position.x,
                    position.y + 20.0,
                    1000.0,
                    20.0,
                );
                
                let mut image_text = crate::ui::text::Text::new_for_render(display_text)
                    .with_font_size(style::font_size::NORMAL)
                    .with_color(text_color)
                    .with_alignment(crate::ui::text::TextAlignment::Left);
                image_text.update_layout(image_text_rect, None, None);
                
                let image_text_id = format!("stylus_image_text_{}", block.id);
                renderer.push_parent(image_text_id.clone());
                renderer.validate_component(&image_text_id, None, "StylusImageText");
                image_text.render(renderer, app, &mut Vec::new(), None);
                renderer.pop_parent();
            }
            _ => {
                // Render placeholder for other block types using Text component
                let placeholder = format!("[{}]", block.block_type.to_str());
                let placeholder_rect = Rect::new(
                    position.x,
                    position.y,
                    1000.0,
                    font_size * 1.5,
                );
                
                let mut placeholder_text = crate::ui::text::Text::new_for_render(&placeholder)
                    .with_font_size(font_size)
                    .with_color(text_color)
                    .with_alignment(crate::ui::text::TextAlignment::Left);
                placeholder_text.update_layout(placeholder_rect, None, None);
                
                let placeholder_id = format!("stylus_placeholder_{}", block.id);
                renderer.push_parent(placeholder_id.clone());
                renderer.validate_component(&placeholder_id, None, "StylusPlaceholder");
                placeholder_text.render(renderer, app, &mut Vec::new(), None);
                renderer.pop_parent();
            }
        }

        // Render slash command menu if active
        if editor.slash_command_active && is_focused {
            Self::render_slash_menu(renderer, editor, position, app);
        }
    }

    /// Split segment into (substring, is_highlight) spans for same-word highlighting.
    /// Highlighted spans are drawn in a high-contrast color (e.g. red) instead of the segment color.
    fn segment_highlight_spans<'a>(segment: &'a str, highlight_term: Option<&str>) -> Vec<(&'a str, bool)> {
        let term = match highlight_term {
            Some(t) if !t.is_empty() => t,
            _ => return vec![(segment, false)],
        };
        let mut result = Vec::new();
        let mut search_start = 0;
        while search_start < segment.len() {
            if let Some(rel_start) = segment[search_start..].find(term) {
                let start = search_start + rel_start;
                let end = (start + term.len()).min(segment.len());
                if start > search_start {
                    result.push((&segment[search_start..start], false));
                }
                result.push((&segment[start..end], true));
                search_start = end;
            } else {
                result.push((&segment[search_start..], false));
                break;
            }
        }
        result
    }

    fn get_cursor_x(renderer: &mut Renderer, text: &str, position: usize, font_size: f32) -> f32 {
        let prefix = &text[..position.min(text.len())];
        renderer.measure_text(prefix, font_size).x
    }

    fn render_slash_menu(
        renderer: &mut Renderer,
        editor: &StylusEditor,
        block_position: Vec2,
        app: &crate::app::App,
    ) {
        let menu_bg = Quad {
            position: block_position + Vec2::new(0.0, 30.0),
            size: Vec2::new(200.0, (editor.slash_command_matches.len() as f32 * 30.0).min(150.0)),
            color: style::bg::PANEL_POPUP(),
            corner_radius: style::corner_radius::SMALL,
            bubble_effect: false,
            slider_effect: false,
        };
        // Note: Menu background rendering would need vertices access
        
        // Slash command query text using Text component
        let query_text = format!("/{}", editor.slash_command_query);
        let query_rect = Rect::new(
            block_position.x,
            block_position.y + 50.0,
            300.0,
            20.0,
        );
        
        let mut query_text_component = crate::ui::text::Text::new_for_render(&query_text)
            .with_font_size(style::font_size::NORMAL)
            .with_color(style::text::PRIMARY())
            .with_alignment(crate::ui::text::TextAlignment::Left);
        query_text_component.update_layout(query_rect, None, None);
        
        let slash_query_id = "stylus_slash_query".to_string();
        renderer.push_parent(slash_query_id.clone());
        renderer.validate_component(&slash_query_id, None, "StylusSlashQuery");
        query_text_component.render(renderer, app, &mut Vec::new(), None);
        renderer.pop_parent();
        
        for (i, cmd) in editor.slash_command_matches.iter().take(5).enumerate() {
            let item_y = block_position.y + 60.0 + (i as f32 * 30.0);
            let _item_color = if i == editor.selected_slash_index {
                style::highlight::ACTIVE()
            } else {
                style::bg::INPUT()
            };
            // Note: Item background rendering would need vertices access
            
            // Slash command item text using Text component
            let item_rect = Rect::new(
                block_position.x + 10.0,
                item_y,
                200.0,
                20.0,
            );
            
            let mut item_text = crate::ui::text::Text::new_for_render(cmd.name())
                .with_font_size(style::font_size::NORMAL)
                .with_color(style::text::PRIMARY())
                .with_alignment(crate::ui::text::TextAlignment::Left);
            item_text.update_layout(item_rect, None, None);
            
            let slash_item_id = format!("stylus_slash_item_{}", i);
            renderer.push_parent(slash_item_id.clone());
            renderer.validate_component(&slash_item_id, None, "StylusSlashItem");
            item_text.render(renderer, app, &mut Vec::new(), None);
            renderer.pop_parent();
        }
    }
}
