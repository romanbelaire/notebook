use crate::gfx::types::{Vertex, Quad};
use crate::gfx::renderer::{CompositeLayer, Renderer};
use crate::app::App;
use glam::{Vec2, Vec4};
use crate::ui::tab_bar::Tab;
use crate::ui::style;
use crate::ui::style::elevation;
use crate::ui::core::Rect;
use crate::ui::{Text, TextAlignment};
use crate::ui::components::Renderable;
use crate::stylus::renderer::StylusRenderer;

/// Stylus editor + block backgrounds; state lives on `App::notepad_window`.
pub struct NotepadEditorViewport;

const NOTEPAD_EDITOR_VIEWPORT: NotepadEditorViewport = NotepadEditorViewport;

/// Opt-in drop shadow for the notepad editor viewport.
pub static SHADOW: crate::ui::shadow::ViewportShadow = crate::ui::shadow::ViewportShadow::new();

impl Renderable for NotepadEditorViewport {
    fn render(&self, renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>, _dirty_rect: Option<Rect>) {
        let Some(notepad) = app.notepad_window.as_ref() else {
            return;
        };
        let component_id = "notepad_editor_viewport";
        renderer.validate_component(component_id, None, "NotepadEditorViewport");

        if let Some(spec) = SHADOW.get() {
            if let Some(rect) = self.bounds_from_app(app) {
                renderer.queue_shadow(&rect, self.corner_radius(), &spec);
            }
        }

        let padding = style::padding::LARGE;
        let editor_area = Rect::new(
            notepad.editor.position.x,
            notepad.editor.position.y,
            notepad.editor.size.x,
            notepad.editor.size.y,
        );

        const BLOCK_CONTENT_PADDING: f32 = 20.0; // Must match StylusRenderer::PADDING
        let block_spacing = 8.0;
        let mut y_offset = editor_area.y + BLOCK_CONTENT_PADDING - notepad.editor.scroll_offset;

        for block in &notepad.editor.document.blocks {
            let block_height = StylusRenderer::get_block_height_static(block, notepad.editor.size.x - padding * 2.0);

            if y_offset + block_height < notepad.editor.position.y {
                y_offset += block_height + block_spacing;
                continue;
            }
            if y_offset > notepad.editor.position.y + notepad.editor.size.y {
                break;
            }

            let is_focused = notepad
                .editor
                .focused_block_id
                .as_ref()
                .map(|id| id == &block.id)
                .unwrap_or(false);

            let block_rect = Rect::new(
                notepad.editor.position.x,
                y_offset,
                notepad.editor.size.x,
                block_height,
            );

            // Only the focused block is a raised card; non-focused are flush with the page.
            if is_focused {
                renderer.queue_shadow(&block_rect, style::corner_radius::SMALL, &elevation::MEDIUM());
            }

            let block_bg = Quad {
                position: block_rect.position(),
                size: block_rect.size(),
                color: if is_focused {
                    style::bg::SECONDARY()
                } else {
                    Vec4::new(0.0, 0.0, 0.0, 0.0)
                },
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&block_bg.to_vertices());

            y_offset += block_height + block_spacing;
        }

        // Stylus glyphs must queue on HudChrome like the notepad chrome row (rendering.md):
        // MainContent Vello is composited before later passes and can leave body text invisible.
        renderer.set_composite_layer(CompositeLayer::HudChrome);
        StylusRenderer::render_blocks(renderer, &notepad.editor, app.mouse_pos, app, vertices);
        renderer.set_composite_layer(CompositeLayer::MainContent);
    }

    fn bounds(&self) -> Rect {
        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    fn bounds_from_app(&self, app: &App) -> Option<Rect> {
        app.notepad_window
            .as_ref()
            .map(|n| Rect::from_pos_size(n.editor.position, n.editor.size))
    }

    fn update_layout(
        &mut self,
        _available_rect: Rect,
        _dirty_rect: Option<Rect>,
        _app: Option<&App>,
    ) {
    }
}

pub fn render_notepad(renderer: &mut Renderer, app: &App, vertices: &mut Vec<Vertex>) {
    if app.ui_state.active_tab != Tab::Notepad {
        return;
    }

    if let Some(ref notepad) = app.notepad_window {
        renderer.set_composite_layer(CompositeLayer::MainContent);
        let bg = Quad {
            position: notepad.position,
            size: notepad.size,
            color: style::bg::PRIMARY(),
            corner_radius: 0.0,
            bubble_effect: false,
            slider_effect: false,
        };
        vertices.extend_from_slice(&bg.to_vertices());

        // Title row, actions, toolbar, @-mention list: queue Vello on HudChrome so glyphs composite
        // after MainContent quads the same way sidebar labels do (see docs/architecture/rendering.md).
        renderer.set_composite_layer(CompositeLayer::HudChrome);

        // Drop shadow + background card for the title row (title input + action buttons)
        {
            let ti = &notepad.title_input;
            let nb = &notepad.new_button;
            let db = &notepad.delete_button;
            // Compute bounding rect that covers title input and all icon buttons
            let row_x = ti.position.x;
            let row_y = ti.position.y;
            let row_right = (db.position.x + db.size.x).max(row_x);
            let row_bottom = (nb.position.y + nb.size.y).max(ti.position.y + ti.size.y);
            let header_rect = Rect::new(
                row_x - style::padding::SMALL,
                row_y - style::padding::SMALL,
                row_right - row_x + style::padding::SMALL * 2.0,
                row_bottom - row_y + style::padding::SMALL * 2.0,
            );
            renderer.queue_shadow(&header_rect, style::corner_radius::SMALL, &elevation::LOW());
            let header_bg = Quad {
                position: header_rect.position(),
                size: header_rect.size(),
                color: style::bg::SECONDARY(),
                corner_radius: style::corner_radius::SMALL,
                bubble_effect: false,
                slider_effect: false,
            };
            vertices.extend_from_slice(&header_bg.to_vertices());
        }

        renderer.validate_component("notepad_title_input", Some("notepad"), "TitleInput");
        renderer.push_parent("notepad_title_input".to_string());
        Renderable::render(&notepad.title_input, renderer, app, vertices, None);
        renderer.pop_parent();

        renderer.validate_component("notepad_buttons", Some("notepad"), "ButtonGroup");
        renderer.push_parent("notepad_buttons".to_string());
        Renderable::render(&notepad.new_button, renderer, app, vertices, None);
        Renderable::render(&notepad.save_button, renderer, app, vertices, None);
        Renderable::render(&notepad.open_button, renderer, app, vertices, None);
        Renderable::render(&notepad.delete_button, renderer, app, vertices, None);
        renderer.pop_parent();

        Renderable::render(&notepad.toolbar, renderer, app, vertices, None);

        if notepad.mention_popup_open {
            if let Some(rect) = notepad.mention_popup_rect() {
                let bg = Quad {
                    position: rect.position(),
                    size: rect.size(),
                    color: style::bg::SECONDARY(),
                    corner_radius: style::corner_radius::SMALL,
                    bubble_effect: false,
                    slider_effect: false,
                };
                vertices.extend_from_slice(&bg.to_vertices());
                for (i, row) in notepad.mention_rows.iter().enumerate().take(12) {
                    let row_y = rect.y + 4.0 + i as f32 * 28.0;
                    let row_rect = Rect::new(rect.x, row_y, rect.width, 28.0);
                    if i == notepad.mention_selected_index {
                        let hi = Quad {
                            position: row_rect.position(),
                            size: row_rect.size(),
                            color: style::highlight::SELECTION(),
                            corner_radius: 0.0,
                            bubble_effect: false,
                            slider_effect: false,
                        };
                        vertices.extend_from_slice(&hi.to_vertices());
                    }
                    let label = match row {
                        crate::ui::chat_window::MentionEntry::Paper(id) => app
                            .papers_cache
                            .iter()
                            .find(|p| p.id == *id)
                            .map(|p| {
                                format!(
                                    "Paper: {}",
                                    p.title.as_deref().unwrap_or(p.filename.as_str())
                                )
                            })
                            .unwrap_or_else(|| format!("Paper {}", id)),
                        crate::ui::chat_window::MentionEntry::Shard { graph_id, shard_id } => {
                            let ss = if shard_id.len() <= 10 {
                                shard_id.as_str()
                            } else {
                                &shard_id[..10]
                            };
                            let gs = if graph_id.len() <= 10 {
                                graph_id.as_str()
                            } else {
                                &graph_id[..10]
                            };
                            format!("Shard {} · graph {}", ss, gs)
                        }
                        crate::ui::chat_window::MentionEntry::Graph { graph_id } => {
                            let g = if graph_id.len() <= 20 {
                                graph_id.as_str()
                            } else {
                                &graph_id[..20]
                            };
                            format!("Graph {}", g)
                        }
                        crate::ui::chat_window::MentionEntry::Notepad {
                            title,
                            document_id,
                        } => {
                            format!("Note: {} ({})", title, document_id)
                        }
                    };
                    let mut row_text = Text::new_for_render(&label)
                        .with_font_size(style::font_size::NORMAL)
                        .with_color(style::text::PRIMARY())
                        .with_alignment(TextAlignment::Left);
                    let text_rect = row_rect.inset(8.0);
                    row_text.update_layout(text_rect, None, None);
                    let cid = format!("notepad_mention_row_{}", i);
                    renderer.push_parent(cid.clone());
                    renderer.validate_component(&cid, Some("notepad"), "MentionRow");
                    row_text.render(renderer, app, vertices, None);
                    renderer.pop_parent();
                }
            }
        }

        renderer.set_composite_layer(CompositeLayer::MainContent);
        Renderable::render(&NOTEPAD_EDITOR_VIEWPORT, renderer, app, vertices, None);
    }
}
