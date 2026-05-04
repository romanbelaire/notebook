use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::gfx::pdf_renderer::{PdfRenderedPage, PdfRenderer};
use crate::persistence::{PdfAnnotationPersistence, PdfReadingPositionPersistence, ReadingPosition};
use crate::ui::core::Rect;
use crate::ui::scroll_view::ScrollView;
use crate::ui::text_input::TextInput;
use crate::ui::Button;

// ──────────────────────────────────────────────────────────────────────────────
// Layout constants shared between modal state and render code
// ──────────────────────────────────────────────────────────────────────────────
pub const HEADER_HEIGHT: f32 = 40.0;
pub const FOOTER_HEIGHT: f32 = 60.0;
pub const TOC_PANEL_WIDTH: f32 = 220.0;
pub const SEARCH_BAR_HEIGHT: f32 = 36.0;
pub const TOC_ENTRY_HEIGHT: f32 = 28.0;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DocumentKind {
    Pdf,
    TextLike,
    Unsupported,
}

/// A user-created highlight annotation stored in PDF coordinate space.
/// PDF coordinates: origin at bottom-left, y increases upward.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PdfAnnotation {
    pub page: u32,
    /// Left edge in PDF points.
    pub x0: f32,
    /// Bottom edge in PDF points (PDF y-axis: 0 at bottom).
    pub y0: f32,
    /// Right edge in PDF points.
    pub x1: f32,
    /// Top edge in PDF points.
    pub y1: f32,
    /// RGBA highlight colour [r, g, b, a].
    pub color: [f32; 4],
    pub note: String,
}

// ──────────────────────────────────────────────────────────────────────────────
// Main struct
// ──────────────────────────────────────────────────────────────────────────────

pub struct PdfModal {
    pub is_open: bool,
    pub filename: Option<String>,
    pub current_page: u32,
    pub total_pages: Option<u32>,
    pub position: Vec2,
    pub size: Vec2,

    // ── Header buttons ──────────────────────────────────────────────────────
    pub close_button: Button,
    pub toc_button: Button,
    pub annotate_button: Button,

    // ── Footer navigation ────────────────────────────────────────────────────
    pub prev_page_button: Button,
    pub next_page_button: Button,
    pub zoom_in_button: Button,
    pub zoom_out_button: Button,
    pub zoom_reset_button: Button,

    // ── Zoom & scroll ────────────────────────────────────────────────────────
    /// Multiplier applied to the content-area width to get the render target width.
    /// 1.0 → fill content width; >1.0 → oversized, scroll to pan.
    pub zoom_level: f32,
    pub scroll_x: f32,
    pub scroll_y: f32,

    // ── Go-to-page ───────────────────────────────────────────────────────────
    pub goto_active: bool,
    pub goto_input: TextInput,

    // ── Table of contents ────────────────────────────────────────────────────
    pub toc_open: bool,
    pub toc_scroll: ScrollView,

    // ── Full-text search ─────────────────────────────────────────────────────
    pub search_active: bool,
    pub search_input: TextInput,
    pub search_matches: Vec<u32>,
    pub search_current: usize,

    // ── Annotations ──────────────────────────────────────────────────────────
    pub annotation_mode: bool,
    pub annotations: Vec<PdfAnnotation>,
    /// Screen position where the current drag started.
    pub annotation_drag_start: Option<Vec2>,
    /// Current drag end position (screen space).
    pub annotation_drag_current: Vec2,
    /// True while the note-entry input is showing after a drag completes.
    pub note_active: bool,
    pub note_input: TextInput,
    /// PDF-coordinate rect for the annotation being drafted.
    pub pending_annotation_rect: Option<[f32; 4]>,

    // ── Internal ─────────────────────────────────────────────────────────────
    pub loading: bool,
    pub error: Option<String>,
    pub pdf_data: Option<Vec<u8>>,
    pub pdf_renderer: PdfRenderer,
    pub rendered_page: Option<PdfRenderedPage>,
    pub document_kind: DocumentKind,
    pub text_content: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// impl
// ──────────────────────────────────────────────────────────────────────────────

impl PdfModal {
    pub fn new() -> Self {
        let modal_width = 1000.0_f32;
        let modal_height = 800.0_f32;
        let center_x = 960.0_f32;
        let center_y = 540.0_f32;

        let pos = Vec2::new(center_x - modal_width * 0.5, center_y - modal_height * 0.5);
        let right = pos.x + modal_width;
        let bottom = pos.y + modal_height;

        let toc_area_pos = Vec2::new(pos.x, pos.y + HEADER_HEIGHT);
        let toc_area_h = modal_height - HEADER_HEIGHT - FOOTER_HEIGHT;

        Self {
            is_open: false,
            filename: None,
            current_page: 1,
            total_pages: None,
            position: pos,
            size: Vec2::new(modal_width, modal_height),

            close_button: Button::new(
                Vec2::new(right - 50.0, pos.y + 5.0),
                Vec2::new(30.0, 30.0),
                "×",
            ),
            toc_button: Button::new(
                Vec2::new(pos.x + 8.0, pos.y + 5.0),
                Vec2::new(30.0, 30.0),
                "≡",
            ),
            annotate_button: Button::new(
                Vec2::new(pos.x + 44.0, pos.y + 5.0),
                Vec2::new(30.0, 30.0),
                "✏",
            ),

            prev_page_button: Button::new(
                Vec2::new(pos.x + 20.0, bottom - 50.0),
                Vec2::new(80.0, 30.0),
                "◀ Prev",
            ),
            next_page_button: Button::new(
                Vec2::new(pos.x + 110.0, bottom - 50.0),
                Vec2::new(80.0, 30.0),
                "Next ▶",
            ),
            zoom_out_button: Button::new(
                Vec2::new(right - 350.0, bottom - 50.0),
                Vec2::new(40.0, 30.0),
                "−",
            ),
            zoom_in_button: Button::new(
                Vec2::new(right - 300.0, bottom - 50.0),
                Vec2::new(40.0, 30.0),
                "+",
            ),
            zoom_reset_button: Button::new(
                Vec2::new(right - 250.0, bottom - 50.0),
                Vec2::new(70.0, 30.0),
                "100%",
            ),

            zoom_level: 1.0,
            scroll_x: 0.0,
            scroll_y: 0.0,

            goto_active: false,
            goto_input: {
                let mut t = TextInput::new(Vec2::ZERO, Vec2::new(50.0, 24.0));
                t.set_placeholder("1".to_string());
                t
            },

            toc_open: false,
            toc_scroll: ScrollView::new(toc_area_pos, Vec2::new(TOC_PANEL_WIDTH, toc_area_h)),

            search_active: false,
            search_input: {
                let mut t = TextInput::new(Vec2::ZERO, Vec2::new(300.0, 28.0));
                t.set_placeholder("Search…".to_string());
                t
            },
            search_matches: Vec::new(),
            search_current: 0,

            annotation_mode: false,
            annotations: Vec::new(),
            annotation_drag_start: None,
            annotation_drag_current: Vec2::ZERO,
            note_active: false,
            note_input: {
                let mut t = TextInput::new(Vec2::ZERO, Vec2::new(400.0, 24.0));
                t.set_placeholder("Add note… (Enter to save, Esc to cancel)".to_string());
                t
            },
            pending_annotation_rect: None,

            loading: false,
            error: None,
            pdf_data: None,
            pdf_renderer: PdfRenderer::new(),
            rendered_page: None,
            document_kind: DocumentKind::Pdf,
            text_content: None,
        }
    }

    // ── Geometry helpers ─────────────────────────────────────────────────────

    pub fn content_area(&self) -> Rect {
        let toc_offset = if self.toc_open { TOC_PANEL_WIDTH } else { 0.0 };
        let search_offset = if self.search_active { SEARCH_BAR_HEIGHT } else { 0.0 };
        Rect::new(
            self.position.x + toc_offset,
            self.position.y + HEADER_HEIGHT + search_offset,
            self.size.x - toc_offset,
            self.size.y - HEADER_HEIGHT - FOOTER_HEIGHT - search_offset,
        )
    }

    pub fn toc_area(&self) -> Rect {
        Rect::new(
            self.position.x,
            self.position.y + HEADER_HEIGHT,
            TOC_PANEL_WIDTH,
            self.size.y - HEADER_HEIGHT - FOOTER_HEIGHT,
        )
    }

    pub fn search_bar_rect(&self) -> Rect {
        let toc_offset = if self.toc_open { TOC_PANEL_WIDTH } else { 0.0 };
        Rect::new(
            self.position.x + toc_offset,
            self.position.y + HEADER_HEIGHT,
            self.size.x - toc_offset,
            SEARCH_BAR_HEIGHT,
        )
    }

    /// Pixel width that the current page should be rendered at.
    pub fn target_render_width(&self) -> u32 {
        let toc_offset = if self.toc_open { TOC_PANEL_WIDTH } else { 0.0 };
        let content_w = self.size.x - toc_offset;
        (content_w * self.zoom_level).round() as u32
    }

    /// (draw_x, draw_y, draw_w, draw_h) for the rendered page bitmap in screen coordinates.
    /// Accounts for scroll offsets and centering when smaller than the content area.
    pub fn page_draw_rect(&self) -> (f32, f32, f32, f32) {
        let content = self.content_area();
        if let Some(ref r) = self.rendered_page {
            let draw_w = r.width as f32;
            let draw_h = r.height as f32;
            let draw_x = content.x + ((content.width - draw_w).max(0.0) * 0.5) - self.scroll_x;
            let draw_y = content.y - self.scroll_y;
            (draw_x, draw_y, draw_w, draw_h)
        } else {
            (content.x, content.y, 0.0, 0.0)
        }
    }

    /// Convert a screen-space point to PDF-coordinate space (points, y-up).
    /// Returns `None` if the modal has no rendered page or if the point is outside the page.
    pub fn screen_to_pdf(&self, screen: Vec2) -> Option<(f32, f32)> {
        let (dx, dy, dw, dh) = self.page_draw_rect();
        if dw == 0.0 || dh == 0.0 {
            return None;
        }
        let info = self.pdf_renderer.get_page_info(self.current_page);
        let rel_x = screen.x - dx;
        let rel_y = screen.y - dy;
        // PDF y-axis is inverted: 0 at bottom.
        let pdf_x = rel_x / dw * info.width_points;
        let pdf_y = (1.0 - rel_y / dh) * info.height_points;
        Some((pdf_x, pdf_y))
    }

    /// Convert a PDF-coordinate point to screen space.
    pub fn pdf_to_screen(&self, pdf_x: f32, pdf_y: f32) -> Vec2 {
        let (dx, dy, dw, dh) = self.page_draw_rect();
        let info = self.pdf_renderer.get_page_info(self.current_page);
        let sx = dx + pdf_x / info.width_points * dw;
        let sy = dy + (1.0 - pdf_y / info.height_points) * dh;
        Vec2::new(sx, sy)
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    pub fn open(&mut self, filename: String, initial_page: Option<u32>) {
        self.filename = Some(filename.clone());
        self.is_open = true;
        self.loading = true;
        self.zoom_level = 1.0;
        self.scroll_x = 0.0;
        self.scroll_y = 0.0;
        self.pdf_data = None;
        self.total_pages = None;
        self.error = None;
        self.text_content = None;
        self.rendered_page = None;
        self.search_active = false;
        self.search_matches.clear();
        self.search_input.text.clear();
        self.goto_active = false;
        self.goto_input.text.clear();
        self.annotation_drag_start = None;
        self.note_active = false;
        self.note_input.text.clear();
        self.pending_annotation_rect = None;

        let clean_filename = filename.split('#').next().unwrap_or(&filename).to_string();
        let ext = clean_filename
            .split('.')
            .next_back()
            .unwrap_or("")
            .to_ascii_lowercase();
        self.document_kind = if ext == "pdf" {
            DocumentKind::Pdf
        } else if matches!(ext.as_str(), "txt" | "md" | "html" | "htm" | "json") {
            DocumentKind::TextLike
        } else {
            DocumentKind::Unsupported
        };

        // Extract page number from "#page=N" fragment.
        let frag_page = filename
            .split("#page=")
            .nth(1)
            .and_then(|s| s.parse::<u32>().ok());

        // Restore reading position when no explicit page is requested.
        let (restored_page, restored_scroll) = if initial_page.is_none() && frag_page.is_none() {
            if let Some(pos) = PdfReadingPositionPersistence::load(&clean_filename) {
                (Some(pos.page), pos.scroll_y)
            } else {
                (None, 0.0)
            }
        } else {
            (None, 0.0)
        };

        self.current_page = frag_page
            .or(initial_page)
            .or(restored_page)
            .unwrap_or(1)
            .max(1);
        self.scroll_y = restored_scroll;

        // Load annotations for this document.
        self.annotations = PdfAnnotationPersistence::load(&clean_filename);
    }

    pub fn close(&mut self) {
        // Persist reading position.
        if let Some(ref filename) = self.filename {
            let clean = filename.split('#').next().unwrap_or(filename).to_string();
            PdfReadingPositionPersistence::save(
                &clean,
                ReadingPosition {
                    page: self.current_page,
                    scroll_y: self.scroll_y,
                },
            );
            PdfAnnotationPersistence::save(&clean, &self.annotations);
        }

        self.is_open = false;
        self.filename = None;
        self.pdf_data = None;
        self.loading = false;
        self.pdf_renderer = PdfRenderer::new();
        self.text_content = None;
        self.rendered_page = None;
        self.search_active = false;
        self.search_matches.clear();
        self.goto_active = false;
        self.note_active = false;
        self.pending_annotation_rect = None;
        self.annotation_drag_start = None;
        self.annotations.clear();
    }

    pub fn load_pdf(&mut self, bytes: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        match self.pdf_renderer.load_pdf(bytes) {
            Ok(_) => {
                self.total_pages = Some(self.pdf_renderer.num_pages() as u32);
                self.current_page = self.current_page.max(1);
                if self.current_page > self.total_pages.unwrap() {
                    self.current_page = self.total_pages.unwrap();
                }
                // Update TOC scroll height now that we know entry count.
                let toc_h = self.pdf_renderer.toc.len() as f32 * TOC_ENTRY_HEIGHT;
                self.toc_scroll.set_content_height(toc_h);

                self.loading = false;
                self.error = None;
                self.render_current_page()?;
                Ok(())
            }
            Err(e) => {
                self.loading = false;
                let msg = format!("Failed to load PDF: {}", e);
                self.error = Some(msg.clone());
                Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, msg)))
            }
        }
    }

    pub fn set_error(&mut self, error: String) {
        self.error = Some(error);
        self.loading = false;
    }

    pub fn load_text_content(&mut self, text: String) {
        self.text_content = Some(text);
        self.loading = false;
        self.error = None;
    }

    // ── Page render ──────────────────────────────────────────────────────────

    pub fn render_current_page(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.document_kind != DocumentKind::Pdf {
            return Ok(());
        }
        let tw = self.target_render_width();
        let rendered = self.pdf_renderer.get_or_render_page(self.current_page, tw)?;
        self.pdf_renderer.preload_adjacent_pages(self.current_page, tw)?;
        self.rendered_page = Some(rendered);
        self.error = None;
        Ok(())
    }

    // ── Navigation ───────────────────────────────────────────────────────────

    pub fn next_page(&mut self) {
        if let Some(total) = self.total_pages {
            if self.current_page < total {
                self.current_page += 1;
                self.scroll_x = 0.0;
                self.scroll_y = 0.0;
                if let Err(e) = self.render_current_page() {
                    self.set_error(format!("Failed to render page {}: {}", self.current_page, e));
                }
            }
        }
    }

    pub fn prev_page(&mut self) {
        if self.current_page > 1 {
            self.current_page -= 1;
            self.scroll_x = 0.0;
            self.scroll_y = 0.0;
            if let Err(e) = self.render_current_page() {
                self.set_error(format!("Failed to render page {}: {}", self.current_page, e));
            }
        }
    }

    pub fn goto_page(&mut self, page: u32) {
        let total = self.total_pages.unwrap_or(1);
        self.current_page = page.max(1).min(total);
        self.scroll_x = 0.0;
        self.scroll_y = 0.0;
        if let Err(e) = self.render_current_page() {
            self.set_error(format!("Failed to render page {}: {}", self.current_page, e));
        }
    }

    // ── Zoom ─────────────────────────────────────────────────────────────────

    pub fn zoom_in(&mut self) {
        self.zoom_level = (self.zoom_level + 0.25).min(5.0);
        self.clamp_scroll_after_zoom();
        if let Err(e) = self.render_current_page() {
            self.set_error(format!("Failed to render zoomed page: {}", e));
        }
    }

    pub fn zoom_out(&mut self) {
        self.zoom_level = (self.zoom_level - 0.25).max(0.25);
        self.clamp_scroll_after_zoom();
        if let Err(e) = self.render_current_page() {
            self.set_error(format!("Failed to render zoomed page: {}", e));
        }
    }

    pub fn zoom_reset(&mut self) {
        self.zoom_level = 1.0;
        self.scroll_x = 0.0;
        self.scroll_y = 0.0;
        if let Err(e) = self.render_current_page() {
            self.set_error(format!("Failed to render page at 100%: {}", e));
        }
    }

    fn clamp_scroll_after_zoom(&mut self) {
        let content = self.content_area();
        if let Some(ref r) = self.rendered_page {
            self.scroll_x = self
                .scroll_x
                .clamp(0.0, (r.width as f32 - content.width).max(0.0));
            self.scroll_y = self
                .scroll_y
                .clamp(0.0, (r.height as f32 - content.height).max(0.0));
        }
    }

    /// Clamp scroll offsets to valid range for the current rendered page and content area.
    pub fn clamp_scroll(&mut self) {
        let content = self.content_area();
        if let Some(ref r) = self.rendered_page {
            self.scroll_x = self
                .scroll_x
                .clamp(0.0, (r.width as f32 - content.width).max(0.0));
            self.scroll_y = self
                .scroll_y
                .clamp(0.0, (r.height as f32 - content.height).max(0.0));
        }
    }

    // ── Search ───────────────────────────────────────────────────────────────

    pub fn search_run(&mut self) {
        if self.search_input.text.is_empty() {
            self.search_matches.clear();
            return;
        }
        self.search_matches = self.pdf_renderer.search_text(&self.search_input.text);
        self.search_current = 0;
        if let Some(&page) = self.search_matches.first() {
            self.goto_page(page);
        }
    }

    pub fn search_next(&mut self) {
        if self.search_matches.is_empty() {
            return;
        }
        self.search_current = (self.search_current + 1) % self.search_matches.len();
        let page = self.search_matches[self.search_current];
        self.goto_page(page);
    }

    pub fn search_prev(&mut self) {
        if self.search_matches.is_empty() {
            return;
        }
        if self.search_current == 0 {
            self.search_current = self.search_matches.len() - 1;
        } else {
            self.search_current -= 1;
        }
        let page = self.search_matches[self.search_current];
        self.goto_page(page);
    }

    pub fn close_search(&mut self) {
        self.search_active = false;
        self.search_matches.clear();
        self.search_input.text.clear();
    }

    // ── Annotations ──────────────────────────────────────────────────────────

    /// Begin an annotation drag from a screen-space position.
    pub fn begin_annotation_drag(&mut self, screen_pos: Vec2) {
        self.annotation_drag_start = Some(screen_pos);
        self.annotation_drag_current = screen_pos;
    }

    /// Update the drag end position.
    pub fn update_annotation_drag(&mut self, screen_pos: Vec2) {
        self.annotation_drag_current = screen_pos;
    }

    /// Finalise the drag: compute the PDF rect and enter note-input mode.
    /// Returns `false` if the drag was too small to create an annotation.
    pub fn finish_annotation_drag(&mut self) -> bool {
        let start = match self.annotation_drag_start {
            Some(s) => s,
            None => return false,
        };
        let end = self.annotation_drag_current;
        if (end - start).length() < 4.0 {
            self.annotation_drag_start = None;
            return false;
        }

        let (sx0, sy0) = match self.screen_to_pdf(start) {
            Some(p) => p,
            None => {
                self.annotation_drag_start = None;
                return false;
            }
        };
        let (sx1, sy1) = match self.screen_to_pdf(end) {
            Some(p) => p,
            None => {
                self.annotation_drag_start = None;
                return false;
            }
        };

        let x0 = sx0.min(sx1);
        let x1 = sx0.max(sx1);
        let y0 = sy0.min(sy1);
        let y1 = sy0.max(sy1);

        self.pending_annotation_rect = Some([x0, y0, x1, y1]);
        self.annotation_drag_start = None;
        self.note_active = true;
        self.note_input.text.clear();
        true
    }

    /// Commit the pending annotation with the current note text.
    pub fn commit_annotation(&mut self) {
        if let Some([x0, y0, x1, y1]) = self.pending_annotation_rect {
            self.annotations.push(PdfAnnotation {
                page: self.current_page,
                x0,
                y0,
                x1,
                y1,
                color: [1.0, 0.95, 0.25, 0.40],
                note: self.note_input.text.clone(),
            });
            if let Some(ref filename) = self.filename {
                let clean = filename.split('#').next().unwrap_or(filename).to_string();
                PdfAnnotationPersistence::save(&clean, &self.annotations);
            }
        }
        self.pending_annotation_rect = None;
        self.note_active = false;
        self.note_input.text.clear();
    }

    pub fn cancel_annotation(&mut self) {
        self.pending_annotation_rect = None;
        self.note_active = false;
        self.note_input.text.clear();
        self.annotation_drag_start = None;
    }

    pub fn delete_annotation(&mut self, index: usize) {
        if index < self.annotations.len() {
            self.annotations.remove(index);
            if let Some(ref filename) = self.filename {
                let clean = filename.split('#').next().unwrap_or(filename).to_string();
                PdfAnnotationPersistence::save(&clean, &self.annotations);
            }
        }
    }

    // ── Hit testing ──────────────────────────────────────────────────────────

    pub fn contains(&self, pos: Vec2) -> bool {
        if !self.is_open {
            return false;
        }
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }

    pub fn content_area_contains(&self, pos: Vec2) -> bool {
        self.content_area().contains_point(pos)
    }

    // ── Layout ───────────────────────────────────────────────────────────────

    pub fn update_layout(&mut self, viewport_size: Vec2) {
        let modal_w = 1000.0_f32;
        let modal_h = 800.0_f32;
        let cx = viewport_size.x * 0.5;
        let cy = viewport_size.y * 0.5;

        self.position = Vec2::new(cx - modal_w * 0.5, cy - modal_h * 0.5);
        self.size = Vec2::new(modal_w, modal_h);

        let p = self.position;
        let right = p.x + modal_w;
        let bottom = p.y + modal_h;

        self.close_button.position = Vec2::new(right - 50.0, p.y + 5.0);
        self.toc_button.position = Vec2::new(p.x + 8.0, p.y + 5.0);
        self.annotate_button.position = Vec2::new(p.x + 44.0, p.y + 5.0);

        self.prev_page_button.position = Vec2::new(p.x + 20.0, bottom - 50.0);
        self.next_page_button.position = Vec2::new(p.x + 110.0, bottom - 50.0);
        self.zoom_out_button.position = Vec2::new(right - 350.0, bottom - 50.0);
        self.zoom_in_button.position = Vec2::new(right - 300.0, bottom - 50.0);
        self.zoom_reset_button.position = Vec2::new(right - 250.0, bottom - 50.0);

        let toc_area = self.toc_area();
        self.toc_scroll.position = toc_area.position();
        self.toc_scroll.size = toc_area.size();
        let toc_h = self.pdf_renderer.toc.len() as f32 * TOC_ENTRY_HEIGHT;
        self.toc_scroll.set_content_height(toc_h);

        let search_bar = self.search_bar_rect();
        self.search_input.position = Vec2::new(search_bar.x + 8.0, search_bar.y + 4.0);
        self.search_input.size = Vec2::new(280.0, 28.0);

        // Goto input sits in the footer.
        let footer_y = bottom - 50.0;
        self.goto_input.position = Vec2::new(p.x + 200.0, footer_y + 3.0);
        self.goto_input.size = Vec2::new(48.0, 24.0);

        // Note input spans bottom of content area.
        let content = self.content_area();
        self.note_input.position = Vec2::new(content.x + 8.0, content.bottom() - 34.0);
        self.note_input.size = Vec2::new(content.width - 16.0, 28.0);
    }
}
