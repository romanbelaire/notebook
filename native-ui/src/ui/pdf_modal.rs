use glam::Vec2;
use crate::ui::Button;
use crate::gfx::pdf_renderer::{PdfRenderedPage, PdfRenderer};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DocumentKind {
    Pdf,
    TextLike,
    Unsupported,
}

pub struct PdfModal {
    pub is_open: bool,
    pub filename: Option<String>,
    pub current_page: u32,
    pub total_pages: Option<u32>,
    pub position: Vec2,
    pub size: Vec2,
    pub close_button: Button,
    pub prev_page_button: Button,
    pub next_page_button: Button,
    pub zoom_in_button: Button,
    pub zoom_out_button: Button,
    pub zoom_reset_button: Button,
    pub zoom_level: f32,
    pub pdf_data: Option<Vec<u8>>,
    pub loading: bool,
    pub error: Option<String>,  // Error message if PDF loading failed
    pub pdf_renderer: PdfRenderer,
    pub rendered_page: Option<PdfRenderedPage>,
    pub document_kind: DocumentKind,
    pub text_content: Option<String>,
}

impl PdfModal {
    pub fn new() -> Self {
        let modal_width = 1000.0;
        let modal_height = 800.0;
        let center_x = 960.0;
        let center_y = 540.0;
        
        Self {
            is_open: false,
            filename: None,
            current_page: 1,
            total_pages: None,
            position: Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0),
            size: Vec2::new(modal_width, modal_height),
            close_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 50.0, center_y - modal_height / 2.0 + 20.0),
                Vec2::new(30.0, 30.0),
                "×",
            ),
            prev_page_button: Button::new(
                Vec2::new(center_x - modal_width / 2.0 + 20.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(80.0, 30.0),
                "Prev",
            ),
            next_page_button: Button::new(
                Vec2::new(center_x - modal_width / 2.0 + 120.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(80.0, 30.0),
                "Next",
            ),
            zoom_in_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 300.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(40.0, 30.0),
                "+",
            ),
            zoom_out_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 350.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(40.0, 30.0),
                "-",
            ),
            zoom_reset_button: Button::new(
                Vec2::new(center_x + modal_width / 2.0 - 250.0, center_y + modal_height / 2.0 - 50.0),
                Vec2::new(70.0, 30.0),
                "100%",
            ),
            zoom_level: 1.0,
            pdf_data: None,
            loading: false,
            error: None,
            pdf_renderer: PdfRenderer::new(),
            rendered_page: None,
            document_kind: DocumentKind::Pdf,
            text_content: None,
        }
    }

    pub fn open(&mut self, filename: String, initial_page: Option<u32>) {
        self.filename = Some(filename.clone());
        self.current_page = initial_page.unwrap_or(1);
        self.is_open = true;
        self.loading = true;
        self.zoom_level = 1.0;
        self.pdf_data = None;
        self.total_pages = None;
        self.error = None;
        self.text_content = None;
        self.rendered_page = None;
        let ext = filename.split('.').next_back().unwrap_or("").to_ascii_lowercase();
        self.document_kind = if ext == "pdf" {
            DocumentKind::Pdf
        } else if ext == "txt" || ext == "md" || ext == "html" || ext == "htm" || ext == "json" {
            DocumentKind::TextLike
        } else {
            DocumentKind::Unsupported
        };
        
        // Extract page number from filename if it contains page info (e.g., "paper.pdf#page=5")
        if let Some(page_str) = filename.split("#page=").nth(1) {
            if let Ok(page) = page_str.parse::<u32>() {
                self.current_page = page;
            }
        }
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.filename = None;
        self.pdf_data = None;
        self.loading = false;
        self.pdf_renderer = PdfRenderer::new();
        self.text_content = None;
        self.rendered_page = None;
    }

    pub fn load_pdf(&mut self, bytes: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        match self.pdf_renderer.load_pdf(bytes) {
            Ok(_) => {
                self.total_pages = Some(self.pdf_renderer.num_pages() as u32);
                if self.current_page == 0 {
                    self.current_page = 1;
                }
                if self.current_page > self.total_pages.unwrap() {
                    self.current_page = self.total_pages.unwrap();
                }
                self.loading = false;
                self.error = None;
                self.render_current_page()?;
                Ok(())
            }
            Err(e) => {
                self.loading = false;
                let error_msg = format!("Failed to load PDF: {}", e);
                self.error = Some(error_msg.clone());
                Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, error_msg)) as Box<dyn std::error::Error>)
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

    pub fn render_current_page(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.document_kind != DocumentKind::Pdf {
            return Ok(());
        }
        let rendered = self
            .pdf_renderer
            .get_or_render_page(self.current_page, self.zoom_level)?;
        self.pdf_renderer
            .preload_adjacent_pages(self.current_page, self.zoom_level)?;
        self.rendered_page = Some(rendered);
        self.error = None;
        Ok(())
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        if !self.is_open {
            return false;
        }
        let rel = pos - self.position;
        rel.x >= 0.0 && rel.x <= self.size.x && rel.y >= 0.0 && rel.y <= self.size.y
    }

    pub fn update_layout(&mut self, viewport_size: Vec2) {
        let modal_width = 1000.0;
        let modal_height = 800.0;
        let center_x = viewport_size.x / 2.0;
        let center_y = viewport_size.y / 2.0;
        
        self.position = Vec2::new(center_x - modal_width / 2.0, center_y - modal_height / 2.0);
        self.size = Vec2::new(modal_width, modal_height);
        
        self.close_button.position = Vec2::new(self.position.x + modal_width - 50.0, self.position.y + 20.0);
        self.prev_page_button.position = Vec2::new(self.position.x + 20.0, self.position.y + modal_height - 50.0);
        self.next_page_button.position = Vec2::new(self.position.x + 120.0, self.position.y + modal_height - 50.0);
        self.zoom_out_button.position = Vec2::new(self.position.x + modal_width - 350.0, self.position.y + modal_height - 50.0);
        self.zoom_in_button.position = Vec2::new(self.position.x + modal_width - 300.0, self.position.y + modal_height - 50.0);
        self.zoom_reset_button.position = Vec2::new(self.position.x + modal_width - 250.0, self.position.y + modal_height - 50.0);
    }

    pub fn next_page(&mut self) {
        if let Some(total) = self.total_pages {
            if self.current_page < total {
                self.current_page += 1;
                if let Err(e) = self.render_current_page() {
                    self.set_error(format!("Failed to render page {}: {}", self.current_page, e));
                }
            }
        }
    }

    pub fn prev_page(&mut self) {
        if self.current_page > 1 {
            self.current_page -= 1;
            if let Err(e) = self.render_current_page() {
                self.set_error(format!("Failed to render page {}: {}", self.current_page, e));
            }
        }
    }

    pub fn zoom_in(&mut self) {
        self.zoom_level = (self.zoom_level + 0.1).min(3.0);
        if let Err(e) = self.render_current_page() {
            self.set_error(format!("Failed to render zoomed page: {}", e));
        }
    }

    pub fn zoom_out(&mut self) {
        self.zoom_level = (self.zoom_level - 0.1).max(0.3);
        if let Err(e) = self.render_current_page() {
            self.set_error(format!("Failed to render zoomed page: {}", e));
        }
    }

    pub fn zoom_reset(&mut self) {
        self.zoom_level = 1.0;
        if let Err(e) = self.render_current_page() {
            self.set_error(format!("Failed to render page at 100%: {}", e));
        }
    }
}

