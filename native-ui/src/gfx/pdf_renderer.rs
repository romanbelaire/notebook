use crate::gfx::pdfium_runtime::bind_pdfium;
use anyhow::{Context, Result};
use pdfium_render::prelude::*;
use std::collections::{HashMap, VecDeque};

const PAGE_CACHE_LIMIT: usize = 3;

#[derive(Clone, Debug)]
pub struct PdfPageInfo {
    pub width_points: f32,
    pub height_points: f32,
    pub text_content: String,
}

#[derive(Clone, Debug)]
pub struct PdfRenderedPage {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub cache_key: String,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct PageCacheKey {
    page_num: u32,
    zoom_bucket: u32,
}

pub struct PdfRenderer {
    pdf_bytes: Vec<u8>,
    pages: Vec<PdfPageInfo>,
    page_cache: HashMap<PageCacheKey, PdfRenderedPage>,
    page_cache_lru: VecDeque<PageCacheKey>,
    document_id: u64,
}

impl PdfRenderer {
    pub fn new() -> Self {
        Self {
            pdf_bytes: Vec::new(),
            pages: Vec::new(),
            page_cache: HashMap::new(),
            page_cache_lru: VecDeque::new(),
            document_id: 0,
        }
    }

    pub fn load_pdf(&mut self, bytes: Vec<u8>) -> Result<()> {
        if bytes.len() < 4 || &bytes[0..4] != b"%PDF" {
            anyhow::bail!("Invalid PDF file: missing PDF header");
        }

        let pdfium = bind_pdfium()?;
        let doc = pdfium
            .load_pdf_from_byte_vec(bytes.clone(), None)
            .context("Failed to parse PDF bytes with PDFium")?;

        self.pdf_bytes = bytes;
        self.pages.clear();
        self.page_cache.clear();
        self.page_cache_lru.clear();
        self.document_id = self.document_id.wrapping_add(1);

        for index in 0..doc.pages().len() {
            let page = doc.pages().get(index).context("Failed to access PDF page")?;
            let width_points = page.width().value;
            let height_points = page.height().value;
            let text_content = page
                .text()
                .map(|t| t.all())
                .unwrap_or_else(|e| format!("Text extraction failed: {}", e));
            self.pages.push(PdfPageInfo {
                width_points,
                height_points,
                text_content,
            });
        }

        if self.pages.is_empty() {
            anyhow::bail!("PDF has zero pages");
        }

        Ok(())
    }

    fn zoom_bucket(zoom: f32) -> u32 {
        (zoom * 1000.0).round() as u32
    }

    fn cache_key_string(&self, page_num: u32, zoom_bucket: u32) -> String {
        format!("pdf:{}:{}:{}", self.document_id, page_num, zoom_bucket)
    }

    fn touch_lru(&mut self, key: PageCacheKey) {
        self.page_cache_lru.retain(|k| *k != key);
        self.page_cache_lru.push_back(key);
        while self.page_cache_lru.len() > PAGE_CACHE_LIMIT {
            let evict = self.page_cache_lru.pop_front().expect("LRU eviction key missing");
            self.page_cache.remove(&evict);
        }
    }

    pub fn get_or_render_page(&mut self, page_num: u32, zoom: f32) -> Result<PdfRenderedPage> {
        let zoom_bucket = Self::zoom_bucket(zoom.max(0.1));
        let key = PageCacheKey {
            page_num,
            zoom_bucket,
        };
        if let Some(page) = self.page_cache.get(&key) {
            let cached = page.clone();
            self.touch_lru(key);
            return Ok(cached);
        }

        if page_num == 0 || page_num as usize > self.pages.len() {
            anyhow::bail!(
                "Invalid page request: page {} out of {}",
                page_num,
                self.pages.len()
            );
        }

        let pdfium = bind_pdfium()?;
        let doc = pdfium
            .load_pdf_from_byte_vec(self.pdf_bytes.clone(), None)
            .context("Failed to reopen PDF bytes for page render")?;
        let page = doc
            .pages()
            .get((page_num - 1) as u16)
            .context("Failed to get requested PDF page for rendering")?;

        let target_width = ((page.width().value * zoom.max(0.1)).round() as u32).max(64);
        let bitmap = page
            .render_with_config(
            &PdfRenderConfig::new()
                .set_target_width(target_width as i32)
                .render_form_data(true),
        )?
            .as_image()
            .to_rgba8();

        let rendered = PdfRenderedPage {
            page_num,
            width: bitmap.width(),
            height: bitmap.height(),
            rgba: bitmap.into_raw(),
            cache_key: self.cache_key_string(page_num, zoom_bucket),
        };

        self.page_cache.insert(key, rendered.clone());
        self.touch_lru(key);
        Ok(rendered)
    }

    pub fn get_page_info(&self, page_num: u32) -> &PdfPageInfo {
        &self.pages[(page_num - 1) as usize]
    }

    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    pub fn preload_adjacent_pages(&mut self, current_page: u32, zoom: f32) -> Result<()> {
        let pages = [current_page.saturating_sub(1), current_page, current_page + 1];
        for page in pages {
            if page >= 1 && page as usize <= self.pages.len() {
                self.get_or_render_page(page, zoom)?;
            }
        }
        Ok(())
    }
}

