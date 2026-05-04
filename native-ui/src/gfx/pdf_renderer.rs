use crate::gfx::pdfium_runtime::bind_pdfium;
use anyhow::{Context, Result};
use pdfium_render::prelude::*;
use std::collections::{HashMap, VecDeque};

const PAGE_CACHE_LIMIT: usize = 5;

#[derive(Clone, Debug)]
pub struct PdfPageInfo {
    pub width_points: f32,
    pub height_points: f32,
    pub text_content: String,
}

/// A single entry in the PDF table of contents (bookmarks / outline).
#[derive(Clone, Debug)]
pub struct TocEntry {
    pub title: String,
    /// 1-indexed page number.
    pub page: u32,
    /// Nesting depth (0 = top-level).
    pub level: u8,
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
    /// Pixel width bucketed to the nearest 8 px.
    width_bucket: u32,
}

pub struct PdfRenderer {
    pdf_bytes: Vec<u8>,
    pages: Vec<PdfPageInfo>,
    pub toc: Vec<TocEntry>,
    page_cache: HashMap<PageCacheKey, PdfRenderedPage>,
    page_cache_lru: VecDeque<PageCacheKey>,
    document_id: u64,
}

impl PdfRenderer {
    pub fn new() -> Self {
        Self {
            pdf_bytes: Vec::new(),
            pages: Vec::new(),
            toc: Vec::new(),
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
        self.toc.clear();
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

        // Extract table of contents from PDF bookmarks.
        if let Some(root) = doc.bookmarks().root() {
            collect_bookmarks(&root, 0, &mut self.toc);
        }

        Ok(())
    }

    /// Bucket pixel width to nearest 8 to avoid redundant cache entries from minor
    /// content-area fluctuations.
    fn width_bucket(target_width: u32) -> u32 {
        ((target_width + 4) / 8) * 8
    }

    fn cache_key_string(&self, page_num: u32, width_bucket: u32) -> String {
        format!("pdf:{}:{}:{}", self.document_id, page_num, width_bucket)
    }

    fn touch_lru(&mut self, key: PageCacheKey) {
        self.page_cache_lru.retain(|k| *k != key);
        self.page_cache_lru.push_back(key);
        while self.page_cache_lru.len() > PAGE_CACHE_LIMIT {
            let evict = self
                .page_cache_lru
                .pop_front()
                .expect("LRU eviction key missing");
            self.page_cache.remove(&evict);
        }
    }

    /// Render `page_num` (1-indexed) at `target_width` pixels wide, returning the bitmap.
    /// Results are cached; `target_width` is bucketed to 8 px to reduce re-renders.
    pub fn get_or_render_page(
        &mut self,
        page_num: u32,
        target_width: u32,
    ) -> Result<PdfRenderedPage> {
        let target_width = target_width.max(64);
        let wb = Self::width_bucket(target_width);
        let key = PageCacheKey {
            page_num,
            width_bucket: wb,
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

        let bitmap = page
            .render_with_config(
                &PdfRenderConfig::new()
                    .set_target_width(wb as i32)
                    .render_form_data(true),
            )?
            .as_image()
            .to_rgba8();

        let rendered = PdfRenderedPage {
            page_num,
            width: bitmap.width(),
            height: bitmap.height(),
            rgba: bitmap.into_raw(),
            cache_key: self.cache_key_string(page_num, wb),
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

    /// Case-insensitive substring search across all pages.
    /// Returns the 1-indexed page numbers that contain a match.
    pub fn search_text(&self, query: &str) -> Vec<u32> {
        let q = query.to_lowercase();
        self.pages
            .iter()
            .enumerate()
            .filter(|(_, p)| p.text_content.to_lowercase().contains(&q))
            .map(|(i, _)| i as u32 + 1)
            .collect()
    }

    pub fn preload_adjacent_pages(&mut self, current_page: u32, target_width: u32) -> Result<()> {
        let pages = [
            current_page.saturating_sub(1),
            current_page,
            current_page + 1,
        ];
        for page in pages {
            if page >= 1 && page as usize <= self.pages.len() {
                self.get_or_render_page(page, target_width)?;
            }
        }
        Ok(())
    }
}

/// Recursively walk a bookmark subtree and collect `TocEntry` values.
fn collect_bookmarks(bookmark: &PdfBookmark<'_>, level: u8, entries: &mut Vec<TocEntry>) {
    let title = bookmark.title().unwrap_or_default();
    if !title.is_empty() {
        // Try direct destination first, then fall back to the action's destination.
        let page = bookmark
            .destination()
            .and_then(|d| d.page_index().ok())
            .map(|i| i as u32 + 1)
            .unwrap_or(0);
        if page > 0 {
            entries.push(TocEntry { title, page, level });
        }
    }
    // Walk children using first_child / next_sibling iteration (no recursion limit risk for
    // typical PDFs, and avoids lifetime issues from holding a parent ref across the call).
    let mut maybe_child = bookmark.first_child();
    while let Some(child) = maybe_child {
        collect_bookmarks(&child, level + 1, entries);
        maybe_child = child.next_sibling();
    }
}
