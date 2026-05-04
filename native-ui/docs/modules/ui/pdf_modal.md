# PDF Modal

The `ui/pdf_modal.rs` and `gfx/components/modals/pdf.rs` modules implement the native PDF viewer modal.

## Overview

The native PDF modal is a fully interactive single-page PDF viewer with:

- Content-width-relative zoom
- Scroll / pan for oversized renders
- Keyboard and mouse navigation
- Go-to-page inline input
- Table of contents panel from PDF bookmarks
- Ctrl/Cmd+F full-text page search
- Drag-to-create highlight annotations with notes
- Per-document reading position persistence

## Core Modules

- `src/ui/pdf_modal.rs` - state, interaction helpers, modal geometry, coordinate mapping
- `src/gfx/components/modals/pdf.rs` - modal rendering, TOC/search/footer UI, page + annotation draws
- `src/gfx/pdf_renderer.rs` - PDFium loading, page raster cache, TOC extraction, page-text search
- `src/app.rs` - event routing for click, drag, wheel, keyboard
- `src/persistence/pdf_annotations.rs` - JSON persistence for annotations
- `src/persistence/pdf_reading_positions.rs` - JSON persistence for last page/scroll

## Controls

### Mouse

- `Wheel` over content: vertical scroll
- `Shift + Wheel`: horizontal scroll
- `Ctrl/Cmd + Wheel`: zoom in/out
- Click page counter area: activate go-to-page input
- Click TOC entry: jump to entry page
- Annotation mode enabled: click-drag on page to create highlight region

### Keyboard

- `ArrowLeft`, `ArrowUp`, `PageUp`: previous page
- `ArrowRight`, `ArrowDown`, `PageDown`: next page
- `Home`: page 1
- `End`: last page
- `Ctrl/Cmd + F`: open search
- `Enter` in search: next result (`Shift+Enter` previous)
- `Enter` in go-to-page: jump to entered page
- `Esc`: close note input, then search, then modal (in that order)

## Zoom and Scroll Model

Zoom is based on content width, not PDF point width.

At render time:

- `target_width_px = content_area_width_px * zoom_level`
- page bitmap is rasterized at `target_width_px`
- if rendered size is larger than the visible content area, `scroll_x/scroll_y` pan the view
- rendered image is clipped to content bounds

This avoids the old behavior where zoom changes were mostly hidden by letterboxing.

## TOC Data Flow

1. `PdfRenderer::load_pdf()` parses the document through PDFium.
2. Bookmark tree is traversed from `doc.bookmarks().root()`.
3. Entries are flattened into:
   - `title`
   - `page` (1-indexed)
   - `level` (indent depth)
4. UI renders entries in a left panel (`toc_open`) using `ScrollView`.

## Search Data Flow

1. PDF load extracts `text_content` per page into `PdfPageInfo`.
2. Search query scans all page text (case-insensitive substring).
3. Matching pages are stored in `search_matches`.
4. `Enter` / `Shift+Enter` cycles result index and jumps page.

Current visual match cue is page-level (matching page indicator). The pipeline is ready for future per-rect in-page highlighting.

## Annotations

Annotations are stored in PDF coordinate space so they remain stable across zoom and viewport sizes.

`PdfAnnotation` stores:

- `page`
- `x0, y0, x1, y1` in PDF points
- `color` RGBA
- `note`

When rendered:

- PDF coordinates are converted to screen coordinates using page dimensions and draw rect
- rect is clipped to the visible content area

Persistence file: `data/pdf_annotations.json` keyed by document filename.

## Reading Position Persistence

On close, modal stores per-document:

- `page`
- `scroll_y`

in `data/pdf_reading_positions.json`.

On open (without explicit `#page=` fragment), the stored position is restored.

## Notes

- Rendering remains raster-based through PDFium (`pdfium-render`), with width-bucketed page cache entries.
- Go-to, search, and annotation note inputs are integrated through the shared `TextEditor` routing in `App`.

