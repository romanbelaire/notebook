# Markdown rendering (constellation and chat)

Native UI renders assistant and user message bodies as **CommonMark** via [pulldown-cmark](https://github.com/raphlinus/pulldown-cmark), with text layout and drawing through **Parley** and **Vello**. Two code paths share the same event walk where it matters for layout:

| Path | Entry | Role |
|------|--------|------|
| Constellation cards | `Renderer::build_markdown_scene` | GPU text layer (cached scenes, wrap, bold/italic) |
| Card height / hit boxes | `GraphState::measure_markdown_block` | Must match `build_markdown_scene` line breaks and width |
| Linear chat bubbles | `walk_markdown` in `gfx/components/chat/markdown.rs` | Queue-based text + measurement |

---

## Supported markdown (authoring)

Authors can rely on the following in shard messages:

### Paragraphs and breaks

- **Blank lines** separate block paragraphs. Each paragraph closes with a pulldown `End(Paragraph)` event, which flushes the current line and advances vertically.
- **Soft line breaks** inside a paragraph (two spaces at end of line, or features that emit `SoftBreak`) behave like a line break: current line is flushed, cursor moves down.
- **`**` does not insert a newline**; it toggles **strong** emphasis only.

### Ordered and unordered lists

- **Ordered lists** (`1.`, `2.`, …): markers are **not** part of the raw `Text` stream; the renderer **prepends** `"N. "` at each `Start(Item)` using the list’s start number from `List(Some(n))` and increments after each item.
- **Bullet lists** (`-`, `*`, `+`): each item is prefixed with `"• "`.
- **Nested lists**: a small stack of list frames tracks ordered vs unordered and the current number per level.
- **Loose vs tight items**: if pulldown emits a paragraph inside an item, `End(Paragraph)` ends the line; if the item is *tight* (no paragraph wrapper), `End(Item)` flushes any remaining text so the next item starts on a new line.

Example:

```markdown
1. **Header**: trailing text on the same item
2. **Second**: more text
```

### Inline emphasis

- **Bold**: `**...**` maps to Parley `FontWeight::BOLD` (not merely a larger font size).
- **Italic**: `*...*` / `_..._` maps to `FontStyle::Italic`.
- Fenced / indented **code blocks** use `Tag::CodeBlock` (styling distinct from body text). **Inline `` `code` ``** (event `Code`) is not wired in `build_markdown_scene` today; prefer fenced blocks for code in constellation if needed.

### Not implemented in the custom walker

Block types that fall through the `match` (for example headings, block quotes, links, images, tables) are **ignored** until explicitly handled. Unknown events do not crash; they simply produce no output.

---

## Implementation notes (for maintainers)

### Paragraph cache and style

- `ParagraphCacheKey` includes **brush bits** and **style bits** (bold, italic) so Parley layouts differ for real weight/style.
- `build_cached_paragraph` applies `FontWeight` / `FontStyle` before `LineHeight` and `Brush`.

### Inline cursor and vertical rhythm

- After each **inline** segment (e.g. strong boundaries), **Y** advances by `seg_h - last_line_height` so the next run stays on the same line when it is still the same wrapped row; **full** line breaks (word wrap flush, paragraph end, soft/hard break) advance by the full segment height.
- Word wrap and soft/hard breaks render at **`current_x`** with **`max_width - (current_x - start_x)`** as the remaining width, not always from `start_x`, so text after a mid-line style change does not redraw from the left margin on the same row.

### Lists and `measure_markdown_block`

- `GraphState::measure_markdown_block` mirrors list prefix injection, `End(Paragraph)`, tight `End(Item)`, and wrap/break behavior so node height and width stay aligned with `build_markdown_scene`.

## Key files

| File | Relevance |
|------|------------|
| `native-ui/src/gfx/renderer.rs` | `build_markdown_scene`, `markdown_scene_end_line`, `measure_markdown_segment_flow`, paragraph cache |
| `native-ui/src/gfx/text_layout.rs` | `ParagraphCacheKey`, `ParagraphWrappedFlow`, `build_cached_paragraph`, `vello_draw_paragraph_layout` |
| `native-ui/src/state/graph.rs` | `measure_markdown_block`, `update_node_sizes` |
| `native-ui/src/gfx/components/chat/markdown.rs` | `walk_markdown`, `measure_message_markdown`, linear chat rendering |

---

## Changelog (high level)

Recent work in this area:

- Real **bold/italic** via Parley styles; cache keys include style bits.
- **Inline Y** correction (`seg_h - last_line_height`) plus **mid-line wrap** at `current_x`.
- **Ordered and unordered lists** with synthetic markers and paragraph/item boundaries.
- Shared **full-line flush** helper for soft/hard break and paragraph end in `build_markdown_scene`.

When extending markdown support, update **all three** walkers (`build_markdown_scene`, `measure_markdown_block`, `walk_markdown`) so constellation layout, scroll regions, and chat stay consistent.
