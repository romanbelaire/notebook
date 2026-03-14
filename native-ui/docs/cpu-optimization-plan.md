# CPU optimization plan (larger refactors)

CPU is the main bottleneck. This plan focuses on **skipping or reducing per-frame CPU work** (layout, measure, render walk, parse, physics). GPU-side partial redraw is already done; these items require structural changes.

---

## 1. Dirty-region CPU culling (high impact)

**Current:** Every frame we call `app.root.update_layout(viewport)` then `app.root.render()`, so we run layout and render for the **entire** tree (all tabs, sidebar, header, chat, library, etc.) even when only a small region is animating.

**Goal:** When we have a non-full-screen dirty rect (from `get_dirty_rects()`), only run **layout** and **render** for components that intersect that rect.

**Approach:**
- Add a **dirty rect** (or `Option<Rect>`) parameter to `update_layout(rect, dirty_rect)` and `render(renderer, app, vertices, dirty_rect)`.
- Each component implements:
  - **Layout:** If `bounds()` (or the rect passed down) does not intersect `dirty_rect`, skip `update_layout` for self and children (or do a no-op / minimal pass).
  - **Render:** If `bounds()` does not intersect `dirty_rect`, skip emitting vertices and skip rendering children.
- Root (or renderer) passes the current dirty rect. When `dirty_rect` is `None`, treat as full screen (current behavior).
- **Caveat:** Bounds must be conservative (include all children). For containers (e.g. Chat, Sidebar), bounds = union of children or the container’s visible rect. First pass can use the **window** rect (chat window, sidebar, header) so we skip whole windows when they’re outside the dirty region.

**Files:** `ui/components/root.rs`, `Renderable` trait and all implementors, `gfx/renderer.rs` (pass dirty rect into render path), `app.rs` (compute dirty rect once and pass through).

**Dependency:** Relies on `get_dirty_rects()` already implemented; reuse that for the same rect we use for GPU partial redraw.

---

## 2. Invalidate-based layout (medium–high impact)

**Current:** `update_layout` is called for the root (and thus whole tree) every frame. Many branches don’t depend on time—only on viewport, sidebar width, tab, or focus.

**Goal:** Only run `update_layout` for a branch when something that affects it has changed (viewport, sidebar open/width, active tab, focused input, etc.).

**Approach:**
- Introduce a **layout generation id** or **dirty flags** at the app (or window) level: e.g. `viewport_dirty`, `sidebar_dirty`, `active_tab_dirty`.
- Root (or a layout driver) only calls `update_layout` on a child when the child’s “layout inputs” have changed. For example: Chat’s layout depends on viewport, sidebar width, and whether it’s the active tab; Sidebar on viewport and open state; Header on viewport.
- Alternatively: **Single global layout generation counter** incremented when viewport, sidebar, or tab changes. Each component (or window) caches “last layout generation I was updated for”; if current generation equals cached, skip `update_layout` for that subtree.
- Reduces work to **only changed branches** instead of the full tree every frame.

**Files:** `app.rs` (layout invalidation triggers and generation counter or flags), `ui/components/root.rs` (decide which children need layout), optionally each window component (cache last layout generation).

---

## 3. Constellation: invalidation for `update_node_sizes` (medium impact)

**Current:** We call `update_node_sizes` every frame for **visible** nodes only (viewport culling already in place). So we still run measure (word-wrap) for every in-view node every frame.

**Goal:** Only run `update_node_sizes` when something that affects node sizes has changed: graph content, viewport size, scale, or editing override.

**Approach:**
- Track **graph content version** (e.g. hash of node ids + content lengths or a simple counter bumped when graph is set or a node is added/edited). Track **viewport size** and **editing override** (node id + size).
- In the render path, if `(content_version, viewport_size, editing_override)` equals the last one we used for `update_node_sizes`, **skip** calling `update_node_sizes` and use existing `node.size` / text heights. When the user edits or viewport resizes or graph loads, invalidate and run `update_node_sizes` again.
- Optional: run `update_node_sizes` at a **lower rate** (e.g. every 2nd or 3rd frame) when only camera/scale is changing and content is static; accept a frame or two of stale sizes for non-editing nodes.

**Files:** `state/graph.rs` (optional content version or hash), `gfx/renderer.rs` (conditionally call `update_node_sizes`), `app.rs` (if we need to expose invalidation).

---

## 4. Markdown / text output cache (medium impact)

**Current:** For every in-view constellation node we call `render_markdown_text` (parse markdown, queue text segments) every frame. Later, every queued text segment is laid out with Parley in `build_text_scenes`. Same for linear chat bubbles.

**Goal:** Cache the **result** of “parse + queue” (or the final text/icon draw commands) per (node_id, role, content_hash, scale_bucket). When content and scale haven’t changed, reuse the cached draw list and only apply the current screen position. Invalidate on content edit or scale change.

**Approach:**
- Cache key: `(node_id, "user"|"assistant", content_hash, scale_rounded)`.
- Cached value: list of `TextDrawCommand`-like structs (text, position relative to bubble origin, size, color) or a pre-built Parley layout / vello scene for that bubble.
- On render: if cache hit, transform positions by current bubble screen position and queue (or blit); if miss, run `render_markdown_text` as now and fill cache.
- Eviction: when graph changes (node removed, content edited), clear or invalidate the relevant cache entries. Limit cache size (e.g. by number of nodes or total memory).

**Files:** `gfx/components/chat.rs` (cache next to or inside render_constellation; or a small struct on ChatWindow), `gfx/renderer.rs` (if we need to accept pre-built command lists or scenes).

---

## 5. Physics: spatial hash for repulsion (medium impact, optional)

**Current:** `step_physics` is O(N²) (repulsion between every pair of nodes). We already throttle so it doesn’t run when settled; with many nodes it can still spike when physics is active.

**Goal:** Reduce repulsion work to O(N) or O(N log N) by only considering **nearby** nodes.

**Approach:**
- **Spatial hash / grid:** Partition world space into cells (e.g. cell size = 2× max node radius). For each node, add it to the cell(s) it overlaps. For repulsion, only sum forces from nodes in the same cell and neighboring cells (e.g. 3×3 or 5×5). Constant number of neighbors per node on average → O(N) with a good constant.
- **Files:** `state/graph.rs` (`step_physics` and a helper to build the grid; keep tether logic as-is).

---

## 6. Lighter-weight “idle” update path (low–medium impact)

**Current:** When `needs_continuous_redraw()` is false we don’t request a redraw, so we don’t run `update()` or `render()` at all. When we *do* redraw (e.g. user moved mouse), we run the **full** update (all windows, all lists, cursor blink, etc.) and full render.

**Goal:** When we’re about to redraw after being idle, avoid doing work that couldn’t have changed (e.g. don’t recompute layout for windows that weren’t the target of input).

**Approach:**
- Track “last input target” (e.g. which window or region received the last input). On the **first** frame after idle, run a **minimal update**: e.g. only update hover state, focus, and the component that received the event; skip or defer full layout for other windows.
- Alternatively: **Incremental update** — only call `update(dt)` on the active tab’s window and the sidebar/header, not on hidden tabs (library, settings, notepad) until they’re switched to. Requires knowing “which windows are visible” and is a bit invasive.

**Files:** `app.rs` (`update()`), possibly `main.rs` (if we need to pass “reason for redraw” into update).

---

## 7. Text measurement cache sharing and invalidation (lower impact)

**Current:** `measure_text` is cached per (text, size) in the renderer. We already avoid re-measuring off-screen nodes via viewport culling. Same text in different nodes can still cause repeated cache lookups (cache hit) or one measure per node (cache miss for unique strings).

**Goal:** Reduce redundant measurement when the same content appears in multiple places (e.g. same message in linear and graph view, or same placeholder text). Optional: **invalidate** measurement cache when font or DPI changes (currently we don’t; if we never change font, this is unnecessary).

**Approach:**
- No major refactor; ensure measurement cache is keyed by (content, font_size) and has a reasonable size limit. Optionally add a “content hash” key so long identical strings share one entry.
- If we add markdown cache (item 4), measurement for cached bubbles is only needed on cache miss, so this becomes less critical.

---

## Implementation order (CPU-focused)

| Order | Item                         | Rationale |
|-------|------------------------------|------------|
| 1     | Dirty-region CPU culling (1) | Biggest win: skip layout + render for everything outside the dirty rect. |
| 2     | Invalidate-based layout (2)  | Stops full-tree layout every frame when only one window is active. |
| 3     | Constellation invalidation (3) | Cuts per-frame measure for in-view nodes when nothing changed. |
| 4     | Markdown / text cache (4)    | Removes repeated parse + Parley for static node content. |
| 5     | Physics spatial hash (5)     | Only if profiling shows physics still hot with many nodes. |
| 6     | Idle update path (6)         | Nice-to-have; smaller gain than 1–4. |
| 7     | Measurement cache (7)        | Optional polish; do after 4 if text layout still shows up in profiles. |

---

## Out of scope / not recommended

- **Changing to “only redraw on input”** without continuous redraw: we already stop redraws when idle; re-enabling redraw only on input is done. No need to go further (e.g. 30 Hz cap) unless profiling shows frame time still too high.
- **Moving layout off the main thread:** would require making the render tree (or a copy) thread-safe and sending only “dirty” data to the render thread; large refactor for uncertain gain.
- **Replacing Parley/vello with a simpler text path:** would reduce quality or features; better to cache the output (item 4) than to replace the stack.
