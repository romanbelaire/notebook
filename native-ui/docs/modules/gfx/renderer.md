# Renderer

The `Renderer` struct is the main graphics rendering coordinator. It manages wgpu resources, vello integration, and the rendering pipeline.

## Markdown scenes (constellation)

Message bodies in constellation shards are turned into Vello scenes in **`build_markdown_scene`** (CommonMark via pulldown-cmark, Parley layout, bold/italic, lists, paragraph breaks). Card sizing uses **`GraphState::measure_markdown_block`**, which must stay aligned with that walk.

Author-facing syntax and maintainer notes: **[Markdown rendering](../../guides/markdown-rendering.md)**.

## Overview

The `Renderer` handles:
- wgpu device, surface, and pipeline management
- Vello renderer for 2D graphics and text
- Render batching and scissor rects
- Text and icon queueing
- Frame rendering

## Initialization

```rust
let renderer = pollster::block_on(Renderer::new(window.clone()));
```

### Initialization Steps

1. Create wgpu instance and surface
2. Request adapter and device
3. Configure surface format
4. Create render and blit pipelines
5. Initialize vello renderer
6. Create font and layout contexts
7. Set up texture pools

## Main Methods

### render()

Renders a frame:

```rust
pub fn render(&mut self, app: &mut App) -> Result<(), wgpu::SurfaceError>
```

Process:
1. Clear component validation state
2. Acquire surface texture
3. Create render pass
4. Render component tree
5. Render vello scenes (text/icons)
6. Blit vello output to surface
7. Present frame

### queue_text()

Queue text for rendering:

```rust
pub fn queue_text(
    &mut self,
    text: &str,
    position: Vec2,
    color: Vec4,
    font_size: f32,
)
```

Text is:
- Queued during component rendering
- Laid out via Parley
- Rendered to vello scene
- Blitted to surface

### queue_icon()

Queue icon for rendering:

```rust
pub fn queue_icon(
    &mut self,
    icon_name: &str,
    position: Vec2,
    size: f32,
    color: Vec4,
)
```

Icons are:
- Loaded from SVG files
- Cached in IconCache
- Rendered to vello scene
- Blitted to surface

### Scissor Rects

Manage clipping:

```rust
pub fn push_scissor(&mut self, rect: ScissorRect)
pub fn pop_scissor(&mut self)
```

Scissor rects:
- Clip rendering to component bounds
- Are automatically intersected for nested components
- Use UI coordinates directly

### Composite layers (`set_composite_layer`)

`set_composite_layer` sets where subsequent **quads** (`add_vertices` / `add_quad`) and **queued** text/icons are tagged. The framebuffer draws layers in a fixed order (`Background` → `MainContent` → `ConstellationText` → `SidebarChrome` → `ComposerChrome` → `HudChrome` → `Modal`). Anything that must appear **above** constellation text (e.g. context menus, sidebar row labels and section headers) should queue its `Text` while the layer is `HudChrome`; sidebar **geometry** usually stays on `SidebarChrome` and toggles layer only around `Text::render`. Full rationale and the sidebar pattern: [GPU compositing layers](../../architecture/rendering.md#gpu-compositing-layers).

### Component Validation

Validate component hierarchy:

```rust
pub fn validate_component(
    &mut self,
    component_id: &str,
    parent_id: Option<&str>,
    component_type: &str,
)
```

Prevents:
- Orphaned components
- Duplicate rendering
- Invalid parent-child relationships

## Render Batches

Vertices are organized into batches:

```rust
struct RenderBatch {
    vertices: Vec<Vertex>,
    scissor: Option<ScissorRect>,
}
```

Batches are:
- Created when scissor rect changes
- Rendered in order
- Clipped by their scissor rect

## Vello Integration

Vello is used for:
- Text rendering (via Parley)
- Icon rendering (SVG paths)
- Complex 2D graphics

### Vello Rendering

1. Create vello scene
2. Queue operations (fills, strokes, text, icons)
3. Render to offscreen texture
4. Blit to surface

## Performance Optimizations

- **Batching**: Vertices batched by scissor rect
- **Caching**: Text measurements and glyph positions cached
- **Texture Pools**: Text and icon textures pooled
- **Culling**: Off-screen components skipped

## Related Documentation

- [Rendering Pipeline](../architecture/rendering.md)
- [Renderable Trait](renderable.md)
- [Types](types.md)

