# Graphics Module

The `gfx/` module provides the graphics rendering system for the Notebook Native UI. It uses wgpu for low-level graphics and vello for advanced 2D rendering.

## Module Structure

```
gfx/
├── renderer.rs      # Main renderer
├── renderable.rs      # Renderable trait
├── types.rs           # Vertex, color types
├── components/        # Graphics components
├── icons.rs           # Icon system
└── pdf_renderer.rs    # PDF rendering
```

## Key Components

### Renderer

The `Renderer` struct manages all graphics operations:
- wgpu device and surface
- Vello renderer
- Text and icon rendering
- Render batching

See [Renderer Documentation](renderer.md) for details.

### Renderable Trait

The `Renderable` trait defines the interface for all renderable components:
- `render()`: Generate vertices and queue text/icons
- `bounds()`: Get component bounds
- `update_layout()`: Update layout

See [Renderable Documentation](renderable.md) for details.

### Types

Graphics types include:
- `Vertex`: Vertex data structure
- `Quad`: Quad primitive
- `ScissorRect`: Clipping rectangle

See [Types Documentation](types.md) for details.

### Shader system

The UI shader (wgpu) draws all quads in one pipeline. Vertex flags and uniforms select effects: rounded rects, borders, glows, bubble (chat border wobble), and liquid slider (nav bar stretch).

See [Shader System Documentation](shader-system.md) for uniforms, vertex format, effect branches, and how to add new effects.

### Components

Graphics components render specific UI elements:
- Sidebar
- Chat interface
- Library
- Settings
- And more

See [Components Documentation](components.md) for details.

## Rendering Pipeline

1. **Component Rendering**: Components generate vertices
2. **Text Queueing**: Text is queued for vello rendering
3. **Icon Queueing**: Icons are queued for vello rendering
4. **Batch Creation**: Vertices are batched by scissor rect
5. **Vello Rendering**: Text and icons rendered via vello
6. **Blit**: Vello output blitted to surface
7. **Present**: Frame presented to display

## Related Documentation

- [Rendering Pipeline](../architecture/rendering.md)
- [Component System](../architecture/components.md)
- [Shader System](shader-system.md)
- [Renderer API](renderer.md)
- [Renderable Trait](renderable.md)

