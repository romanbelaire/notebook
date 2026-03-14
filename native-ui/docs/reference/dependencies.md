# Dependencies

This document explains the key dependencies used in the Notebook Native UI.

## Core Dependencies

### winit

Window creation and event handling:

```toml
winit = "0.30"
```

Provides:
- Cross-platform window creation
- Event handling (mouse, keyboard, touch)
- Window management

### wgpu

Modern graphics API:

```toml
wgpu = "22.0"
```

Provides:
- Graphics API abstraction (Vulkan/Metal/DirectX 12)
- Device and surface management
- Render pipelines
- Shader support

### vello

GPU-accelerated 2D graphics:

```toml
vello = "0.3"
```

Provides:
- 2D graphics rendering
- Text rendering
- SVG path rendering

### parley

Advanced text layout:

```toml
parley = "0.2"
```

Provides:
- Text layout engine
- Font management
- Glyph positioning

### glam

Fast math library:

```toml
glam = "0.29"
```

Provides:
- Vec2, Vec4, Mat4 types
- Math operations
- Graphics math utilities

## Async & Networking

### tokio

Async runtime:

```toml
tokio = { version = "1", features = ["rt", "rt-multi-thread"] }
```

Provides:
- Async runtime
- Task spawning
- Async I/O

### reqwest

HTTP client:

```toml
reqwest = { version = "0.11", features = ["json"] }
```

Provides:
- HTTP client
- JSON serialization
- Async requests

## Serialization

### serde

Serialization framework:

```toml
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

Provides:
- JSON serialization
- Data structure serialization

## Text & Markdown

### pulldown-cmark

Markdown parser:

```toml
pulldown-cmark = "0.12"
```

Provides:
- Markdown parsing
- Text formatting

## SVG

### usvg

SVG parsing:

```toml
usvg = "0.40"
```

Provides:
- SVG file parsing
- Path extraction

## Utilities

### anyhow

Error handling:

```toml
anyhow = "1.0"
```

### env_logger

Logging:

```toml
env_logger = "0.11"
```

### rfd

File dialogs:

```toml
rfd = "0.15"
```

## Related Documentation

- [Cargo.toml](../../Cargo.toml)

