# Icon System

The `gfx/icons.rs` module provides icon caching and rendering.

## IconCache

The `IconCache` struct caches parsed SVG icons:

```rust
pub struct IconCache {
    // Internal cache storage
}
```

### Methods

#### new()

Creates a new icon cache:

```rust
pub fn new() -> Self
```

#### get_icon()

Gets an icon by name:

```rust
pub fn get_icon(&mut self, name: &str) -> Option<&ParsedIcon>
```

Icons are:
- Loaded from SVG files
- Parsed into paths
- Cached for reuse

## Icon Rendering

Icons are rendered via vello:

```rust
renderer.queue_icon(
    &icon_name,
    position,
    size,
    color,
);
```

Icons are:
1. Loaded from `assets/icons/`
2. Parsed from SVG
3. Cached in `IconCache`
4. Rendered to vello scene
5. Blitted to surface

## Icon Files

Icons are stored as SVG files in `assets/icons/`.

## Related Documentation

- [Renderer](renderer.md)
- [Rendering Pipeline](../architecture/rendering.md)

