# Migration Notes

This document covers migration notes for major changes in the Notebook Native UI.

## Vello Refactor

The rendering system was migrated from glyph_brush to vello + parley.

### Changes

- Text rendering now uses Parley for layout
- Icons rendered via vello
- Blit pipeline added for vello output

### Migration Guide

See [VELLO_REFACTOR.md](../../VELLO_REFACTOR.md) for details.

## Winit 0.30 Migration

The window system was migrated to winit 0.30.

### Changes

- `EventLoop` API changed
- `ApplicationHandler` trait introduced
- `KeyEvent` replaces `KeyboardInput`
- `VirtualKeyCode` replaced with `Key::Named`

### Migration Guide

- Update event handling to use new APIs
- Replace `VirtualKeyCode` with `Key::Named`
- Use `ApplicationHandler` trait

## WGPU 22.0 Migration

The graphics system was migrated to wgpu 22.0.

### Changes

- `RenderPassDescriptor` API changed
- `StoreOp` enum introduced
- Additional fields required in descriptors

### Migration Guide

- Update render pass descriptors
- Add required fields (occlusion_query_set, timestamp_writes)
- Use `Some()` wrappers for optional fields

## Related Documentation

- [VELLO_REFACTOR.md](../../VELLO_REFACTOR.md)
- [Known Issues](known-issues.md)

