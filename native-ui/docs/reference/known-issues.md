# Known Issues

This document lists known issues and limitations in the Notebook Native UI.

## Rendering Issues

### Text Rendering

- Text measurement caching may need optimization for large amounts of text
- Complex markdown rendering may have performance issues

### Scissor Rects

- Nested scissor rects work correctly but may have edge cases with very deep nesting

## Event Handling

### Focus Management

- Focus traversal (Tab/Shift+Tab) may need improvements
- Focus state persistence across window focus changes

## Performance

### Large Lists

- Rendering very large lists (1000+ items) may have performance issues
- Consider virtual scrolling for large lists

## Platform-Specific

### Windows

- Some DPI scaling edge cases may exist

### macOS

- Window dragging behavior may need refinement

### Linux

- Some window managers may have compatibility issues

## Related Documentation

- [Migration Notes](migration.md)

