# WGPU Idiosyncrasies and Quirks

This document tracks non-obvious behaviors, quirks, and gotchas discovered while working with WGPU. These are things that aren't immediately obvious from the documentation or that work differently than expected.

## Scissor Rect Coordinate System

**Issue**: WGPU documentation states that scissor rectangles use bottom-left origin (y=0 at bottom, y increases upward), but with our specific projection matrix setup, this doesn't apply.

**Our Setup**:
- Projection matrix: `orthographic_rh(0.0, width, height, 0.0, -1.0, 1.0)`
- This creates a top-left origin coordinate system (y=0 at top, y increases downward)
- UI coordinates are in top-left origin

**Discovery**: Despite WGPU docs saying scissor uses bottom-left origin, **with our projection matrix, scissor rects work correctly when using UI coordinates directly** (no conversion needed).

**Working Code**:
```rust
pub fn from_rect(rect: &Rect, viewport_height: f32) -> Self {
    // Use rect.y directly - no coordinate conversion needed!
    Self {
        x: rect.x.max(0.0) as u32,
        y: rect.y.max(0.0) as u32,  // Direct use, no flipping
        width: rect.width.max(0.0) as u32,
        height: rect.height.max(0.0) as u32,
    }
}
```

**What We Tried (That Didn't Work)**:
- `y = viewport_height - rect.y - rect.height` (bottom edge conversion) ❌
- `y = viewport_height - rect.y` (top edge conversion) ❌
- Various other coordinate transformations ❌

**Why This Works**: The projection matrix's top-left origin appears to make WGPU scissor operate in the same coordinate space as our UI coordinates. The scissor is applied after the projection transform, so it works in the projected coordinate space (which is top-left for us).

**Files Affected**:
- `native-ui/src/gfx/renderer.rs` - `ScissorRect::from_rect`
- `SCISSOR_RECT_IMPLEMENTATION.md` - Documentation

---

## Adding New Idiosyncrasies

When you discover a quirk or non-obvious behavior:

1. **Document it here** with:
   - Clear title
   - Description of the issue/behavior
   - What we expected vs. what actually happens
   - The working solution
   - What we tried that didn't work
   - Why it works (if known)
   - Files affected

2. **Keep it concise** - this is a reference document, not a tutorial

3. **Update related docs** - if there's a specific implementation doc, link to it

