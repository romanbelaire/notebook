# Scissor Rect Clipping Implementation

## Overview

Implemented hardware-accelerated **scissor rectangle clipping** to prevent rendered content from bleeding outside container boundaries. This is essential for scrollable UI elements where items should be clipped to the visible region of their container.

## Problem Solved

**Before**: Scrollable list items would overlap/render outside their container boundaries when scrolled, creating visual artifacts:

```
┌─────────────────┐
│ Title Bar       │ ← Always visible
├─────────────────┤
│ Item 1 (half)   │ ← Partially scrolled out
│ Item 2          │ ← Fully visible  
│ Item 3          │ ← Fully visible
│ Item 4 (half)   │ ← Partially scrolled out, bleeds outside!
└─────────────────┘
     ↓
   [Overlap!] ← Item 4 renders over content below
```

**After**: Items are clipped to container bounds using GPU scissor rects:

```
┌─────────────────┐
│ Title Bar       │ ← Always visible
├─────────────────┤
│ Item 2          │ ← Top/bottom items clipped cleanly
│ Item 3          │
│ Item 4 (half)   │ ← Clipped at container boundary
└─────────────────┘
     ✓
   [Clean!] ← Nothing bleeds outside
```

## Technical Approach

### Scissor Rectangles

Used **WGPU's scissor rect** feature - a hardware-accelerated GPU feature that clips all rendering to a rectangular region. This is:
- **Fast**: Done in hardware, zero CPU cost
- **Perfect for UI**: UI containers are rectangular
- **Composable**: Scissor rects can be nested (intersected)

### Batched Rendering

Since scissor rects must be set before draw calls, and we want different clips for different components, implemented a **batching system**:

1. **Collect vertices** with current scissor state
2. **When scissor changes**, flush current batch and start new one
3. **At render time**, draw each batch with its scissor rect

## Implementation Details

### 1. ScissorRect Structure

```rust
#[derive(Debug, Clone, Copy)]
pub struct ScissorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ScissorRect {
    /// Convert from UI coordinates to WGPU scissor coordinates
    /// 
    /// NOTE: Despite WGPU docs saying scissor uses bottom-left origin, with our
    /// projection matrix (top-left origin), we can use UI coordinates directly!
    pub fn from_rect(rect: &Rect, viewport_height: f32) -> Self {
        // Use rect.y directly - no conversion needed with our projection matrix
        Self {
            x: rect.x.max(0.0) as u32,
            y: rect.y.max(0.0) as u32,
            width: rect.width.max(0.0) as u32,
            height: rect.height.max(0.0) as u32,
        }
    }
    
    /// Intersect two scissor rects (for nested clipping)
    pub fn intersect(&self, other: &ScissorRect) -> ScissorRect {
        // Returns smallest rectangle that fits within both
        // ...
    }
}
```

**Key Insight**: Despite WGPU documentation stating that scissor uses bottom-left origin, with our projection matrix (`orthographic_rh(0.0, width, height, 0.0, ...)`) which uses top-left origin, **the scissor rect can use UI coordinates directly without conversion**. The `from_rect` method simply passes through `rect.y` without flipping.

**IMPORTANT**: This is a quirk of our specific setup. The projection matrix's top-left origin appears to make WGPU scissor work in the same coordinate space. See `WGPU_IDIOSYNCRASIES.md` for details.

### 2. RenderBatch Structure

```rust
#[derive(Debug, Clone)]
struct RenderBatch {
    vertices: Vec<Vertex>,
    scissor: Option<ScissorRect>,
}
```

Each batch contains:
- **vertices**: Geometry to render
- **scissor**: Optional scissor rect to apply before rendering

### 3. Renderer State

Added to `Renderer` struct:

```rust
pub struct Renderer {
    // ... existing fields ...
    scissor_stack: Vec<ScissorRect>,
    viewport_height: f32,
    render_batches: Vec<RenderBatch>,
    current_batch_vertices: Vec<Vertex>,
}
```

- **scissor_stack**: Stack of active scissor rects (for nesting)
- **viewport_height**: For coordinate conversion
- **render_batches**: All batches to render this frame
- **current_batch_vertices**: Vertices being collected for current batch

### 4. Renderer API

```rust
impl Renderer {
    /// Push a scissor rect onto the stack for clipping
    pub fn push_scissor(&mut self, rect: &Rect) {
        self.flush_current_batch(); // Flush before changing state
        
        let scissor = ScissorRect::from_rect(rect, self.viewport_height);
        
        // Intersect with parent scissor if nested
        let final_scissor = if let Some(parent) = self.scissor_stack.last() {
            parent.intersect(&scissor)
        } else {
            scissor
        };
        
        self.scissor_stack.push(final_scissor);
    }
    
    /// Pop the most recent scissor rect
    pub fn pop_scissor(&mut self) {
        self.flush_current_batch(); // Flush before changing state
        self.scissor_stack.pop();
    }
    
    /// Flush current batch into render_batches
    fn flush_current_batch(&mut self) {
        if !self.current_batch_vertices.is_empty() {
            self.render_batches.push(RenderBatch {
                vertices: std::mem::take(&mut self.current_batch_vertices),
                scissor: self.scissor_stack.last().copied(),
            });
        }
    }
}
```

### 5. Frame Rendering Flow

```rust
pub fn render(&mut self, app: &mut App) -> Result<()> {
    // 1. Begin frame - clear batches
    self.begin_frame();
    
    // 2. Render all components (they call push/pop_scissor as needed)
    let mut vertices = Vec::new();
    for component in components {
        component.render(self, app, &mut vertices);
    }
    
    // 3. Add final vertices and flush
    self.add_vertices(&vertices);
    self.flush_current_batch();
    
    // 4. Render each batch with its scissor rect
    let mut render_pass = encoder.begin_render_pass(/* ... */);
    
    for batch in &self.render_batches {
        // Upload batch vertices to GPU
        self.queue.write_buffer(&self.vertex_buffer, offset, &batch.vertices);
        
        // Apply scissor rect
        if let Some(scissor) = &batch.scissor {
            render_pass.set_scissor_rect(
                scissor.x, scissor.y,
                scissor.width, scissor.height
            );
        }
        
        // Draw batch
        render_pass.draw(/* ... */);
    }
}
```

### 6. Component Usage

In sidebar rendering:

```rust
fn render_conversations_section(/* ... */) {
    // Render title (no clipping needed)
    renderer.queue_text("Conversations", /* ... */);
    
    // Get content rect and push scissor for clipping
    let content_rect = section.content_rect(sidebar_rect, y_offset);
    renderer.push_scissor(&content_rect);
    
    // Render scrollable items (will be clipped)
    for (i, conv) in conversations.iter().enumerate() {
        render_item(/* ... */);
    }
    
    // Pop scissor after rendering items
    renderer.pop_scissor();
}
```

## Nested Clipping

Scissor rects automatically nest via **intersection**:

```rust
// Outer container
renderer.push_scissor(&outer_rect); // Scissor A

// Inner container (nested)
renderer.push_scissor(&inner_rect); // Scissor B = A ∩ B

// Rendering here is clipped to both!

renderer.pop_scissor(); // Back to A
renderer.pop_scissor(); // Back to none
```

**Example**:
```
┌─────────────────────────┐ ← Outer scissor (sidebar)
│  Sidebar                │
│  ┌──────────────┐       │
│  │ Section 1    │       │ ← Inner scissor (section content)
│  │ - Item A     │       │
│  │ - Item B     │       │
│  └──────────────┘       │
│  ┌──────────────┐       │
│  │ Section 2    │       │ ← Another inner scissor
│  │ - Item C     │       │
│  └──────────────┘       │
└─────────────────────────┘
```

Each section's scissor is automatically intersected with the sidebar's scissor.

## Performance

### Advantages

1. **Hardware-accelerated**: GPU does the clipping, zero CPU cost
2. **Efficient batching**: Multiple draw calls only when scissor changes
3. **No overdraw**: Pixels outside scissor never reach fragment shader

### Trade-offs

- **Multiple draw calls**: One per batch (but batches are large - one per section typically)
- **Buffer uploads**: Each batch uploads vertices separately (could be optimized with suballocations)

### Optimization Opportunities

Future improvements:
1. **Pre-allocate large vertex buffer**: Use `write_buffer` with offsets instead of one upload per batch
2. **Batch merging**: Merge adjacent batches with same scissor state
3. **Dirty tracking**: Only re-batch when scissors change between frames

## Coordinate System Notes

### UI Coordinates (Top-Left Origin)
```
(0,0)
  ┌───────────────► X
  │
  │    ┌──────┐
  │    │ Item │
  │    └──────┘
  ▼
  Y
```

### WGPU Scissor Coordinates (Bottom-Left Origin)
```
  Y
  ▲
  │    ┌──────┐
  │    │ Item │
  │    └──────┘
  │
(0,0)─────────────► X
```

**Conversion**:
```rust
wgpu_y = viewport_height - ui_y - ui_height
```

## Usage Examples

### Example 1: Scrollable List

```rust
// Define visible region
let visible_rect = Rect::new(x, y, width, height);

// Push scissor before rendering items
renderer.push_scissor(&visible_rect);

// Render items (which may be partially outside visible_rect)
for item in items {
    let item_y = start_y + index * item_height - scroll_offset;
    render_item(item, item_y); // Automatically clipped
}

// Pop scissor
renderer.pop_scissor();
```

### Example 2: Modal with Scrollable Content

```rust
// Modal background (no clipping)
render_modal_bg(/* ... */);

// Modal title (no clipping)
render_modal_title(/* ... */);

// Modal content area (clipped)
let content_rect = modal_rect.inset(20.0); // 20px padding
renderer.push_scissor(&content_rect);
render_modal_content(/* scrolled content */);
renderer.pop_scissor();

// Modal buttons (no clipping)
render_modal_buttons(/* ... */);
```

### Example 3: Nested Containers

```rust
// Outer container
renderer.push_scissor(&outer_rect);

// Section 1
renderer.push_scissor(&section1_rect); // Intersects with outer
render_section1_items();
renderer.pop_scissor();

// Section 2
renderer.push_scissor(&section2_rect); // Intersects with outer
render_section2_items();
renderer.pop_scissor();

renderer.pop_scissor();
```

## Best Practices

### ✅ DO

1. **Always pair push/pop**: Every `push_scissor` must have matching `pop_scissor`
   ```rust
   renderer.push_scissor(&rect);
   // ... rendering ...
   renderer.pop_scissor();
   ```

2. **Use for scrollable containers**: Any content that scrolls should be scissored
   ```rust
   renderer.push_scissor(&visible_area);
   render_scrolled_content();
   renderer.pop_scissor();
   ```

3. **Nest naturally**: Push inner scissors without worrying about parents
   ```rust
   // Automatic intersection!
   renderer.push_scissor(&outer);
   renderer.push_scissor(&inner);
   // ...
   renderer.pop_scissor();
   renderer.pop_scissor();
   ```

4. **Scissor content, not containers**: Apply scissor to scrollable content, not the entire container
   ```rust
   render_title(); // No scissor - always visible
   renderer.push_scissor(&content_rect);
   render_items(); // Scissor - can scroll
   renderer.pop_scissor();
   ```

### ❌ DON'T

1. **Don't forget to pop**: Unbalanced push/pop causes incorrect clipping
   ```rust
   // BAD
   renderer.push_scissor(&rect);
   render_items();
   // Missing pop!
   ```

2. **Don't scissor static content**: Unnecessary overhead
   ```rust
   // BAD
   renderer.push_scissor(&screen_rect);
   render_background(); // Doesn't need clipping
   renderer.pop_scissor();
   ```

3. **Don't scissor for simple bounds checking**: Use early-return instead
   ```rust
   // BAD
   renderer.push_scissor(&visible_rect);
   for item in items {
       render_item(item); // All items rendered, GPU clips
   }
   renderer.pop_scissor();
   
   // GOOD
   for item in items {
       if !visible_rect.intersects(item.rect) {
           continue; // Skip rendering entirely
       }
       render_item(item);
   }
   ```

## Debugging

### Visual Debugging

To visualize scissor rects, add debug rendering:

```rust
if DEBUG_SCISSORS {
    for batch in &render_batches {
        if let Some(scissor) = &batch.scissor {
            // Render scissor rect outline
            render_debug_rect(scissor, Color::RED);
        }
    }
}
```

### Common Issues

**Issue**: Content still bleeds outside
- **Cause**: Missing `push_scissor` call
- **Fix**: Add `renderer.push_scissor(&content_rect)` before rendering items

**Issue**: Nothing renders
- **Cause**: Scissor rect has zero width/height or is offscreen
- **Fix**: Check rect calculations, ensure positive width/height

**Issue**: Wrong clipping region
- **Cause**: Coordinate system mismatch (top-left vs bottom-left)
- **Fix**: Use `ScissorRect::from_rect` which handles conversion

**Issue**: Unbalanced push/pop
- **Cause**: Missing `pop_scissor` or early return without popping
- **Fix**: Use RAII guard pattern (future improvement)

## Future Enhancements

### 1. RAII Guard Pattern

```rust
pub struct ScissorGuard<'a> {
    renderer: &'a mut Renderer,
}

impl<'a> Drop for ScissorGuard<'a> {
    fn drop(&mut self) {
        self.renderer.pop_scissor();
    }
}

impl Renderer {
    pub fn push_scissor_guard(&mut self, rect: &Rect) -> ScissorGuard<'_> {
        self.push_scissor(rect);
        ScissorGuard { renderer: self }
    }
}

// Usage
{
    let _guard = renderer.push_scissor_guard(&rect);
    render_items();
} // Automatically pops on scope exit
```

### 2. Stencil Buffer Support

For non-rectangular clipping (circles, rounded corners):

```rust
renderer.push_stencil_mask(|r| {
    r.render_rounded_rect(/* ... */); // Mask shape
});
render_content(); // Clipped to mask
renderer.pop_stencil_mask();
```

### 3. Shader-Based Clipping

For complex clipping without state changes:

```wgsl
// In fragment shader
if (position.x < clip_rect.x || position.x > clip_rect.z ||
    position.y < clip_rect.y || position.y > clip_rect.w) {
    discard;
}
```

## Summary

**Scissor rect clipping provides:**
- ✅ Clean rendering of scrollable content
- ✅ Hardware-accelerated performance
- ✅ Automatic nested clipping
- ✅ Simple API (`push`/`pop`)
- ✅ Zero overdraw for clipped regions

**Key takeaway**: Always use `push_scissor`/`pop_scissor` around scrollable content to prevent visual artifacts!

## Related Files

- **`native-ui/src/gfx/renderer.rs`**: Core implementation
- **`native-ui/src/gfx/components/sidebar_content.rs`**: Usage example in sidebar
- **`native-ui/src/ui/core.rs`**: `Rect` utilities
- **`ARCHITECTURE_RULES.md`**: Component architecture principles
- **`COMPONENT_USAGE_EXAMPLE.md`**: Component system examples

