# Animation System

The `utils/animation.rs` module provides animation utilities.

## SpringAnimation

Spring-based animation for smooth transitions:

```rust
pub struct SpringAnimation {
    pub value: f32,
    pub velocity: f32,
    pub target: f32,
    pub stiffness: f32,
    pub damping: f32,
}
```

### Methods

- `update(dt)`: Update animation
- `set_target(target)`: Set target value

## Related Documentation

- [Utils Module](index.md)

