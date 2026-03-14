# Testing

This guide explains testing strategies for the Notebook Native UI.

## Unit Tests

Test individual components:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_component_layout() {
        let mut component = MyComponent::new();
        component.update_layout(Rect::new(0.0, 0.0, 100.0, 50.0));
        assert_eq!(component.bounds().width, 100.0);
    }
}
```

## Integration Tests

Test component interactions:

```rust
#[test]
fn test_component_rendering() {
    let component = MyComponent::new();
    let mut vertices = Vec::new();
    // ... setup renderer and app
    component.render(&mut renderer, &app, &mut vertices);
    assert!(!vertices.is_empty());
}
```

## Best Practices

1. **Test layout**: Verify `update_layout()` works correctly
2. **Test bounds**: Ensure `bounds()` returns correct values
3. **Test rendering**: Verify components render without errors
4. **Test events**: Test event handling logic

## Related Documentation

- [Component System](../architecture/components.md)

