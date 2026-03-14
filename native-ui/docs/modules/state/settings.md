# Settings State

The `state/settings.rs` module manages application settings.

## SettingsState

```rust
pub struct SettingsState {
    pub api_base_url: String,
    pub model_id: Option<String>,
    // ... other settings
}
```

## Related Documentation

- [State Management](../architecture/state.md)
- [Persistence](../persistence/settings.md)

