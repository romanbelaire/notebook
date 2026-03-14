# Insights State

The `state/insights.rs` module manages insights data.

## InsightsState

```rust
pub struct InsightsState {
    pub insights: Vec<Insight>,
    pub selected_insight: Option<String>,
    // ... other insight state
}
```

## Related Documentation

- [State Management](../architecture/state.md)

