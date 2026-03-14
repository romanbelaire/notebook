# Architecture Rules

The Notebook Native UI follows strict architecture rules to ensure modularity, maintainability, and extensibility.

## Core Principles

See the [Architecture Rules](../../../requirements/ARCHITECTURE_RULES.md) document for complete rules.

### Key Rules

1. **Component-Based**: Every UI element is a self-contained, composable renderable
2. **Composition Over Inheritance**: Build complex UIs by composing simple components
3. **Data-Driven**: Never hardcode UI structure, always initialize from data
4. **Plan for Unknowns**: Design components to handle future requirements
5. **Encapsulation**: Each component manages its own layout, state, and rendering
6. **Declarative Structure**: Describe WHAT the UI should be, not HOW to build it
7. **Type Safety**: Use the type system to prevent errors at compile time

## Anti-Patterns to Avoid

- Global state access
- Hardcoded positions
- Manual Y coordinate calculations
- Tight coupling between components
- Inheritance hierarchies
- Procedural UI construction
- Components knowing about their parents
- Direct manipulation of siblings

## Related Documentation

- [Architecture Rules](../../../requirements/ARCHITECTURE_RULES.md)
- [Component System](../architecture/components.md)

