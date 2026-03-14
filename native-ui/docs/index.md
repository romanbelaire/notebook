# Notebook Native UI

A high-performance native UI implementation for Notebook, built with Rust using winit and wgpu. This documentation covers the complete architecture, API reference, and development guides for the Rust native UI infrastructure.

## Overview

The Notebook Native UI is a modern, GPU-accelerated user interface framework built from the ground up in Rust. It replaces the previous React + Tauri frontend with a fully native implementation that provides:

- **High Performance**: GPU-accelerated rendering using wgpu and vello
- **Native Feel**: Direct window management and event handling via winit
- **Type Safety**: Leverages Rust's type system for compile-time guarantees
- **Modular Architecture**: Component-based design with clear separation of concerns
- **Rich Text Support**: Advanced text rendering with Parley and markdown support

## Quick Start

### Prerequisites

- Rust 1.77 or later
- Cargo (Rust package manager)
- A graphics driver that supports Vulkan, Metal, or DirectX 12

### Building

```bash
cd native-ui
cargo build
```

### Running

```bash
cargo run
```

### Development Mode

For development with hot reloading support:

```bash
cargo run --features hot-reload
```

## Project Structure

The native UI codebase is organized into several key modules:

```
native-ui/src/
├── main.rs          # Application entry point, event loop
├── app.rs           # Application state and update logic
├── gfx/             # Graphics rendering system
│   ├── renderer.rs  # wgpu renderer, vello integration
│   ├── renderable.rs # Renderable trait definition
│   ├── components/  # Graphics components (sidebar, chat, etc.)
│   ├── types.rs     # Vertex, color, and shader types
│   ├── icons.rs     # Icon caching and rendering
│   └── pdf_renderer.rs # PDF rendering system
├── ui/              # UI component system
│   ├── core.rs      # Core primitives (Rect, Alignment, etc.)
│   ├── components/  # Container components (Root, VStack, etc.)
│   ├── window.rs    # Window abstraction
│   ├── button.rs    # Button component
│   ├── text_input.rs # Text input component
│   ├── text_editor.rs # Rich text editor
│   └── ...          # Other UI components
├── state/           # Application state management
│   ├── chat.rs      # Chat state
│   ├── ui.rs        # UI state
│   ├── settings.rs  # Settings state
│   └── insights.rs  # Insights state
├── api/             # Backend API client
│   ├── client.rs    # HTTP client implementation
│   └── models.rs    # API data structures
├── persistence/     # Data persistence
│   ├── conversation.rs
│   ├── document.rs
│   └── settings.rs
├── stylus/          # Rich text editor system
└── utils/           # Utility functions
    ├── layout.rs
    └── animation.rs
```

## Key Technologies

- **winit**: Cross-platform window creation and event handling
- **wgpu**: Modern graphics API abstraction (Vulkan, Metal, DirectX 12)
- **vello**: GPU-accelerated 2D graphics rendering
- **parley**: Advanced text layout and rendering
- **glam**: Fast math library for graphics
- **tokio**: Async runtime for network operations
- **reqwest**: HTTP client for API communication

## Architecture Highlights

### Component-Based Design

The UI follows a component-based architecture where every UI element is a self-contained, composable renderable. Components can be nested arbitrarily and manage their own layout, state, and rendering.

### Rendering Pipeline

The rendering system uses a hybrid approach:
- **Quad Rendering**: Simple UI elements rendered as quads via wgpu
- **Vello Rendering**: Complex 2D graphics, text, and icons via vello
- **Blit Pipeline**: Vello output is blitted to the main surface

### Event System

Events flow from winit → App → UI Components, with each component handling its own events. The system supports mouse, keyboard, touch, and window events.

### State Management

Application state is organized into logical modules (chat, UI, settings, insights) with clear boundaries. State can be persisted to disk and loaded on startup.

## Documentation Structure

This documentation is organized into several sections:

- **[Architecture](architecture/overview.md)**: High-level design, component system, rendering pipeline
- **[Modules](modules/app.md)**: Detailed documentation for each module
- **[API Reference](api/index.md)**: Complete API documentation
- **[Guides](guides/creating-components.md)**: Step-by-step development guides
- **[Examples](examples/basic-component.md)**: Code examples and tutorials
- **[Reference](reference/dependencies.md)**: Dependencies, architecture rules, migration notes

## Getting Help

- Check the [Architecture Overview](architecture/overview.md] for high-level design
- Review [Development Guides](guides/creating-components.md) for common tasks
- See [Examples](examples/basic-component.md) for code samples
- Check [Known Issues](reference/known-issues.md) for troubleshooting

## Contributing

When contributing to the native UI:

1. Follow the [Architecture Rules](reference/architecture-rules.md)
2. Ensure components are modular and composable
3. Write tests for new components
4. Update documentation for new features

## Next Steps

- Read the [Architecture Overview](architecture/overview.md) to understand the system design
- Explore [Creating Components](guides/creating-components.md) to learn how to build UI elements
- Check out [Examples](examples/basic-component.md) for practical code samples

