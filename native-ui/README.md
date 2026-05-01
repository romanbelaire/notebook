# Notebook Native UI

Native Rust UI implementation using winit + wgpu, replacing the React + Tauri frontend.

## Building

```bash
cargo build
cargo run
```

## PDF Preview Runtime Requirements

PDF preview in the modal uses PDFium through `pdfium-render`.

- Install a PDFium runtime library for your platform.
- Ensure the PDFium shared library is available on `PATH` before launching `notebook-native-ui`.
- If PDFium cannot be loaded, PDF preview fails loudly with a modal error explaining that PDFium binding failed.

### Viewer Controls

- `Prev` / `Next` page navigation.
- `-` and `+` zoom controls.
- `100%` zoom reset.

## Architecture

- `src/main.rs` - Entry point, event loop
- `src/app.rs` - App state, update logic
- `src/ui/` - UI components (windows, buttons, etc.)
- `src/gfx/` - Graphics rendering (wgpu, shaders)
- `src/state/` - State management (to be implemented)
- `src/api/` - Backend API client (to be implemented)
- `src/utils/` - Utilities (layout, animation) (to be implemented)

## Next Steps

See `../PORT_PLAN.plan.md` for the full migration plan.

