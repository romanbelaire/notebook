# Notebook Native UI

Native Rust UI implementation using winit + wgpu, replacing the React + Tauri frontend.

## Building

```bash
cargo build
cargo run
```

## PDF Preview Runtime Requirements

PDF preview uses PDFium via `pdfium-render`. The app loads **`pdfium.dll`** (Windows), `libpdfium.so` (Linux), or `libpdfium.dylib` (macOS) in this order:

1. Path from env **`NOTEBOOK_PDFIUM_DLL`** (absolute path to the shared library).
2. Next to **`notebook-native-ui.exe`** (same folder as the executable).
3. **`native-ui/pdfium/`** (see `pdfium/README.md` for download/extract commands).
4. Current working directory when you launch the process (expects `pdfium.dll` in the cwd).
5. System library search (`PATH` on Windows).

Prebuilt binaries: [bblanchon/pdfium-binaries releases](https://github.com/bblanchon/pdfium-binaries/releases) — see `native-ui/pdfium/README.md` for a one-shot Windows download/copy, or drop **`pdfium.dll`** next to the exe (`NOTEBOOK_PDFIUM_DLL` also works).

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

