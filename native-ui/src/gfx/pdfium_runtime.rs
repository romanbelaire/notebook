use anyhow::{Context, Result};
use pdfium_render::prelude::Pdfium;

pub fn bind_pdfium() -> Result<Pdfium> {
    let bindings = Pdfium::bind_to_system_library().context(
        "Failed to bind to system PDFium library. Install PDFium and ensure it is on PATH.",
    )?;
    Ok(Pdfium::new(bindings))
}

