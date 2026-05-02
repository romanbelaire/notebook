# PDFium (local binary)

Git ignores `pdfium.dll` / `*.so` / `*.dylib` here — download a prebuilt for your OS and place the shared library file in this folder as `pdfium.dll` (Windows) or matching name per platform.

Windows x64 refresh (PowerShell from `native-ui/`):

```powershell
curl.exe -L -o pdfium.tgz "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz"
New-Item -ItemType Directory -Force pdfium.tmp | Out-Null
tar -xzf pdfium.tgz -C pdfium.tmp
Copy-Item -Force pdfium.tmp/bin/pdfium.dll pdfium/pdfium.dll
Remove-Item -Recurse -Force pdfium.tmp, pdfium.tgz
```

Release index: https://github.com/bblanchon/pdfium-binaries/releases
