export async function readTextFile(path: string): Promise<string> {
  // In the desktop build (Tauri), this function is provided by the FS plugin.
  // In a regular browser we fall back to fetching the file via HTTP.
  try {
    const resp = await fetch(path.startsWith('/') ? path : `/${path}`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${path}: ${resp.status} ${resp.statusText}`);
    }
    return await resp.text();
  } catch (err) {
    console.error(err);
    alert(`Failed to read file '${path}': ${String(err)}`);
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Additional Tauri FS plugin APIs required by web build
// ---------------------------------------------------------------------------

export const BaseDirectory = {
  Data: 'data',
  App: 'app',
  Home: 'home',
  Desktop: 'desktop',
  Document: 'document',
  Download: 'download',
  Music: 'music',
  Picture: 'picture',
  Public: 'public',
  Template: 'template',
  Video: 'video',
} as const;

export type BaseDirectory = keyof typeof BaseDirectory;

export async function readFile(path: string, _opts?: { baseDir?: BaseDirectory }): Promise<Uint8Array> {
  // Simple fetch-based fallback. We deliberately ignore baseDir because in the
  // browser build we serve files under /notes or /papers directly via FastAPI.
  try {
    const url = path.startsWith('/') ? path : `/${path}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
    }
    const buf = await resp.arrayBuffer();
    return new Uint8Array(buf);
  } catch (err) {
    console.error(err);
    alert(`Failed to read file '${path}': ${String(err)}`);
    throw err;
  }
}

export async function mkdir(_path: string, _opts?: { recursive?: boolean; baseDir?: BaseDirectory }): Promise<void> {
  console.warn('mkdir stub called in browser build – no-op');
}

export async function writeFile(_path: string, _data: Uint8Array | string, _opts?: { baseDir?: BaseDirectory }): Promise<void> {
  console.warn('writeFile stub called in browser build – no-op');
} 