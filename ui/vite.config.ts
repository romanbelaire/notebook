// @ts-nocheck
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import svgr from 'vite-plugin-svgr'
import fs from 'node:fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Fix: Prevent CSS files from being transformed to JS exports before PostCSS
const fixCssPlugin = () => ({
  name: 'fix-css-before-postcss',
  enforce: 'pre', // Run before vite:css plugin
  load(id) {
    // Load CSS files directly, preventing JS export transformation
    if (id.endsWith('.css') && !id.includes('html-proxy')) {
      try {
        const content = fs.readFileSync(id, 'utf8');
        // Return the CSS content as-is, preventing transformation to JS export
        return content;
      } catch (e) {
        // File might not exist, let Vite handle it
        return null;
      }
    }
  },
  transform(code, id) {
    // If CSS file was transformed to JS export, restore original CSS content
    if (id.endsWith('.css') && !id.includes('html-proxy')) {
      if (code.startsWith('export default')) {
        try {
          const content = fs.readFileSync(id, 'utf8');
          return content;
        } catch (e) {
          return null;
        }
      }
    }
  },
  // Handle HMR updates to prevent CSS from being re-transformed incorrectly
  handleHotUpdate({ file, server }) {
    if (file.endsWith('.css')) {
      // Force reload CSS files to ensure they're processed correctly
      server.ws.send({
        type: 'update',
        updates: [{
          type: 'css-update',
          path: file,
          timestamp: Date.now(),
        }],
      });
      return [];
    }
  },
  // Handle CSS files served via middleware (for dynamically loaded CSS)
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      // If requesting a CSS file from whisk-submodule, serve it as static asset without processing
      if (req.url?.endsWith('.css') && req.url.includes('/whisk-submodule/')) {
        // Set proper content type and let it pass through as static asset
        res.setHeader('Content-Type', 'text/css');
        next();
      } else {
        next();
      }
    });
  },
});

// Fix: Rewrite Wisk plugin import paths to use correct asset path
const fixWiskImportsPlugin = () => ({
  name: 'fix-wisk-imports',
  enforce: 'pre',
  transform(code, id) {
    // Only transform files from whisk-submodule that have import statements
    if (id.includes('whisk-submodule') && id.endsWith('.js')) {
      // Rewrite absolute paths like /a7/cdn/... to /whisk-submodule/a7/cdn/...
      if (code.includes("from '/a7/") || code.includes('from "/a7/')) {
        return {
          code: code.replace(/from\s+['"]\/a7\//g, "from '/whisk-submodule/a7/"),
          map: null
        };
      }
    }
  },
  // Also handle requests for /a7/ paths and redirect to /whisk-submodule/a7/
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      // Redirect /a7/ requests to /whisk-submodule/a7/ for Wisk assets
      if (req.url?.startsWith('/a7/')) {
        req.url = `/whisk-submodule${req.url}`;
      }
      next();
    });
  },
});

// https://vite.dev/config/
export default defineConfig({
  plugins: [fixCssPlugin(), fixWiskImportsPlugin(), react(), svgr()],
  server: {
    fs: {
      allow: ['..'] // Allow serving files from parent directories (for whisk-submodule)
    }
  },
  publicDir: 'public',
  // Configure static file serving for Wisk assets
  assetsInclude: ['**/*.js', '**/*.css', '**/*.html'],
  resolve: {
    alias: {
      '@tauri-apps/api/shell': path.resolve(__dirname, 'src/tauri-shell-stub.ts'),
      '@tauri-apps/plugin-fs': path.resolve(__dirname, 'src/tauri-fs-stub.ts'),
      '@tauri-apps/plugin-dialog': path.resolve(__dirname, 'src/tauri-dialog-stub.ts'),
    },
    dedupe: ['react', 'react-dom'],
  },
})
