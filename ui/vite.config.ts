// @ts-nocheck
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import svgr from 'vite-plugin-svgr'

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), svgr()],
  resolve: {
    alias: {
      react: path.resolve(__dirname, '../node_modules/react'),
      'react-dom': path.resolve(__dirname, '../node_modules/react-dom'),
      '@tauri-apps/api/shell': path.resolve(__dirname, 'src/tauri-shell-stub.ts'),
      '@tauri-apps/plugin-fs': path.resolve(__dirname, 'src/tauri-fs-stub.ts'),
      '@tauri-apps/plugin-dialog': path.resolve(__dirname, 'src/tauri-dialog-stub.ts'),
    },
    dedupe: ['react', 'react-dom'],
  },
  optimizeDeps: {
    include: ['react', 'react-dom'],
  },
})
