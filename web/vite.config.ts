import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The backend (python -m app.server) listens on 127.0.0.1:8000; the dev server proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  preview: {
    port: 4173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 900 },
});
