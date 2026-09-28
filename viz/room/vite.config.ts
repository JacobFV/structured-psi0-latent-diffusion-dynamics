import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { rrpApi } from './rrp-api-plugin.ts';

// viz/CONTRACT.md: port 3013, strict. Bind host defaults to 127.0.0.1; D-132 (owner) allows a network listener via
// RRP_ROOM_HOST=0.0.0.0. The API stays read-only either way.
const host = process.env.RRP_ROOM_HOST || '127.0.0.1';
export default defineConfig({
  plugins: [react(), rrpApi()],
  base: './',
  server: { host, port: 3013, strictPort: true },
  preview: { host, port: 3013, strictPort: true },
  build: { outDir: 'dist', emptyOutDir: true, sourcemap: false, chunkSizeWarningLimit: 2500 },
});
