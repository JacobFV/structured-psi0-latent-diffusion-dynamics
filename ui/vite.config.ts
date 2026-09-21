import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built assets are served by the FastAPI backend (ui/dist mounted at / and /assets).
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false, chunkSizeWarningLimit: 2000 },
  server: { host: "127.0.0.1", strictPort: true },
  preview: { host: "127.0.0.1" },
});
