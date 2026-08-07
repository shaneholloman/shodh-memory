import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { viteSingleFile } from "vite-plugin-singlefile";
import path from "node:path";

// The shodh-front Rust binary embeds the built UI with include_str!, so the
// build MUST produce exactly one self-contained file at dist/index.html —
// no sibling .js/.css, no hashed asset names to chase. viteSingleFile inlines
// every chunk and stylesheet into the HTML. Code splitting buys nothing here
// (one screen, loaded from localhost) and would break the embed.
//
// Dev mirrors what the Rust front does in production: proxy /api/* to the
// backend. Without this, fetches 404 in `vite dev` and work in the binary,
// which is the worst possible way for a bug to present.
const BACKEND = process.env.SHODH_API_URL ?? "http://127.0.0.1:3030";
const API_KEY = process.env.SHODH_API_KEY ?? "";

export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
  server: {
    port: 8788,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
        // Server-Sent Events (/api/events — the live recall river) must not be
        // buffered, and the dev proxy has to inject the key the Rust front
        // injects in production.
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            if (API_KEY) proxyReq.setHeader("X-API-Key", API_KEY);
          });
        },
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Inlining requires a single chunk; assetsInlineLimit lifts the cutoff so
    // nothing is emitted as a separate file the Rust binary would not embed.
    assetsInlineLimit: 100_000_000,
    cssCodeSplit: false,
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
