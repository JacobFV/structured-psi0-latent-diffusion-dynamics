import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// Acceptance suite for the workbench. Always launched through tests/browser/run_browser_tests.py,
// which starts the real backend and provides RRP_BASE_URL / RRP_TOKEN. Headless chromium with
// software GL only (SwiftShader); no GPU is available inside the lease.
const out = process.env.RRP_PW_OUT ?? path.resolve(here, "../.cache/browser-run");

export default defineConfig({
  testDir: path.resolve(here, "../tests/browser"),
  testMatch: /.*\.spec\.ts$/,
  outputDir: path.join(out, "test-results"),
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: [["list"], ["json", { outputFile: path.join(out, "report.json") }]],
  use: {
    baseURL: process.env.RRP_BASE_URL ?? "http://127.0.0.1:8791",
    headless: true,
    video: "on",
    viewport: { width: 1600, height: 1000 },
    actionTimeout: 15_000,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1600, height: 1000 },
        launchOptions: {
          args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-gpu-compositing",
            "--renderer-process-limit=2", "--disable-dev-shm-usage"],
        },
      },
    },
  ],
});
