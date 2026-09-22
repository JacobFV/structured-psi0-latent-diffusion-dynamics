// Workbench acceptance tests: real backend (started by tests/browser/run_browser_tests.py), no mocks.
// Assertions are semantic (backend-reported state rendered by the UI), not pixel snapshots.
import { test, expect, type Page } from "@playwright/test";

const TOKEN = process.env.RRP_TOKEN ?? "";
const BASE = process.env.RRP_BASE_URL ?? "http://127.0.0.1:8791";

async function open(page: Page, query = "") {
  page.on("dialog", (d) => void d.accept());          // confirmations (teleport, reset, debug) are accepted explicitly
  await page.goto(`/?token=${encodeURIComponent(TOKEN)}${query}`);
  await expect(page.getByLabel("robot", { exact: true })).toBeVisible();
}

async function createSession(page: Page, robot = "parm5_pg2", seed = 1): Promise<string> {
  await page.getByLabel("robot", { exact: true }).selectOption(robot);
  await page.getByLabel("seed", { exact: true }).fill(String(seed));
  await page.getByRole("button", { name: "Create session" }).click();
  await expect(page.getByTestId("session-id")).toContainText(robot);
  await expect(page.getByTestId("ws-state")).toHaveText("open");
  const sid = new URL(page.url()).searchParams.get("session");
  expect(sid).toBeTruthy();
  return sid!;
}

async function step(page: Page, n: number) {
  await page.getByLabel("steps per click", { exact: true }).fill(String(n));
  await page.getByRole("button", { name: "Step", exact: true }).click();
}

const tab = (page: Page, name: string) => page.getByRole("tab", { name, exact: true }).click();
const graphVersion = async (page: Page) => Number((await page.getByTestId("graph-version").innerText()).replace(/\D+/g, ""));
const alertWithCode = (page: Page, code: string) => page.locator(`[role=alert][data-code="${code}"]`);
const num = async (page: Page, testId: string) => Number((await page.getByTestId(testId).innerText()).trim());

async function selectEvent(page: Page, id: string) {
  await page.getByTestId(`event-node-${id}`).click();
  await tab(page, "Selection");
  await expect(page.getByTestId("event-inspector")).toContainText(`Event ${id}`);
}

async function addDependency(page: Page, src: string, dst: string) {
  await page.getByLabel("dependency source (prerequisite)", { exact: true }).selectOption(src);
  await page.getByLabel("dependency target (dependent)", { exact: true }).selectOption(dst);
  await page.getByRole("button", { name: "Connect", exact: true }).click();
}

function vecOf(s: string): number[] { return s.replace(/[[\]]/g, "").split(",").map(Number); }

test("select robot, create session, step, inspect one joint and see joint values change", async ({ page }) => {
  await open(page);
  await createSession(page, "parm5_tf3");
  await expect(page.getByTestId("mode-badge")).toHaveAttribute("data-mode", "hold");
  // inspect one joint from the morphology tree
  await tab(page, "Morphology");
  await expect(page.getByTestId("independent-controls")).not.toHaveText("");
  await page.getByTestId("morph-joint-r0_j1").click();
  await tab(page, "Selection");
  await expect(page.getByTestId("joint-detail")).toContainText("r0_j1");
  const q0 = Number(await page.getByTestId("joint-q").innerText());
  // command joint 1 through the validated controller, then step physics
  const target = q0 + 0.4;
  await page.getByLabel("arm[1] value", { exact: true }).fill(target.toFixed(3));
  await page.getByRole("button", { name: "Send arm joint target" }).click();
  await step(page, 40);
  await expect(page.getByTestId("mode-badge")).toHaveAttribute("data-mode", "user");
  await expect.poll(async () => Math.abs(Number(await page.getByTestId("joint-q").innerText()) - q0), { timeout: 15_000 }).toBeGreaterThan(0.15);
  // an out-of-bounds command is rejected by the backend with a visible error
  await page.getByLabel("arm[1] value", { exact: true }).fill("9");
  await page.getByRole("button", { name: "Send arm joint target" }).click();
  await expect(alertWithCode(page, "out_of_bounds")).toBeVisible();
});

test("click-to-select a body in the 3D scene", async ({ page }) => {
  await open(page);
  await createSession(page);
  await page.waitForFunction(() => (window.__rrpScene?.rendered ?? 0) > 0);
  const pt = await page.evaluate(() => window.__rrpScene!.projectBody("cube"));
  expect(pt).not.toBeNull();
  await page.mouse.click(pt!.x, pt!.y);
  await expect(page.getByTestId("selected-body")).toHaveText("cube");
  await expect(page.getByTestId("body-detail")).toContainText("red cube");
});

test("move an end-effector target and observe motion after stepping", async ({ page }) => {
  await open(page);
  await createSession(page);
  await page.getByLabel("select body", { exact: true }).selectOption("r0_wrist_palm");
  const p0 = vecOf(await page.getByTestId("body-pos").innerText());
  const goal = [0.3, -0.12, 0.22];
  await page.getByLabel("ee target x", { exact: true }).fill(String(goal[0]));
  await page.getByLabel("ee target y", { exact: true }).fill(String(goal[1]));
  await page.getByLabel("ee target z", { exact: true }).fill(String(goal[2]));
  await page.getByRole("button", { name: "Send EE target" }).click();
  await expect(page.getByTestId("mode-badge")).toHaveAttribute("data-mode", "user");
  await step(page, 80);
  await expect.poll(async () => {
    const p = vecOf(await page.getByTestId("body-pos").innerText());
    return Math.hypot(p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]);
  }, { timeout: 15_000 }).toBeGreaterThan(0.05);
  const p1 = vecOf(await page.getByTestId("body-pos").innerText());
  const d = (p: number[]) => Math.hypot(p[0] - goal[0], p[1] - goal[1]);
  expect(d(p1)).toBeLessThan(d(p0));       // moved toward the requested goal (horizontal)
  // unreachable goal is rejected with a visible reason
  await page.getByLabel("ee target x", { exact: true }).fill("3");
  await page.getByRole("button", { name: "Send EE target" }).click();
  await expect(alertWithCode(page, "unreachable")).toBeVisible();
});

test("create an event, connect dependencies (form and handle drag), reject a cycle", async ({ page }) => {
  await open(page);
  const sid = await createSession(page);
  const v0 = await graphVersion(page);
  await page.getByRole("button", { name: "Add event…" }).click();
  await page.getByLabel("new event id", { exact: true }).fill("inspect_cube");
  await page.getByLabel("new event operator", { exact: true }).fill("inspect");
  await page.getByLabel("new event actor", { exact: true }).selectOption("");
  await page.getByLabel("new event patient", { exact: true }).selectOption("cube");
  await page.getByRole("button", { name: "Add event", exact: true }).click();
  await expect(page.getByTestId("event-node-inspect_cube")).toBeVisible();
  await expect.poll(() => graphVersion(page)).toBe(v0 + 1);
  // connect: inspect_cube must complete before place (keyboard-accessible form)
  await addDependency(page, "inspect_cube", "place");
  await expect(page.getByTestId("rf__edge-c:inspect_cube->place")).toBeVisible();
  await expect.poll(() => graphVersion(page)).toBe(v0 + 2);
  // connect by dragging from grasp's output handle to inspect_cube's input handle
  const src = page.locator('[data-testid="event-node-grasp"] .react-flow__handle.source');
  const dst = page.locator('[data-testid="event-node-inspect_cube"] .react-flow__handle.target');
  const a = (await src.boundingBox())!, b = (await dst.boundingBox())!;
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
  await page.mouse.down();
  await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 5 });
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 5 });
  await page.mouse.up();
  await expect(page.getByTestId("rf__edge-c:grasp->inspect_cube")).toBeVisible();
  await expect.poll(() => graphVersion(page)).toBe(v0 + 3);
  // cycle: place -> grasp would close grasp -> place; rejected, graph unchanged
  await addDependency(page, "place", "grasp");
  await expect(alertWithCode(page, "dependency_cycle")).toBeVisible();
  await page.waitForTimeout(300);
  expect(await graphVersion(page)).toBe(v0 + 3);
  const g = await (await page.request.get(`${BASE}/api/sessions/${sid}/graph`)).json();
  expect(g.graph.document.events.find((e: { id: string }) => e.id === "grasp").requires_completed).toEqual([]);
});

test("reject an unsatisfied execution request (place before grasp)", async ({ page }) => {
  await open(page);
  await createSession(page);
  await selectEvent(page, "place");
  await page.getByRole("button", { name: "Request execution" }).click();
  await expect(alertWithCode(page, "prerequisite_unsatisfied")).toBeVisible();
  await expect(page.getByTestId("rejections-place")).toContainText("prerequisite_unsatisfied");
  await expect(page.getByTestId("status-place")).not.toHaveText("active");
  await expect(page.getByTestId("status-grasp")).not.toHaveText("succeeded");   // a future request never completes predecessors
  await expect(page.getByTestId("event-rejections")).toContainText("prerequisite_unsatisfied");
});

test("change actor binding and observe changed routing and controller ownership", async ({ page }) => {
  await open(page);
  await createSession(page);
  await tab(page, "Routing");
  await expect(page.getByTestId("route-grasp")).toContainText("actor[0]=gripper");
  await page.getByRole("button", { name: "Add entity…" }).click();
  await page.getByLabel("new entity id", { exact: true }).fill("gripper2");
  await page.getByLabel("new entity type", { exact: true }).selectOption("manipulator");
  await page.getByLabel("new entity descriptor", { exact: true }).fill("second manipulator");
  await page.getByRole("button", { name: "Add entity", exact: true }).click();
  await expect(page.getByTestId("owner-gripper2")).toContainText("free");
  // rebind grasp's actor[0] to gripper2
  await selectEvent(page, "grasp");
  await page.getByLabel("bind role", { exact: true }).selectOption("actor");
  await page.getByLabel("bind ordinal", { exact: true }).fill("0");
  await page.getByLabel("binding kind", { exact: true }).selectOption("entity");
  await page.getByLabel("bind entity", { exact: true }).selectOption("gripper2");
  await page.getByRole("button", { name: "Bind", exact: true }).click();
  await expect(page.getByTestId("event-node-grasp")).toContainText("actor[0] → gripper2");
  await page.getByRole("button", { name: "Request execution" }).click();
  await expect(page.getByTestId("status-grasp")).toHaveText("active");
  await tab(page, "Routing");
  await expect(page.getByTestId("route-grasp")).toContainText("actor[0]=gripper2");
  await expect(page.getByTestId("route-grasp")).toContainText("gripper2");
  await expect(page.getByTestId("owner-gripper2")).toContainText("grasp");
  await expect(page.getByTestId("owner-gripper")).toContainText("free");
  // manipulator task query for the original gripper no longer routes grasp to it
  await tab(page, "Probes");
  await page.getByLabel("probe query", { exact: true }).selectOption("manipulator_tasks");
  await page.getByLabel("probe subject", { exact: true }).fill("gripper");
  await page.getByRole("button", { name: "Run probe" }).click();
  const res = page.getByTestId("probe-result-manipulator_tasks").first();
  await expect(res).toContainText("place");
  await expect(res).not.toContainText('"event":"grasp"');
});

test("edit while executing drops queued chunks; stale-version edit rejected with 409", async ({ page }) => {
  await open(page);
  const sid = await createSession(page);
  await page.getByLabel("control mode", { exact: true }).selectOption("scripted_teacher");
  await page.getByRole("button", { name: "Apply mode" }).click();
  await expect(page.getByTestId("mode-badge")).toContainText("SCRIPTED TEACHER (privileged)");
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause" })).toBeVisible();
  const v0 = await graphVersion(page);
  await selectEvent(page, "grasp");
  await page.getByLabel("priority", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Set priority" }).click();
  await expect.poll(() => graphVersion(page)).toBe(v0 + 1);
  await page.getByRole("button", { name: "Pause" }).click();
  await tab(page, "Log");
  await expect(page.locator('[data-kind="graph_committed"]').first()).toContainText("queue_dropped=true");
  await expect(page.locator('[data-kind="graph_committed"]').first()).toContainText(`graph v${v0 + 1}`);
  // another client commits an edit behind this UI's back
  const r = await page.request.post(`${BASE}/api/sessions/${sid}/graph`, {
    headers: { "x-rrp-token": TOKEN, "content-type": "application/json" },
    data: { expected_version: v0 + 1, request_id: "other-client-1", operations: [{ op: "set_priority", event_id: "place", priority: 5 }] },
  });
  expect(r.status()).toBe(200);
  await expect.poll(() => graphVersion(page)).toBe(v0 + 2);
  await expect(page.getByTestId("edit-base")).toContainText("stale");
  await tab(page, "Selection");
  await page.getByLabel("priority", { exact: true }).fill("3");
  await page.getByRole("button", { name: "Set priority" }).click();
  const conflict = page.locator('[role=alert]').filter({ hasText: "HTTP 409" });
  await expect(conflict).toBeVisible();
  expect(await graphVersion(page)).toBe(v0 + 2);                     // no partial mutation
  await conflict.getByRole("button", { name: "Refresh & rebase" }).click();
  await expect(page.getByTestId("edit-base")).not.toContainText("stale");
  await page.getByRole("button", { name: "Set priority" }).click();
  await expect.poll(() => graphVersion(page)).toBe(v0 + 3);
});

test("disconnect/reconnect WebSocket rehydrates without duplicating commands", async ({ page }) => {
  await open(page);
  await createSession(page);
  await step(page, 5);
  await expect(page.getByTestId("timeline-count")).toHaveText("5 recorded steps");
  await tab(page, "Log");
  const snaps0 = await num(page, "snapshots-received");
  const connects0 = await num(page, "ws-connects");
  await page.getByRole("button", { name: "Reconnect WS" }).click();
  await expect.poll(() => num(page, "ws-connects")).toBe(connects0 + 1);
  await expect(page.getByTestId("ws-state")).toHaveText("open");
  await expect.poll(() => num(page, "snapshots-received")).toBeGreaterThan(snaps0);
  await page.waitForTimeout(800);
  await expect(page.getByTestId("timeline-count")).toHaveText("5 recorded steps");   // nothing replayed
  await expect(page.getByTestId("sim-time")).toContainText("seq 5");
  await step(page, 1);
  await expect(page.getByTestId("timeline-count")).toHaveText("6 recorded steps");
  await expect(page.getByTestId("sim-time")).toContainText("seq 6");
});

test("physics replay reproduces the recorded episode; export downloads it", async ({ page }) => {
  await open(page);
  await createSession(page);
  await page.getByLabel("arm[0] value", { exact: true }).fill("0.3");
  await page.getByRole("button", { name: "Send arm joint target" }).click();
  await step(page, 25);
  await expect(page.getByTestId("timeline-count")).toHaveText("25 recorded steps");
  await page.getByRole("button", { name: "Physics replay" }).click();
  await expect(page.getByTestId("replay-result")).toContainText("final_state_match=true");
  await expect(page.getByTestId("replay-result")).toContainText("replay 25 steps");
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export episode" }).click()]);
  expect(dl.suggestedFilename()).toMatch(/^episode-.*\.json$/);
});

test("probes show null/unknown answers for contact, held_by and object QA", async ({ page }) => {
  await open(page);
  await createSession(page);
  await tab(page, "Probes");
  await page.getByLabel("probe query", { exact: true }).selectOption("held_by");
  await page.getByLabel("probe subject", { exact: true }).fill("gripper");
  await page.getByRole("button", { name: "Run probe" }).click();
  const held = page.getByTestId("probe-result-held_by").first();
  await expect(held).toContainText("source: public_estimator");
  await expect(held).toContainText("holds=[] (none)");
  await expect(held).toContainText("null=true");
  await page.getByLabel("probe query", { exact: true }).selectOption("contact_mode");
  await page.getByRole("button", { name: "Run probe" }).click();
  await expect(page.getByTestId("probe-result-contact_mode").first()).toContainText("contact_mode=free");
  await page.getByLabel("probe query", { exact: true }).selectOption("acting_on");
  await page.getByRole("button", { name: "Run probe" }).click();
  await expect(page.getByTestId("probe-result-acting_on").first()).toContainText("acting_on=[] (none)");
  await page.getByLabel("probe query", { exact: true }).selectOption("object_qa");
  await page.getByLabel("probe subject", { exact: true }).fill("");
  await page.getByLabel("probe object", { exact: true }).fill("cube");
  await page.getByLabel("probe question", { exact: true }).fill("what color is it?");
  await page.getByRole("button", { name: "Run probe" }).click();
  const qa = page.getByTestId("probe-result-object_qa").first();
  await expect(qa.getByTestId("probe-unknown")).toContainText("no object-QA model");
  await expect(qa.getByTestId("probe-null")).toBeVisible();
});

test("teacher / learned / user / debug labels render with stable colors", async ({ page }) => {
  await open(page);
  await createSession(page);
  const legend = page.getByTestId("source-legend");
  await expect(page.getByTestId("legend-scripted_teacher")).toHaveText("SCRIPTED TEACHER (privileged)");
  await expect(page.getByTestId("legend-learned")).toHaveText("LEARNED policy");
  await expect(page.getByTestId("legend-user")).toHaveText("USER teleoperation");
  await expect(page.getByTestId("legend-debug")).toHaveText("DEBUG — excluded from evaluation");
  await expect(legend.getByTestId("legend-learned")).toHaveCSS("background-color", "rgb(59, 130, 246)");
  await expect(legend.getByTestId("legend-scripted_teacher")).toHaveCSS("background-color", "rgb(245, 158, 11)");
  // teacher mode → amber badge; recorded steps are labeled with their source
  await page.getByLabel("control mode", { exact: true }).selectOption("scripted_teacher");
  await page.getByRole("button", { name: "Apply mode" }).click();
  const badge = page.getByTestId("mode-badge");
  await expect(badge).toContainText("SCRIPTED TEACHER (privileged)");
  await expect(badge).toHaveCSS("background-color", "rgb(245, 158, 11)");
  await step(page, 10);
  await expect(page.getByTestId("count-scripted_teacher")).toContainText("SCRIPTED TEACHER (privileged): 10");
  // learned mode without a registered policy is refused by the backend (no silent fallback)
  await page.getByLabel("control mode", { exact: true }).selectOption("learned");
  await page.getByRole("button", { name: "Apply mode" }).click();
  await expect(alertWithCode(page, "unknown_policy")).toBeVisible();
  // user teleoperation → green
  await page.getByLabel("gripper[0] value", { exact: true }).fill("0.02");
  await page.getByRole("button", { name: "Send gripper joint target" }).click();
  await expect(badge).toContainText("USER teleoperation");
  await expect(badge).toHaveCSS("background-color", "rgb(34, 197, 94)");
  await step(page, 3);
  await expect(page.getByTestId("count-user")).toContainText("USER teleoperation: 3");
  // debug override → red, confirmed, contaminated
  await page.getByLabel("control mode", { exact: true }).selectOption("hold");
  await page.getByRole("button", { name: "Apply mode" }).click();
  await expect(badge).toHaveAttribute("data-mode", "hold");
  await selectEvent(page, "grasp");
  await page.getByRole("button", { name: "DEBUG force success…" }).click();
  await expect(badge).toContainText("DEBUG — excluded from evaluation");
  await expect(badge).toHaveCSS("background-color", "rgb(239, 68, 68)");
  await expect(page.getByTestId("contaminated-badge")).toBeVisible();
});

test("teleport requires confirmation and marks the run contaminated; privileged overlay is labeled", async ({ page }) => {
  await open(page);
  await createSession(page);
  await expect(page.getByTestId("contaminated-badge")).toHaveCount(0);
  await page.getByLabel("Teleport object (contaminates evaluation)", { exact: true }).check();
  await page.getByLabel("teleport object", { exact: true }).selectOption("cube");
  await page.getByLabel("teleport position x", { exact: true }).fill("0.42");
  await page.getByLabel("teleport position y", { exact: true }).fill("0.1");
  await page.getByLabel("teleport position z", { exact: true }).fill("0.05");
  let confirmText = "";
  page.removeAllListeners("dialog");
  page.on("dialog", (d) => { confirmText = d.message(); void d.accept(); });
  await page.getByRole("button", { name: "Teleport…" }).click();
  await expect(page.getByTestId("contaminated-badge")).toBeVisible();
  expect(confirmText).toContain("CONTAMINATED");
  await tab(page, "Privileged");
  await page.getByLabel("Show privileged simulator truth overlay (display only)", { exact: true }).check();
  await expect(page.getByTestId("priv-banner")).toContainText("PRIVILEGED SIMULATOR TRUTH (display only; never a policy input)");
  await tab(page, "Observation");
  await expect(page.getByTestId("observation")).toContainText("PUBLIC ESTIMATES");
});

test("dragging a graph node changes layout only", async ({ page }) => {
  await open(page);
  const sid = await createSession(page);
  const v0 = await graphVersion(page);
  const node = page.getByTestId("event-node-place");
  const bb = (await node.boundingBox())!;
  await page.mouse.move(bb.x + 40, bb.y + 12);
  await page.mouse.down();
  await page.mouse.move(bb.x + 90, bb.y + 90, { steps: 8 });
  await page.mouse.up();
  await expect.poll(async () => {
    const g = await (await page.request.get(`${BASE}/api/sessions/${sid}/graph`)).json();
    return g.graph.layout?.place ? 1 : 0;
  }).toBe(1);
  expect(await graphVersion(page)).toBe(v0);
  const g = await (await page.request.get(`${BASE}/api/sessions/${sid}/graph`)).json();
  expect(g.graph.version).toBe(v0);
});

test("low-load stream mode shows server-rendered frames", async ({ page }) => {
  await open(page, "&lowload=1");
  await createSession(page);
  const img = page.getByTestId("stream-view").locator("img");
  await expect.poll(() => img.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth), { timeout: 20_000 }).toBeGreaterThan(0);
  await expect(page.getByTestId("scene-canvas")).toHaveCount(0);
});

test("demo tour: teacher executes pick-and-place while graph and inspector follow", async ({ page }) => {
  await open(page);
  await createSession(page);
  await page.getByLabel("control mode", { exact: true }).selectOption("scripted_teacher");
  await page.getByRole("button", { name: "Apply mode" }).click();
  await expect(page.getByTestId("mode-badge")).toContainText("SCRIPTED TEACHER (privileged)");
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect.poll(async () => page.getByTestId("status-grasp").innerText(), { timeout: 60_000 }).toMatch(/active|succeeded/);
  await page.waitForTimeout(4000);
  await tab(page, "Probes");
  await page.getByRole("button", { name: "Run all (no args)" }).click();
  await page.waitForTimeout(3000);
  await tab(page, "Routing");
  await page.waitForTimeout(3000);
  if (await page.getByRole("button", { name: "Pause" }).isVisible()) await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByTestId("count-scripted_teacher")).toBeVisible();
});
