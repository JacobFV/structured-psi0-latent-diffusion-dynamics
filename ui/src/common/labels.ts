// Stable colors identify control/data provenance, never research performance.
export interface SourceStyle { key: string; label: string; color: string; fg: string }

export const SOURCE_STYLES: Record<string, SourceStyle> = {
  scripted_teacher: { key: "scripted_teacher", label: "SCRIPTED TEACHER (privileged)", color: "#f59e0b", fg: "#1a1200" },
  learned: { key: "learned", label: "LEARNED policy", color: "#3b82f6", fg: "#ffffff" },
  user: { key: "user", label: "USER teleoperation", color: "#22c55e", fg: "#032b12" },
  debug: { key: "debug", label: "DEBUG — excluded from evaluation", color: "#ef4444", fg: "#ffffff" },
  hold: { key: "hold", label: "HOLD (no controller input)", color: "#6b7280", fg: "#ffffff" },
  privileged: { key: "privileged", label: "PRIVILEGED TRUTH (display only)", color: "#d946ef", fg: "#ffffff" },
  system: { key: "system", label: "system", color: "#94a3b8", fg: "#0f172a" },
};

export function sourceStyle(source: string | null | undefined): SourceStyle {
  const s = source ?? "system";
  if (s.startsWith("learned")) {
    const name = s.includes(":") ? s.split(":").slice(1).join(":") : "";
    return { ...SOURCE_STYLES.learned, label: name ? `LEARNED policy ${name}` : SOURCE_STYLES.learned.label };
  }
  return SOURCE_STYLES[s] ?? SOURCE_STYLES.system;
}

export const STATUS_COLORS: Record<string, string> = {
  pending: "#64748b", ready: "#0ea5e9", active: "#f59e0b", succeeded: "#16a34a", failed: "#dc2626",
  cancelled: "#71717a", blocked: "#a855f7",
};
export const statusColor = (s: string | undefined) => STATUS_COLORS[s ?? ""] ?? "#94a3b8";

export function fmt(v: unknown, digits = 3): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(digits) : String(v);
  if (Array.isArray(v)) return `[${v.map((x) => fmt(x, digits)).join(", ")}]`;
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
