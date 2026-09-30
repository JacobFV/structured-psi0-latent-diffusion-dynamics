import { useWB } from "./store";

export function Toasts() {
  const { toasts, dismissToast } = useWB();
  return (
    <div className="toasts" aria-live="assertive">
      {toasts.map((t) => (
        <div key={t.id} role="alert" className={`toast toast-${t.level}`} data-code={t.code}>
          <div className="toast-title"><strong>{t.title}</strong>{t.code && <code className="toast-code">{t.code}</code>}</div>
          {t.message && <div className="toast-msg">{t.message}</div>}
          <div className="toast-actions">
            {t.action && (
              <button onClick={() => { t.action!.fn(); dismissToast(t.id); }}>{t.action.label}</button>
            )}
            <button aria-label="Dismiss notification" onClick={() => dismissToast(t.id)}>Dismiss</button>
          </div>
        </div>
      ))}
    </div>
  );
}
