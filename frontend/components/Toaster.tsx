"use client";

/**
 * Sistema de notificaciones flotantes + diálogo de confirmación.
 *
 *   const t = useToast();
 *   t.success("Documento subido", { description: "informe.pdf" });
 *   t.error("No se pudo guardar", { description: err.message });
 *   t.info("Conversación eliminada", { action: { label: "Deshacer", onClick: undo }, onDismiss: (r) => r !== "action" && commit() });
 *   t.promise(upload(), { loading: "Subiendo…", success: "Subido", error: (e) => e.message });
 *   if (await t.confirm({ title: "¿Eliminar?", tone: "danger", confirmLabel: "Eliminar" })) …
 *
 * Detalles: barra de progreso con la cuenta atrás (se pausa al pasar el ratón o enfocar), cierre con botón, apilado con tope,
 * `id` para reemplazar una notificación existente, accesibilidad (región aria-live; los errores usan role="alert"),
 * respeto de prefers-reduced-motion (ver globals.css).
 */

import { AlertTriangle, CheckCircle2, Info, Loader2, X, XCircle } from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

export type ToastKind = "success" | "error" | "info" | "warning" | "loading";
export type DismissReason = "timeout" | "close" | "action" | "replaced" | "api";

export interface ToastOptions {
  id?: string;
  description?: string;
  /** ms visibles; 0 = permanece hasta cerrarse. Por defecto según el tipo. */
  duration?: number;
  action?: { label: string; onClick: () => void };
  /** Se invoca una sola vez al desaparecer, con el motivo (útil para acciones diferidas con «Deshacer»). */
  onDismiss?: (reason: DismissReason) => void;
}

export interface ConfirmOptions {
  title: string;
  message?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  tone?: "danger" | "default";
}

interface Item extends ToastOptions {
  id: string;
  kind: ToastKind;
  title: string;
  duration: number;
  rev: number; // cambia al actualizar: reinicia la barra de progreso
  leaving: boolean;
}

interface PromiseMessages<T> {
  loading: string;
  success: string | ((value: T) => string | { title: string; description?: string; kind?: ToastKind });
  error: string | ((error: Error) => string | { title: string; description?: string });
}

interface ToastApi {
  show: (kind: ToastKind, title: string, opts?: ToastOptions) => string;
  success: (title: string, opts?: ToastOptions) => string;
  error: (title: string, opts?: ToastOptions) => string;
  info: (title: string, opts?: ToastOptions) => string;
  warning: (title: string, opts?: ToastOptions) => string;
  dismiss: (id: string, reason?: DismissReason) => void;
  promise: <T>(p: Promise<T>, msgs: PromiseMessages<T>, opts?: ToastOptions) => Promise<T>;
  confirm: (opts: ConfirmOptions) => Promise<boolean>;
}

const DEFAULT_DURATION: Record<ToastKind, number> = { success: 4000, info: 4500, warning: 7000, error: 9000, loading: 0 };
const MAX_VISIBLE = 5;
const EXIT_MS = 180;

const Ctx = createContext<ToastApi | null>(null);

export function useToast(): ToastApi {
  const c = useContext(Ctx);
  if (!c) throw new Error("useToast debe usarse dentro de <ToastProvider>");
  return c;
}

const STYLE: Record<ToastKind, { icon: React.ReactNode; accent: string; bar: string }> = {
  success: { icon: <CheckCircle2 className="h-5 w-5 text-ok" aria-hidden />, accent: "border-l-ok", bar: "bg-ok" },
  error: { icon: <XCircle className="h-5 w-5 text-bad" aria-hidden />, accent: "border-l-bad", bar: "bg-bad" },
  warning: { icon: <AlertTriangle className="h-5 w-5 text-warn" aria-hidden />, accent: "border-l-warn", bar: "bg-warn" },
  info: { icon: <Info className="h-5 w-5 text-petrol" aria-hidden />, accent: "border-l-petrol", bar: "bg-petrol" },
  loading: { icon: <Loader2 className="h-5 w-5 animate-spin text-petrol" aria-hidden />, accent: "border-l-petrol", bar: "bg-petrol" },
};

let counter = 0;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<Item[]>([]);
  const itemsRef = useRef<Item[]>([]);
  itemsRef.current = items;
  const finished = useRef(new Set<string>()); // ids cuyo onDismiss ya se invocó
  const [confirmState, setConfirmState] = useState<{ opts: ConfirmOptions; resolve: (v: boolean) => void } | null>(null);

  const remove = useCallback((id: string) => setItems((xs) => xs.filter((x) => x.id !== id)), []);

  const dismiss = useCallback(
    (id: string, reason: DismissReason = "api") => {
      const it = itemsRef.current.find((x) => x.id === id);
      if (!it || it.leaving) return;
      setItems((xs) => xs.map((x) => (x.id === id ? { ...x, leaving: true } : x)));
      if (!finished.current.has(`${id}:${it.rev}`)) {
        finished.current.add(`${id}:${it.rev}`);
        try {
          it.onDismiss?.(reason);
        } catch (e) {
          console.error("onDismiss falló", e);
        }
      }
      window.setTimeout(() => remove(id), EXIT_MS);
    },
    [remove],
  );

  const show = useCallback(
    (kind: ToastKind, title: string, opts: ToastOptions = {}): string => {
      const id = opts.id ?? `t${++counter}`;
      const existing = itemsRef.current.find((x) => x.id === id);
      if (existing && !existing.leaving) {
        // Reemplazo en el sitio (p. ej. «Subiendo…» → «Subido»). No cuenta como cierre de la anterior salvo que tenga onDismiss.
        if (existing.onDismiss && !finished.current.has(`${id}:${existing.rev}`)) {
          finished.current.add(`${id}:${existing.rev}`);
          existing.onDismiss("replaced");
        }
      }
      const item: Item = {
        ...opts,
        id,
        kind,
        title,
        duration: opts.duration ?? DEFAULT_DURATION[kind],
        rev: (existing?.rev ?? 0) + 1,
        leaving: false,
      };
      setItems((xs) => (existing && !existing.leaving ? xs.map((x) => (x.id === id ? item : x)) : [...xs.filter((x) => x.id !== id), item]));
      return id;
    },
    [dismiss],
  );

  // Tope de notificaciones visibles: se retiran primero las más antiguas.
  useEffect(() => {
    const live = items.filter((i) => !i.leaving);
    if (live.length > MAX_VISIBLE) live.slice(0, live.length - MAX_VISIBLE).forEach((i) => dismiss(i.id, "replaced"));
  }, [items, dismiss]);

  const api = useMemo<ToastApi>(() => {
    const promise: ToastApi["promise"] = async (p, msgs, opts) => {
      const id = show("loading", msgs.loading, { ...opts, id: opts?.id });
      try {
        const v = await p;
        const r = typeof msgs.success === "function" ? msgs.success(v) : msgs.success;
        const o = typeof r === "string" ? { title: r } : r;
        show(("kind" in o && o.kind) || "success", o.title, { id, description: o.description });
        return v;
      } catch (e) {
        const err = e instanceof Error ? e : new Error(String(e));
        const r = typeof msgs.error === "function" ? msgs.error(err) : msgs.error;
        const o = typeof r === "string" ? { title: r, description: undefined } : r;
        show("error", o.title, { id, description: o.description });
        throw e;
      }
    };
    return {
      show,
      success: (t, o) => show("success", t, o),
      error: (t, o) => show("error", t, o),
      info: (t, o) => show("info", t, o),
      warning: (t, o) => show("warning", t, o),
      dismiss,
      promise,
      confirm: (opts) => new Promise<boolean>((resolve) => setConfirmState({ opts, resolve })),
    };
  }, [show, dismiss]);

  // Al cerrar o recargar la página, las acciones diferidas pendientes se confirman (ver Chat: eliminación con «Deshacer»).
  useEffect(() => {
    const flush = () => itemsRef.current.forEach((it) => !it.leaving && it.onDismiss && dismiss(it.id, "api"));
    window.addEventListener("pagehide", flush);
    return () => window.removeEventListener("pagehide", flush);
  }, [dismiss]);

  return (
    <Ctx.Provider value={api}>
      {children}
      <ToastViewport items={items} dismiss={dismiss} />
      {confirmState && (
        <ConfirmDialog
          opts={confirmState.opts}
          onClose={(v) => {
            confirmState.resolve(v);
            setConfirmState(null);
          }}
        />
      )}
    </Ctx.Provider>
  );
}

// ------------------------------------------------------------------------------------------------ viewport
function ToastViewport({ items, dismiss }: { items: Item[]; dismiss: (id: string, r?: DismissReason) => void }) {
  return (
    <div
      role="region"
      aria-label="Notificaciones"
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 bottom-0 z-[100] flex flex-col items-end gap-2 p-4 sm:inset-x-auto sm:right-0"
    >
      {items.map((it) => (
        <ToastCard key={it.id} item={it} dismiss={dismiss} />
      ))}
    </div>
  );
}

function ToastCard({ item, dismiss }: { item: Item; dismiss: (id: string, r?: DismissReason) => void }) {
  const [paused, setPaused] = useState(false);
  const st = STYLE[item.kind];
  const timed = item.duration > 0;
  return (
    <div
      role={item.kind === "error" ? "alert" : "status"}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocusCapture={() => setPaused(true)}
      onBlurCapture={() => setPaused(false)}
      className={`${item.leaving ? "animate-toast-out" : "animate-toast-in"} pointer-events-auto relative w-full max-w-[22.5rem] overflow-hidden rounded-lg border border-line border-l-4 bg-white shadow-lg ring-1 ring-black/5 sm:w-[22.5rem] ${st.accent}`}
    >
      <div className="flex items-start gap-3 px-3.5 py-3">
        <span className="mt-0.5 shrink-0">{st.icon}</span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold leading-snug text-ink">{item.title}</p>
          {item.description && <p className="mt-0.5 break-words text-xs leading-relaxed text-ink-soft">{item.description}</p>}
          {item.action && (
            <button
              className="mt-2 rounded px-1 py-0.5 text-xs font-semibold text-petrol underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-petrol"
              onClick={() => {
                item.action!.onClick();
                dismiss(item.id, "action");
              }}
            >
              {item.action.label}
            </button>
          )}
        </div>
        <button
          className="btn-icon -mr-1 -mt-1 h-6 w-6 shrink-0"
          aria-label="Cerrar notificación"
          onClick={() => dismiss(item.id, "close")}
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {timed && (
        <div className="h-0.5 w-full bg-line/60" aria-hidden>
          <div
            key={item.rev}
            className={`h-full origin-left animate-toast-progress ${st.bar}`}
            style={{ animationDuration: `${item.duration}ms`, animationPlayState: paused || item.leaving ? "paused" : "running" }}
            onAnimationEnd={() => dismiss(item.id, "timeout")}
          />
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ confirmación
function ConfirmDialog({ opts, onClose }: { opts: ConfirmOptions; onClose: (v: boolean) => void }) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const danger = opts.tone === "danger";

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    cancelRef.current?.focus(); // en acciones destructivas, el foco inicial es «Cancelar»
    return () => previous?.focus?.();
  }, []);

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.stopPropagation();
      onClose(false);
    } else if (e.key === "Tab") {
      // Trampa de foco entre los dos botones
      const order = [cancelRef.current, confirmRef.current].filter(Boolean) as HTMLElement[];
      const i = order.indexOf(document.activeElement as HTMLElement);
      const next = e.shiftKey ? order[(i - 1 + order.length) % order.length] : order[(i + 1) % order.length];
      e.preventDefault();
      next.focus();
    }
  }

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center p-4" onKeyDown={onKeyDown}>
      <div className="absolute inset-0 animate-fade-in bg-ink/40 backdrop-blur-[1px]" onClick={() => onClose(false)} aria-hidden />
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby={opts.message ? "confirm-desc" : undefined}
        className="relative w-full max-w-sm animate-pop-in rounded-xl border border-line bg-white p-5 shadow-xl"
      >
        <div className="flex items-start gap-3">
          <span className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${danger ? "bg-red-50 text-bad" : "bg-petrol-tint text-petrol"}`}>
            {danger ? <AlertTriangle className="h-5 w-5" aria-hidden /> : <Info className="h-5 w-5" aria-hidden />}
          </span>
          <div className="min-w-0">
            <h2 id="confirm-title" className="text-base font-semibold text-ink">
              {opts.title}
            </h2>
            {opts.message && (
              <p id="confirm-desc" className="mt-1 text-sm leading-relaxed text-ink-soft">
                {opts.message}
              </p>
            )}
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <button ref={cancelRef} className="btn" onClick={() => onClose(false)}>
            {opts.cancelLabel ?? "Cancelar"}
          </button>
          <button
            ref={confirmRef}
            onClick={() => onClose(true)}
            className={
              danger
                ? "inline-flex items-center justify-center gap-1.5 rounded-md border border-bad bg-bad px-3 py-1.5 text-sm text-white hover:bg-red-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-bad"
                : "btn-primary"
            }
          >
            {opts.confirmLabel ?? "Aceptar"}
          </button>
        </div>
      </div>
    </div>
  );
}
