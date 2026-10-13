"use client";

import { AlertTriangle, CheckCircle2, ExternalLink, KeyRound, Loader2, Pencil, Plug, Plus, ShieldAlert, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useToast } from "@/components/Toaster";
import { api, isLocalUrl, type LlmKind, type LlmPreset, type LlmProfile, type LlmSettings, type LlmTestResult } from "@/lib/api";

const KIND_LABEL: Record<LlmKind, string> = { ollama: "Ollama", "openai-compatible": "Compatible con OpenAI" };

interface Draft {
  id: string | null; // null = alta nueva
  presetId: string;
  name: string;
  kind: LlmKind;
  base_url: string;
  model: string;
  api_key: string;
  clear_api_key: boolean;
  allow_external: boolean;
  hasKey: boolean;
  keyHint: string | null;
}

const emptyDraft = (p?: LlmPreset): Draft => ({
  id: null, presetId: p?.id ?? "", name: p && p.id !== "custom-openai" && p.id !== "custom-ollama" ? p.label : "", kind: p?.kind ?? "openai-compatible",
  base_url: p?.base_url ?? "", model: "", api_key: "", clear_api_key: false, allow_external: false, hasKey: false, keyHint: null,
});

function host(url: string) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export default function LlmSettingsPanel() {
  const toast = useToast();
  const [data, setData] = useState<LlmSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [probe, setProbe] = useState<(LlmTestResult & { forId: string }) | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setData(await api.llmSettings());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
      toast.error("No se pudo cargar la configuración", { id: "llm-load", description: (e as Error).message });
    }
  }, [toast]);
  useEffect(() => {
    void load();
  }, [load]);

  async function run<T>(
    key: string,
    fn: () => Promise<T>,
    ok?: { title: string; description?: string },
    failTitle = "No se pudo completar la acción",
  ): Promise<T | undefined> {
    setBusy(key);
    try {
      const r = await fn();
      if (ok) toast.success(ok.title, { description: ok.description, id: `llm-${key}` });
      return r;
    } catch (e) {
      toast.error(failTitle, { id: `llm-${key}-err`, description: (e as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function activate(p: LlmProfile) {
    const r = await run(`act-${p.id}`, () => api.setActiveLlm(p.id),
      { title: `«${p.name}» es ahora el predeterminado`, description: `${p.model} · se usará en la siguiente pregunta` }, "No se pudo activar el proveedor");
    if (r) setData(r);
  }

  async function test(p: LlmProfile) {
    const r = await run(`test-${p.id}`, () => api.testLlm({ profile_id: p.id }), undefined, "No se pudo probar la conexión");
    if (r) setProbe({ ...r, forId: p.id });
  }

  function edit(p: LlmProfile) {
    setProbe(null);
    setModels([]);
    setDraft({ id: p.id, presetId: "", name: p.name, kind: p.kind, base_url: p.base_url, model: p.model, api_key: "", clear_api_key: false,
      allow_external: p.allow_external, hasKey: p.has_key, keyHint: p.key_hint });
  }

  async function loadModels() {
    if (!draft) return;
    const r = await run("models", () => api.testLlm({ profile_id: draft.id, kind: draft.kind, base_url: draft.base_url, api_key: draft.api_key || null }), undefined, "No se pudieron cargar los modelos");
    if (!r) return;
    setProbe({ ...r, forId: "draft" });
    setModels(r.models);
  }

  async function save(andActivate: boolean) {
    if (!draft) return;
    const external = !isLocalUrl(draft.base_url);
    const body = { name: draft.name, kind: draft.kind, base_url: draft.base_url, model: draft.model, allow_external: external ? draft.allow_external : false };
    const saved = await run("save", async () => {
      let view: LlmSettings;
      if (draft.id) {
        view = await api.updateLlmProfile(draft.id, { ...body, api_key: draft.api_key || undefined, clear_api_key: draft.clear_api_key });
      } else {
        view = await api.createLlmProfile({ ...body, api_key: draft.api_key || undefined });
      }
      if (andActivate) {
        const id = draft.id ?? view.profiles.find((p) => p.name === draft.name.trim())!.id;
        view = await api.setActiveLlm(id);
      }
      return view;
    }, andActivate
      ? { title: `«${draft.name.trim()}» guardado y activado`, description: "Se usará en la siguiente pregunta." }
      : { title: `«${draft.name.trim()}» guardado` }, "No se pudo guardar el proveedor");
    if (saved) {
      setData(saved);
      setDraft(null);
      setModels([]);
      setProbe(null);
    }
  }

  if (!data) {
    return error ? (
      <p className="text-sm text-bad">{error}</p>
    ) : (
      <div className="flex flex-col gap-2" aria-busy="true" aria-label="Cargando proveedores">
        <div className="skeleton h-20 w-full" />
        <div className="skeleton h-9 w-40" />
      </div>
    );
  }

  const preset = data.presets.find((p) => p.id === draft?.presetId);
  const external = draft ? !isLocalUrl(draft.base_url) : false;
  const input = "w-full rounded-md border border-line bg-white px-2.5 py-1.5 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-petrol";

  return (
    <div className="flex flex-col gap-4">

      <ul className="flex flex-col gap-2">
        {data.profiles.map((p) => (
          <li key={p.id} className={`rounded-lg border bg-white p-3 ${p.active ? "border-petrol ring-1 ring-petrol" : "border-line"}`}>
            <div className="flex flex-wrap items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{p.name}</span>
                  {p.active && <span className="badge bg-petrol text-white">Predeterminado</span>}
                  <span className={`badge ${p.external ? "bg-amber-50 text-warn" : "bg-emerald-50 text-ok"}`}>
                    {p.external ? "Servicio externo" : "Local"}
                  </span>
                  {!p.editable && <span className="badge bg-panel text-ink-soft">Definido en .env</span>}
                </div>
                <div className="mt-1 text-xs text-ink-soft">
                  {KIND_LABEL[p.kind]} · {host(p.base_url) || "—"} · modelo <span className="font-medium text-ink">{p.model}</span>
                </div>
                <div className="mt-0.5 flex items-center gap-1 text-xs text-ink-faint">
                  <KeyRound className="h-3 w-3" aria-hidden />
                  {p.key_error ? (
                    <span className="text-bad">Clave ilegible: vuelva a introducirla</span>
                  ) : p.has_key ? (
                    <span>Clave guardada{p.key_hint ? ` (${p.key_hint})` : ""}</span>
                  ) : (
                    <span>Sin clave</span>
                  )}
                  {p.external && (
                    <span className={`ml-2 ${p.allow_external ? "text-ok" : "text-warn"}`}>
                      · {p.allow_external ? "envío de fragmentos permitido" : "envío de fragmentos NO permitido"}
                    </span>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {!p.active && (
                  <button className="btn-primary" disabled={!!busy} onClick={() => void activate(p)}>
                    {busy === `act-${p.id}` && <Loader2 className="h-4 w-4 animate-spin" />} Usar por defecto
                  </button>
                )}
                <button className="btn" disabled={!!busy} onClick={() => void test(p)}>
                  {busy === `test-${p.id}` ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plug className="h-4 w-4" />} Probar
                </button>
                {p.editable && (
                  <>
                    <button className="btn" onClick={() => edit(p)} aria-label={`Editar ${p.name}`}>
                      <Pencil className="h-4 w-4" /> Editar
                    </button>
                    <button
                      className="btn-icon"
                      aria-label={`Eliminar ${p.name}`}
                      onClick={async () => {
                        const ok = await toast.confirm({
                          title: "¿Eliminar este proveedor?",
                          message: `Se borrarán «${p.name}» y su clave API guardada${p.active ? ". Es el predeterminado: se volverá a Ollama (entorno)" : ""}. No se puede deshacer.`,
                          confirmLabel: "Eliminar proveedor",
                          tone: "danger",
                        });
                        if (!ok) return;
                        const r = await run("del", () => api.deleteLlmProfile(p.id), { title: `«${p.name}» eliminado` }, "No se pudo eliminar el proveedor");
                        if (r) setData(r);
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </>
                )}
              </div>
            </div>
            {probe?.forId === p.id && (
              <p className={`mt-2 flex items-start gap-1.5 text-xs ${probe.ok ? "text-ok" : "text-bad"}`} role="status">
                {probe.ok ? <CheckCircle2 className="mt-px h-3.5 w-3.5 shrink-0" /> : <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" />}
                {probe.detail}
                {probe.ok && !probe.models.includes(p.model) && <span className="text-warn"> · el modelo «{p.model}» no aparece en la lista del proveedor</span>}
              </p>
            )}
          </li>
        ))}
      </ul>

      {!draft ? (
        <div>
          <button className="btn" onClick={() => { setDraft(emptyDraft(data.presets[1])); setModels([]); setProbe(null); }}>
            <Plus className="h-4 w-4" /> Añadir proveedor
          </button>
        </div>
      ) : (
        <form
          className="rounded-lg border border-line bg-panel p-4"
          onSubmit={(e) => {
            e.preventDefault();
            void save(false);
          }}
        >
          <h3 className="mb-3 text-sm font-semibold">{draft.id ? "Editar proveedor" : "Nuevo proveedor"}</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {!draft.id && (
              <div className="sm:col-span-2">
                <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="preset">Plantilla</label>
                <select id="preset" className={input} value={draft.presetId}
                  onChange={(e) => { const p = data.presets.find((x) => x.id === e.target.value); setDraft(emptyDraft(p)); setModels([]); setProbe(null); }}>
                  {data.presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                </select>
                {preset && (
                  <p className="mt-1 text-xs text-ink-soft">
                    {preset.note}{" "}
                    {preset.key_url && (
                      <a className="inline-flex items-center gap-0.5 text-petrol underline" href={preset.key_url} target="_blank" rel="noopener noreferrer">
                        Obtener clave <ExternalLink className="h-3 w-3" aria-hidden />
                      </a>
                    )}
                  </p>
                )}
              </div>
            )}
            <div>
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="pname">Nombre</label>
              <input id="pname" className={input} value={draft.name} required maxLength={120} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="pkind">Tipo de API</label>
              <select id="pkind" className={input} value={draft.kind} onChange={(e) => setDraft({ ...draft, kind: e.target.value as LlmKind })}>
                <option value="ollama">Ollama</option>
                <option value="openai-compatible">Compatible con OpenAI (Groq, Gemini, OpenAI, LM Studio…)</option>
              </select>
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="purl">URL base de la API</label>
              <input id="purl" className={input} value={draft.base_url} required placeholder="https://…" onChange={(e) => setDraft({ ...draft, base_url: e.target.value })} />
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="pkey">Clave API</label>
              <input id="pkey" type="password" autoComplete="off" spellCheck={false} className={input} value={draft.api_key}
                disabled={draft.clear_api_key}
                placeholder={draft.hasKey ? `Clave guardada${draft.keyHint ? ` (${draft.keyHint})` : ""} — déjelo vacío para conservarla` : "Opcional para servicios locales"}
                onChange={(e) => setDraft({ ...draft, api_key: e.target.value })} />
              <p className="mt-1 text-xs text-ink-faint">Se guarda cifrada en el servidor y no vuelve a mostrarse en la interfaz.</p>
              {draft.id && draft.hasKey && (
                <label className="mt-1 flex items-center gap-2 text-xs text-ink-soft">
                  <input type="checkbox" className="accent-petrol" checked={draft.clear_api_key} onChange={(e) => setDraft({ ...draft, clear_api_key: e.target.checked, api_key: "" })} />
                  Eliminar la clave guardada
                </label>
              )}
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="pmodel">Modelo</label>
              <div className="flex gap-2">
                <input id="pmodel" className={input} list="llm-models" value={draft.model} required placeholder="Nombre exacto del modelo en ese proveedor"
                  onChange={(e) => setDraft({ ...draft, model: e.target.value })} />
                <datalist id="llm-models">{models.map((m) => <option key={m} value={m} />)}</datalist>
                <button type="button" className="btn shrink-0" disabled={!draft.base_url || !!busy} onClick={() => void loadModels()}>
                  {busy === "models" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plug className="h-4 w-4" />} Cargar modelos
                </button>
              </div>
              {probe?.forId === "draft" && (
                <p className={`mt-1 text-xs ${probe.ok ? "text-ok" : "text-bad"}`} role="status">
                  {probe.detail}
                </p>
              )}
            </div>
          </div>

          {external && (
            <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs text-warn">
              <p className="flex items-start gap-1.5 font-medium">
                <ShieldAlert className="mt-px h-4 w-4 shrink-0" aria-hidden />
                {host(draft.base_url)} es un servicio externo
              </p>
              <p className="mt-1">
                Al preguntar, se enviará a ese servicio <strong>su pregunta y los fragmentos de sus documentos</strong> que resulten relevantes (no el PDF completo).
                Revise la política de retención y de uso de datos del proveedor antes de usarlo con información sensible.
              </p>
              <label className="mt-2 flex items-start gap-2 text-ink">
                <input type="checkbox" className="mt-0.5 accent-petrol" checked={draft.allow_external} onChange={(e) => setDraft({ ...draft, allow_external: e.target.checked })} />
                <span>Permitir el envío de fragmentos a este servicio externo</span>
              </label>
            </div>
          )}

          <div className="mt-4 flex flex-wrap gap-2">
            <button type="submit" className="btn-primary" disabled={!!busy}>
              {busy === "save" && <Loader2 className="h-4 w-4 animate-spin" />} Guardar
            </button>
            <button type="button" className="btn" disabled={!!busy || (external && !draft.allow_external)} onClick={() => void save(true)}
              title={external && !draft.allow_external ? "Marque el permiso de envío para poder activarlo" : undefined}>
              Guardar y usar por defecto
            </button>
            <button type="button" className="btn" onClick={() => { setDraft(null); setProbe(null); }}>Cancelar</button>
          </div>
        </form>
      )}
    </div>
  );
}
