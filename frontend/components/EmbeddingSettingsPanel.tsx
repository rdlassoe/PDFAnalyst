"use client";

import { AlertTriangle, CheckCircle2, Database, ExternalLink, KeyRound, Loader2, Pencil, Plug, Plus, ShieldAlert, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useToast } from "@/components/Toaster";
import { api, isLocalUrl, type EmbedKind, type EmbedPreset, type EmbedProfile, type EmbedSettings, type EmbedTestResult } from "@/lib/api";

const KIND_LABEL: Record<EmbedKind, string> = {
  "sentence-transformers": "Modelo local (en proceso)",
  ollama: "Ollama",
  "openai-compatible": "Compatible con OpenAI",
};

interface Draft {
  id: string | null; // null = alta nueva
  active: boolean;
  presetId: string;
  name: string;
  kind: "ollama" | "openai-compatible";
  base_url: string;
  model: string;
  api_key: string;
  clear_api_key: boolean;
  allow_external: boolean;
  hasKey: boolean;
  keyHint: string | null;
  query_prefix: string;
  passage_prefix: string;
  dimensions: string;
  min_similarity: string;
}

const fromPreset = (p?: EmbedPreset): Draft => ({
  id: null, active: false, presetId: p?.id ?? "", name: p && !p.id.startsWith("custom") ? p.label : "", kind: p?.kind ?? "openai-compatible",
  base_url: p?.base_url ?? "", model: p?.model ?? "", api_key: "", clear_api_key: false, allow_external: false, hasKey: false, keyHint: null,
  query_prefix: p?.query_prefix ?? "", passage_prefix: p?.passage_prefix ?? "", dimensions: p?.dimensions ? String(p.dimensions) : "",
  min_similarity: p?.min_similarity != null ? String(p.min_similarity) : "",
});

function host(url: string) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

const fmtChars = (n: number) => (n > 1_000_000 ? `${(n / 1_000_000).toFixed(1)} M de caracteres` : `${Math.max(1, Math.round(n / 1000))} mil caracteres`);

export default function EmbeddingSettingsPanel() {
  const toast = useToast();
  const [data, setData] = useState<EmbedSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [probe, setProbe] = useState<(EmbedTestResult & { forId: string }) | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setData(await api.embedSettings());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
      toast.error("No se pudo cargar la configuración de embeddings", { id: "emb-load", description: (e as Error).message });
    }
  }, [toast]);
  useEffect(() => {
    void load();
  }, [load]);

  async function run<T>(key: string, fn: () => Promise<T>, ok?: { title: string; description?: string }, failTitle = "No se pudo completar la acción") {
    setBusy(key);
    try {
      const r = await fn();
      if (ok) toast.success(ok.title, { id: `emb-${key}`, description: ok.description });
      return r;
    } catch (e) {
      toast.error(failTitle, { id: `emb-${key}-err`, description: (e as Error).message });
    } finally {
      setBusy(null);
    }
  }

  /** Vista previa → confirmación informada (qué se reindexa y, si es externo, qué texto se envía) → activación. */
  async function startActivation(p: EmbedProfile) {
    if (p.external && !p.allow_external) {
      toast.error("Falta el permiso de envío", {
        id: "emb-consent",
        description: "Este servicio es externo. Edite el proveedor y marque «Permitir el envío del texto de los documentos» antes de activarlo.",
      });
      return;
    }
    const pv = await run(`prev-${p.id}`, () => api.previewEmbed(p.id), undefined, "No se pudo calcular la vista previa");
    if (!pv) return;
    const parts: string[] = [];
    parts.push(
      pv.documents_to_index === 0
        ? "Todos los documentos ya están indexados con este modelo: el cambio es inmediato."
        : `Se reindexarán ${pv.documents_to_index} de ${pv.documents_total} documentos (${pv.chunks_to_index} fragmentos). Mientras tanto no se podrán consultar.`,
    );
    if (pv.external && pv.documents_to_index > 0) {
      parts.push(`Se enviará a ${pv.host} el texto íntegro de esos documentos (≈${fmtChars(pv.chars_to_send)}).`);
    }
    if (pv.external) parts.push(`A partir de ahora, cada pregunta que haga también se enviará a ${pv.host} para buscar.`);
    else parts.push("Todo ocurre en su red: no sale texto del equipo.");
    const ok = await toast.confirm({
      title: `¿Usar «${pv.name}» como modelo de embeddings?`,
      message: parts.join(" "),
      confirmLabel: pv.documents_to_index > 0 ? "Cambiar y reindexar" : "Cambiar",
      tone: pv.external ? "danger" : "default",
    });
    if (!ok) return;
    const r = await run(`act-${p.id}`, () => api.activateEmbed(p.id), undefined, "No se pudo cambiar el modelo de embeddings");
    if (!r) return;
    setData(r);
    const a = r.activation;
    toast.success(`«${pv.name}» es el modelo de embeddings`, {
      id: "emb-activated",
      description:
        a.queued > 0 ? `Reindexando ${a.queued} documento${a.queued > 1 ? "s" : ""}… le avisaremos de cada uno al terminar.` : "Sus documentos ya estaban indexados con este modelo.",
      duration: 7000,
    });
  }

  async function test(p: EmbedProfile) {
    const r = await run(`test-${p.id}`, () => api.testEmbed({ profile_id: p.id }), undefined, "No se pudo probar la conexión");
    if (r) setProbe({ ...r, forId: p.id });
  }

  function edit(p: EmbedProfile) {
    setProbe(null);
    setDraft({
      id: p.id, active: p.active, presetId: "", name: p.name, kind: p.kind === "sentence-transformers" ? "openai-compatible" : p.kind,
      base_url: p.base_url, model: p.model, api_key: "", clear_api_key: false, allow_external: p.allow_external, hasKey: p.has_key,
      keyHint: p.key_hint, query_prefix: p.query_prefix, passage_prefix: p.passage_prefix, dimensions: p.dimensions ? String(p.dimensions) : "",
      min_similarity: p.min_similarity != null ? String(p.min_similarity) : "",
    });
  }

  function dims(d: Draft): number | null | "invalid" {
    if (!d.dimensions.trim()) return null;
    const n = Number(d.dimensions);
    return Number.isInteger(n) && n >= 8 && n <= 8192 ? n : "invalid";
  }

  function sim(d: Draft): number | null | "invalid" {
    if (!d.min_similarity.trim()) return null;
    const n = Number(d.min_similarity.replace(",", "."));
    return Number.isFinite(n) && n > 0 && n <= 1 ? n : "invalid";
  }

  async function testDraft() {
    if (!draft) return;
    const dm = dims(draft);
    if (dm === "invalid") return void toast.error("Dimensiones no válidas", { description: "Use un entero entre 8 y 8192 o déjelo vacío." });
    const r = await run("test-draft", () =>
      api.testEmbed({ profile_id: draft.id, kind: draft.kind, base_url: draft.base_url, model: draft.model, api_key: draft.api_key || null,
        query_prefix: draft.query_prefix, dimensions: dm }), undefined, "No se pudo probar la conexión");
    if (r) setProbe({ ...r, forId: "draft" });
  }

  async function save(andActivate: boolean) {
    if (!draft) return;
    const dm = dims(draft);
    if (dm === "invalid") return void toast.error("Dimensiones no válidas", { description: "Use un entero entre 8 y 8192 o déjelo vacío." });
    const sm = sim(draft);
    if (sm === "invalid") return void toast.error("Umbral no válido", { description: "Use un número entre 0 y 1 (p. ej. 0,72) o déjelo vacío." });
    const external = !isLocalUrl(draft.base_url);
    const body = {
      name: draft.name, kind: draft.kind, base_url: draft.base_url, model: draft.model, allow_external: external ? draft.allow_external : false,
      query_prefix: draft.query_prefix, passage_prefix: draft.passage_prefix, dimensions: dm, min_similarity: sm,
    };
    const view = await run("save", async () => {
      return draft.id
        ? await api.updateEmbedProfile(draft.id, { ...body, api_key: draft.api_key || undefined, clear_api_key: draft.clear_api_key })
        : await api.createEmbedProfile({ ...body, api_key: draft.api_key || undefined });
    }, { title: `«${draft.name.trim()}» guardado` }, "No se pudo guardar el proveedor");
    if (!view) return;
    setData(view);
    const saved = view.profiles.find((p) => (draft.id ? p.id === draft.id : p.name === draft.name.trim()));
    setDraft(null);
    setProbe(null);
    if (andActivate && saved) await startActivation(saved);
  }

  async function purge(id: number, name: string, vectors: number) {
    const ok = await toast.confirm({
      title: "¿Borrar los vectores de este modelo?",
      message: `Se eliminarán ${vectors} vectores de «${name}». Volver a usar ese modelo exigirá reindexar los documentos (y, si es un servicio externo, volver a enviar su texto).`,
      confirmLabel: "Borrar vectores",
      tone: "danger",
    });
    if (!ok) return;
    const r = await run(`purge-${id}`, () => api.purgeEmbedModel(id), { title: "Vectores eliminados", description: name }, "No se pudieron borrar los vectores");
    if (r) setData(r);
  }

  if (!data) {
    return error ? (
      <p className="text-sm text-bad">{error}</p>
    ) : (
      <div className="flex flex-col gap-2" aria-busy="true" aria-label="Cargando modelos de embeddings">
        <div className="skeleton h-20 w-full" />
        <div className="skeleton h-9 w-40" />
      </div>
    );
  }

  const preset = data.presets.find((p) => p.id === draft?.presetId);
  const external = draft ? !isLocalUrl(draft.base_url) : false;
  const locked = !!draft?.active; // un proveedor en uso no puede cambiar de modelo/prefijos sin invalidar sus vectores
  const input = "w-full rounded-md border border-line bg-white px-2.5 py-1.5 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-petrol disabled:bg-panel disabled:text-ink-faint";

  return (
    <div className="flex flex-col gap-4">
      <p className="rounded-md bg-panel px-3 py-2 text-xs text-ink-soft">
        Los documentos se indexan <strong>solo con el modelo seleccionado</strong>. Al cambiar de modelo se reindexan los documentos que aún no tengan
        sus vectores (no consultables mientras tanto); volver a un modelo ya indexado es inmediato. Indexados con el modelo activo:{" "}
        <strong>{data.coverage.documents_indexed} de {data.coverage.documents_total}</strong> documentos.
      </p>

      <ul className="flex flex-col gap-2">
        {data.profiles.map((p) => (
          <li key={p.id} className={`rounded-lg border bg-white p-3 ${p.active ? "border-petrol ring-1 ring-petrol" : "border-line"}`}>
            <div className="flex flex-wrap items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{p.name}</span>
                  {p.active && <span className="badge bg-petrol text-white">Predeterminado</span>}
                  <span className={`badge ${p.external ? "bg-amber-50 text-warn" : "bg-emerald-50 text-ok"}`}>{p.external ? "Servicio externo" : "Local"}</span>
                  {!p.editable && <span className="badge bg-panel text-ink-soft">Definido en .env</span>}
                </div>
                <div className="mt-1 text-xs text-ink-soft">
                  {KIND_LABEL[p.kind]}
                  {p.base_url && <> · {host(p.base_url)}</>} · modelo <span className="font-medium text-ink">{p.model}</span>
                  {p.dimensions ? <> · {p.dimensions} dim.</> : null}
                  {p.min_similarity != null ? <> · umbral {String(p.min_similarity).replace(".", ",")}</> : null}
                </div>
                {(p.query_prefix || p.passage_prefix) && (
                  <div className="mt-0.5 truncate text-[11px] text-ink-faint" title={`${p.query_prefix} / ${p.passage_prefix}`}>
                    Prefijos: «{p.query_prefix}» / «{p.passage_prefix}»
                  </div>
                )}
                {p.kind !== "sentence-transformers" && (
                  <div className="mt-0.5 flex items-center gap-1 text-xs text-ink-faint">
                    <KeyRound className="h-3 w-3" aria-hidden />
                    {p.key_error ? <span className="text-bad">Clave ilegible: vuelva a introducirla</span> : p.has_key ? <span>Clave guardada{p.key_hint ? ` (${p.key_hint})` : ""}</span> : <span>Sin clave</span>}
                    {p.external && (
                      <span className={`ml-2 ${p.allow_external ? "text-ok" : "text-warn"}`}>
                        · {p.allow_external ? "envío del texto permitido" : "envío del texto NO permitido"}
                      </span>
                    )}
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {!p.active && (
                  <button className="btn-primary" disabled={!!busy} onClick={() => void startActivation(p)}>
                    {(busy === `prev-${p.id}` || busy === `act-${p.id}`) && <Loader2 className="h-4 w-4 animate-spin" />} Usar por defecto
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
                          title: "¿Eliminar este proveedor de embeddings?",
                          message: `Se borrarán «${p.name}» y su clave guardada${p.active ? ". Es el predeterminado: se volverá al modelo del entorno" : ""}. Los vectores ya indexados se conservan hasta que los borre en «Vectores indexados».`,
                          confirmLabel: "Eliminar proveedor",
                          tone: "danger",
                        });
                        if (!ok) return;
                        const r = await run("del", () => api.deleteEmbedProfile(p.id), { title: `«${p.name}» eliminado` }, "No se pudo eliminar el proveedor");
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
              </p>
            )}
          </li>
        ))}
      </ul>

      {!draft ? (
        <div>
          <button className="btn" onClick={() => { setDraft(fromPreset(data.presets[0])); setProbe(null); }}>
            <Plus className="h-4 w-4" /> Añadir proveedor de embeddings
          </button>
        </div>
      ) : (
        <form className="rounded-lg border border-line bg-panel p-4" onSubmit={(e) => { e.preventDefault(); void save(false); }}>
          <h3 className="mb-3 text-sm font-semibold">{draft.id ? "Editar proveedor de embeddings" : "Nuevo proveedor de embeddings"}</h3>
          {locked && (
            <p className="mb-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-warn">
              Es el proveedor predeterminado: el modelo, los prefijos y las dimensiones no se pueden cambiar sin invalidar los vectores ya indexados. Active otro
              proveedor primero, o cree uno nuevo.
            </p>
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            {!draft.id && (
              <div className="sm:col-span-2">
                <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="epreset">Plantilla</label>
                <select id="epreset" className={input} value={draft.presetId}
                  onChange={(e) => { setDraft(fromPreset(data.presets.find((x) => x.id === e.target.value))); setProbe(null); }}>
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
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="ename">Nombre</label>
              <input id="ename" className={input} value={draft.name} required maxLength={120} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="ekind">Tipo de API</label>
              <select id="ekind" className={input} value={draft.kind} disabled={locked} onChange={(e) => setDraft({ ...draft, kind: e.target.value as Draft["kind"] })}>
                <option value="openai-compatible">Compatible con OpenAI (/embeddings)</option>
                <option value="ollama">Ollama (/api/embed)</option>
              </select>
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="eurl">URL base de la API</label>
              <input id="eurl" className={input} value={draft.base_url} required placeholder="https://…" onChange={(e) => setDraft({ ...draft, base_url: e.target.value })} />
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="ekey">Clave API</label>
              <input id="ekey" type="password" autoComplete="off" spellCheck={false} className={input} value={draft.api_key} disabled={draft.clear_api_key}
                placeholder={draft.hasKey ? `Clave guardada${draft.keyHint ? ` (${draft.keyHint})` : ""} — déjelo vacío para conservarla` : "Opcional para servidores propios"}
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
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="emodel">Modelo</label>
              <input id="emodel" className={input} value={draft.model} required disabled={locked} placeholder="Nombre exacto del modelo en ese servicio"
                onChange={(e) => setDraft({ ...draft, model: e.target.value })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="eqp">Prefijo de consulta</label>
              <input id="eqp" className={input} value={draft.query_prefix} disabled={locked} maxLength={200} placeholder="(vacío si el modelo no lo necesita)"
                onChange={(e) => setDraft({ ...draft, query_prefix: e.target.value })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="epp">Prefijo de documento</label>
              <input id="epp" className={input} value={draft.passage_prefix} disabled={locked} maxLength={200} placeholder="(vacío si el modelo no lo necesita)"
                onChange={(e) => setDraft({ ...draft, passage_prefix: e.target.value })} />
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="edim">Dimensiones (opcional, solo modelos Matryoshka)</label>
              <input id="edim" className={input} value={draft.dimensions} disabled={locked} inputMode="numeric" placeholder="vacío = todas (p. ej. 768)"
                onChange={(e) => setDraft({ ...draft, dimensions: e.target.value })} />
              <p className="mt-1 text-xs text-ink-faint">Un modelo usa siempre los mismos prefijos al indexar y al buscar; cambiarlos exige reindexar.</p>
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="esim">Umbral de similitud (opcional)</label>
              <input id="esim" className={input} value={draft.min_similarity} inputMode="decimal" placeholder="vacío = el global del servidor (0,80, calibrado para e5)"
                onChange={(e) => setDraft({ ...draft, min_similarity: e.target.value })} />
              <p className="mt-1 text-xs text-ink-faint">
                Similitud coseno mínima para considerar relevante un fragmento. Cada modelo reparte sus similitudes de forma distinta (EmbeddingGemma 2 ≈ 0,72):
                se puede cambiar con el proveedor activo y rige de inmediato, sin reindexar. Calíbrelo con <code>python -m app.evaluate</code>.
              </p>
            </div>
          </div>

          {probe?.forId === "draft" && (
            <p className={`mt-3 text-xs ${probe.ok ? "text-ok" : "text-bad"}`} role="status">{probe.detail}</p>
          )}

          {external && (
            <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs text-warn">
              <p className="flex items-start gap-1.5 font-medium">
                <ShieldAlert className="mt-px h-4 w-4 shrink-0" aria-hidden />
                {host(draft.base_url)} es un servicio externo
              </p>
              <p className="mt-1">
                Para indexar, se enviará a ese servicio <strong>el texto íntegro de todos sus documentos</strong> (todos los fragmentos), y después cada pregunta que
                haga. Es mucho más que con un modelo de lenguaje, que solo recibe los fragmentos relevantes. Revise la política de retención y de uso de datos del
                proveedor antes de usarlo con información sensible.
              </p>
              <label className="mt-2 flex items-start gap-2 text-ink">
                <input type="checkbox" className="mt-0.5 accent-petrol" checked={draft.allow_external} onChange={(e) => setDraft({ ...draft, allow_external: e.target.checked })} />
                <span>Permitir el envío del texto de los documentos a este servicio externo</span>
              </label>
            </div>
          )}

          <div className="mt-4 flex flex-wrap gap-2">
            <button type="submit" className="btn-primary" disabled={!!busy}>
              {busy === "save" && <Loader2 className="h-4 w-4 animate-spin" />} Guardar
            </button>
            <button type="button" className="btn" disabled={!!busy || !draft.base_url || !draft.model} onClick={() => void testDraft()}>
              {busy === "test-draft" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plug className="h-4 w-4" />} Probar conexión
            </button>
            {!draft.active && (
              <button type="button" className="btn" disabled={!!busy || (external && !draft.allow_external)} onClick={() => void save(true)}
                title={external && !draft.allow_external ? "Marque el permiso de envío para poder activarlo" : undefined}>
                Guardar y usar por defecto…
              </button>
            )}
            <button type="button" className="btn" onClick={() => { setDraft(null); setProbe(null); }}>Cancelar</button>
          </div>
        </form>
      )}

      <section aria-labelledby="vec-h" className="rounded-lg border border-line bg-white p-3">
        <h3 id="vec-h" className="flex items-center gap-1.5 text-sm font-semibold">
          <Database className="h-4 w-4 text-petrol" aria-hidden /> Vectores indexados
        </h3>
        <p className="mt-0.5 text-xs text-ink-faint">Cada modelo conserva sus vectores por separado; no se mezclan. Borre los de modelos que ya no use para liberar espacio.</p>
        {data.models.length === 0 ? (
          <p className="mt-2 text-xs text-ink-soft">Aún no hay vectores (suba un documento para indexarlo).</p>
        ) : (
          <ul className="mt-2 divide-y divide-line">
            {data.models.map((m) => (
              <li key={m.id} className="flex flex-wrap items-center gap-2 py-1.5 text-xs">
                <span className="min-w-0 flex-1 truncate font-medium" title={m.name}>{m.name}</span>
                <span className="text-ink-faint">{m.provider} · {m.dimension} dim. · {m.vectors} vectores</span>
                {m.active ? (
                  <span className="badge bg-petrol text-white">En uso</span>
                ) : (
                  <button className="btn" disabled={!!busy} onClick={() => void purge(m.id, m.name, m.vectors)}>
                    <Trash2 className="h-3.5 w-3.5" /> Borrar vectores
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
