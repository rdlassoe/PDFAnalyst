"use client";

import { AlertTriangle, CheckCircle2, ExternalLink, ImageIcon, KeyRound, Loader2, Pencil, Plug, Plus, ShieldAlert, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useToast } from "@/components/Toaster";
import { api, isLocalUrl, type LlmKind, type LlmProfile, type LlmTestResult, type VisionImageTest, type VisionPreset, type VisionSettings } from "@/lib/api";

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

const emptyDraft = (p?: VisionPreset): Draft => ({
  id: null, presetId: p?.id ?? "", name: p && p.id !== "custom-vision" ? p.label : "", kind: p?.kind ?? "openai-compatible",
  base_url: p?.base_url ?? "", model: p?.model ?? "", api_key: "", clear_api_key: false, allow_external: false, hasKey: false, keyHint: null,
});

function host(url: string) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export default function VisionSettingsPanel() {
  const toast = useToast();
  const [data, setData] = useState<VisionSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [probe, setProbe] = useState<(LlmTestResult & { forId: string }) | null>(null);
  const [imgTest, setImgTest] = useState<(VisionImageTest & { forId: string }) | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setData(await api.visionSettings());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
      toast.error("No se pudo cargar la configuración del modelo de visión", { id: "vis-load", description: (e as Error).message });
    }
  }, [toast]);
  useEffect(() => {
    void load();
  }, [load]);

  async function run<T>(key: string, fn: () => Promise<T>, ok?: { title: string; description?: string }, failTitle = "No se pudo completar la acción"): Promise<T | undefined> {
    setBusy(key);
    try {
      const r = await fn();
      if (ok) toast.success(ok.title, { description: ok.description, id: `vis-${key}` });
      return r;
    } catch (e) {
      toast.error(failTitle, { id: `vis-${key}-err`, description: (e as Error).message });
    } finally {
      setBusy(null);
    }
  }

  async function activate(p: LlmProfile) {
    if (p.external) {
      const ok = await toast.confirm({
        title: `¿Describir las imágenes con «${p.name}»?`,
        message: `Cada imagen que suba (y cada figura de un PDF, cuando esté disponible) se enviará a ${host(p.base_url)} para que la describa. Con una cuenta gratuita, el proveedor puede usar ese contenido para mejorar sus productos. No lo use con imágenes privadas.`,
        confirmLabel: "Activar",
        tone: "danger",
      });
      if (!ok) return;
    }
    const r = await run(`act-${p.id}`, () => api.setActiveVision(p.id), { title: `«${p.name}» describirá las imágenes`, description: `${p.model} · se usará en las próximas imágenes` }, "No se pudo activar el modelo de visión");
    if (r) setData(r);
  }

  async function deactivate() {
    const r = await run("deact", () => api.setActiveVision("none"), { title: "Descripción de imágenes desactivada", description: "Las imágenes se seguirán indexando con su vector y su OCR." }, "No se pudo desactivar");
    if (r) setData(r);
  }

  async function test(p: LlmProfile) {
    const r = await run(`test-${p.id}`, () => api.testVision({ profile_id: p.id }), undefined, "No se pudo probar la conexión");
    if (r) {
      setProbe({ ...r, forId: p.id });
      setImgTest(null);
    }
  }

  async function testImage(p: LlmProfile) {
    if (p.external && !p.allow_external) {
      toast.error("Falta el permiso de envío", { id: "vis-consent", description: "Edite el modelo y marque «Permitir el envío de las imágenes» antes de probarlo." });
      return;
    }
    const r = await run(`img-${p.id}`, () => api.testVisionImage(p.id), undefined, "No se pudo probar con la imagen");
    if (r) {
      setImgTest({ ...r, forId: p.id });
      setProbe(null);
    }
  }

  function edit(p: LlmProfile) {
    setProbe(null);
    setImgTest(null);
    setModels([]);
    setDraft({ id: p.id, presetId: "", name: p.name, kind: p.kind, base_url: p.base_url, model: p.model, api_key: "", clear_api_key: false,
      allow_external: p.allow_external, hasKey: p.has_key, keyHint: p.key_hint });
  }

  async function loadModels() {
    if (!draft) return;
    const r = await run("models", () => api.testVision({ profile_id: draft.id, kind: draft.kind, base_url: draft.base_url, api_key: draft.api_key || null }), undefined, "No se pudieron cargar los modelos");
    if (!r) return;
    setProbe({ ...r, forId: "draft" });
    setModels(r.models);
  }

  async function save(andActivate: boolean) {
    if (!draft) return;
    const external = !isLocalUrl(draft.base_url);
    const body = { name: draft.name, kind: draft.kind, base_url: draft.base_url, model: draft.model, allow_external: external ? draft.allow_external : false };
    const saved = await run("save", async () => {
      let view: VisionSettings;
      if (draft.id) {
        view = await api.updateVisionProfile(draft.id, { ...body, api_key: draft.api_key || undefined, clear_api_key: draft.clear_api_key });
      } else {
        view = await api.createVisionProfile({ ...body, api_key: draft.api_key || undefined });
      }
      if (andActivate) {
        const id = draft.id ?? view.profiles.find((p) => p.name === draft.name.trim())!.id;
        view = await api.setActiveVision(id);
      }
      return view;
    }, andActivate
      ? { title: `«${draft.name.trim()}» guardado y activado`, description: "Describirá las próximas imágenes." }
      : { title: `«${draft.name.trim()}» guardado` }, "No se pudo guardar el modelo de visión");
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
      <div className="flex flex-col gap-2" aria-busy="true" aria-label="Cargando modelos de visión">
        <div className="skeleton h-20 w-full" />
        <div className="skeleton h-9 w-40" />
      </div>
    );
  }

  const preset = data.presets.find((p) => p.id === draft?.presetId);
  const external = draft ? !isLocalUrl(draft.base_url) : false;
  const input = "w-full rounded-md border border-line bg-white px-2.5 py-1.5 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-petrol";
  const activeProfile = data.profiles.find((p) => p.active);

  return (
    <div className="flex flex-col gap-4">
      <div className={`flex items-start gap-2 rounded-lg border p-3 text-sm ${activeProfile ? "border-petrol bg-white" : "border-line bg-panel"}`}>
        <ImageIcon className="mt-0.5 h-4 w-4 shrink-0 text-petrol" aria-hidden />
        <div className="min-w-0 flex-1">
          {activeProfile ? (
            <p>
              Las imágenes se describen con <strong>{activeProfile.name}</strong> ({activeProfile.model}){activeProfile.external ? ", un servicio externo: cada imagen sale de su equipo" : ", en su equipo"}.
            </p>
          ) : (
            <p>
              <strong>Sin modelo de visión.</strong> Las imágenes se indexan con su vector (EmbeddingGemma 2) y su OCR, pero el sistema no puede decir qué ve en ellas.
              Configure uno abajo (Gemini, Groq, Ollama…).
            </p>
          )}
          <p className="mt-1 text-xs text-ink-soft">
            Las descripciones son automáticas y pueden equivocarse: se muestran siempre junto a la imagen. Este modelo es independiente del de chat; puede usar la misma cuenta con otro modelo que admita imágenes.
          </p>
        </div>
        {activeProfile && (
          <button className="btn shrink-0" disabled={!!busy} onClick={() => void deactivate()}>
            {busy === "deact" && <Loader2 className="h-4 w-4 animate-spin" />} Desactivar
          </button>
        )}
      </div>

      <label className="flex items-start gap-2 rounded-lg border border-line bg-white p-3 text-sm">
        <input
          type="checkbox"
          className="mt-0.5 accent-petrol"
          checked={data.answer_with_images}
          disabled={!!busy}
          onChange={async (e) => {
            const r = await run("answer", () => api.setVisionAnswer(e.target.checked), { title: e.target.checked ? "Responderá mirando las figuras" : "Responderá solo con texto" }, "No se pudo cambiar el ajuste");
            if (r) setData(r);
          }}
        />
        <span>
          <strong>Responder mirando las figuras de los PDF.</strong> Cuando la evidencia de una pregunta incluye figuras (imágenes o gráficos), el modelo de visión recibe la pregunta, los
          fragmentos relevantes y esas imágenes (hasta 3) y contesta mirándolas. Con un servicio externo, <strong>eso envía también la pregunta y los fragmentos</strong>, no solo las imágenes.
          Si lo desactiva, las figuras solo se usan a través de su descripción guardada.
        </span>
      </label>

      <ul className="flex flex-col gap-2">
        {data.profiles.map((p) => (
          <li key={p.id} className={`rounded-lg border bg-white p-3 ${p.active ? "border-petrol ring-1 ring-petrol" : "border-line"}`}>
            <div className="flex flex-wrap items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{p.name}</span>
                  {p.active && <span className="badge bg-petrol text-white">En uso</span>}
                  <span className={`badge ${p.external ? "bg-amber-50 text-warn" : "bg-emerald-50 text-ok"}`}>{p.external ? "Servicio externo" : "Local"}</span>
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
                      · {p.allow_external ? "envío de imágenes permitido" : "envío de imágenes NO permitido"}
                    </span>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {!p.active && (
                  <button className="btn-primary" disabled={!!busy || (p.external && !p.allow_external)} onClick={() => void activate(p)}
                    title={p.external && !p.allow_external ? "Edite el modelo y marque el permiso de envío de imágenes" : undefined}>
                    {busy === `act-${p.id}` && <Loader2 className="h-4 w-4 animate-spin" />} Usar para describir
                  </button>
                )}
                <button className="btn" disabled={!!busy} onClick={() => void test(p)}>
                  {busy === `test-${p.id}` ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plug className="h-4 w-4" />} Probar conexión
                </button>
                <button className="btn" disabled={!!busy} onClick={() => void testImage(p)} title="Envía una imagen sintética (un círculo rojo) y muestra lo que el modelo describe">
                  {busy === `img-${p.id}` ? <Loader2 className="h-4 w-4 animate-spin" /> : <ImageIcon className="h-4 w-4" />} Probar con una imagen de prueba
                </button>
                <button className="btn" onClick={() => edit(p)} aria-label={`Editar ${p.name}`}>
                  <Pencil className="h-4 w-4" /> Editar
                </button>
                <button
                  className="btn-icon"
                  aria-label={`Eliminar ${p.name}`}
                  onClick={async () => {
                    const ok = await toast.confirm({
                      title: "¿Eliminar este modelo de visión?",
                      message: `Se borrarán «${p.name}» y su clave API guardada${p.active ? ". Está en uso: se desactivará la descripción de imágenes" : ""}. No se puede deshacer.`,
                      confirmLabel: "Eliminar modelo",
                      tone: "danger",
                    });
                    if (!ok) return;
                    const r = await run("del", () => api.deleteVisionProfile(p.id), { title: `«${p.name}» eliminado` }, "No se pudo eliminar el modelo");
                    if (r) setData(r);
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
            {probe?.forId === p.id && (
              <p className={`mt-2 flex items-start gap-1.5 text-xs ${probe.ok ? "text-ok" : "text-bad"}`} role="status">
                {probe.ok ? <CheckCircle2 className="mt-px h-3.5 w-3.5 shrink-0" /> : <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" />}
                {probe.detail}
                {probe.ok && !probe.models.includes(p.model) && <span className="text-warn"> · el modelo «{p.model}» no aparece en la lista del proveedor</span>}
              </p>
            )}
            {imgTest?.forId === p.id && (
              <div className={`mt-2 rounded-md border p-2 text-xs ${imgTest.ok ? "border-emerald-200 bg-emerald-50" : "border-red-200 bg-red-50 text-bad"}`} role="status">
                {imgTest.ok ? (
                  <>
                    <p className="font-medium text-ok">El modelo ve imágenes. Respuesta para un círculo rojo sobre fondo blanco:</p>
                    <p className="mt-1 text-ink">«{imgTest.description}»</p>
                  </>
                ) : (
                  <p className="flex items-start gap-1.5"><AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" />{imgTest.detail}</p>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>

      {!draft ? (
        <div>
          <button className="btn" onClick={() => { setDraft(emptyDraft(data.presets[0])); setModels([]); setProbe(null); setImgTest(null); }}>
            <Plus className="h-4 w-4" /> Añadir modelo de visión
          </button>
        </div>
      ) : (
        <form className="rounded-lg border border-line bg-panel p-4" onSubmit={(e) => { e.preventDefault(); void save(false); }}>
          <h3 className="mb-3 text-sm font-semibold">{draft.id ? "Editar modelo de visión" : "Nuevo modelo de visión"}</h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {!draft.id && (
              <div className="sm:col-span-2">
                <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="vpreset">Plantilla</label>
                <select id="vpreset" className={input} value={draft.presetId}
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
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="vname">Nombre</label>
              <input id="vname" className={input} value={draft.name} required maxLength={120} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="vkind">Tipo de API</label>
              <select id="vkind" className={input} value={draft.kind} onChange={(e) => setDraft({ ...draft, kind: e.target.value as LlmKind })}>
                <option value="ollama">Ollama</option>
                <option value="openai-compatible">Compatible con OpenAI (Gemini, Groq, OpenAI…)</option>
              </select>
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="vurl">URL base de la API</label>
              <input id="vurl" className={input} value={draft.base_url} required placeholder="https://…" onChange={(e) => setDraft({ ...draft, base_url: e.target.value })} />
            </div>
            <div className="sm:col-span-2">
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="vkey">Clave API</label>
              <input id="vkey" type="password" autoComplete="off" spellCheck={false} className={input} value={draft.api_key} disabled={draft.clear_api_key}
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
              <label className="mb-1 block text-xs font-medium text-ink-soft" htmlFor="vmodel">Modelo (debe admitir imágenes)</label>
              <div className="flex gap-2">
                <input id="vmodel" className={input} list="vision-models" value={draft.model} required placeholder="Nombre exacto de un modelo con visión"
                  onChange={(e) => setDraft({ ...draft, model: e.target.value })} />
                <datalist id="vision-models">{models.map((m) => <option key={m} value={m} />)}</datalist>
                <button type="button" className="btn shrink-0" disabled={!draft.base_url || !!busy} onClick={() => void loadModels()}>
                  {busy === "models" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plug className="h-4 w-4" />} Cargar modelos
                </button>
              </div>
              <p className="mt-1 text-xs text-ink-faint">La lista del proveedor no distingue qué modelos admiten imágenes: compruébelo con «Probar con una imagen de prueba» tras guardar.</p>
              {probe?.forId === "draft" && (
                <p className={`mt-1 text-xs ${probe.ok ? "text-ok" : "text-bad"}`} role="status">{probe.detail}</p>
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
                Cada imagen que suba se enviará a ese servicio <strong>reducida y sin metadatos (EXIF)</strong>, pero visible para él. Las imágenes pueden contener caras, voces escritas, documentos
                o ubicaciones. Revise su política de retención y de uso de datos (en las cuentas gratuitas suele permitirse usarlos para mejorar el producto) y no lo use con imágenes privadas.
              </p>
              <label className="mt-2 flex items-start gap-2 text-ink">
                <input type="checkbox" className="mt-0.5 accent-petrol" checked={draft.allow_external} onChange={(e) => setDraft({ ...draft, allow_external: e.target.checked })} />
                <span>Permitir el envío de las imágenes a este servicio externo</span>
              </label>
            </div>
          )}

          <div className="mt-4 flex flex-wrap gap-2">
            <button type="submit" className="btn-primary" disabled={!!busy}>
              {busy === "save" && <Loader2 className="h-4 w-4 animate-spin" />} Guardar
            </button>
            <button type="button" className="btn" disabled={!!busy || (external && !draft.allow_external)} onClick={() => void save(true)}
              title={external && !draft.allow_external ? "Marque el permiso de envío para poder activarlo" : undefined}>
              Guardar y usar para describir
            </button>
            <button type="button" className="btn" onClick={() => { setDraft(null); setProbe(null); }}>Cancelar</button>
          </div>
        </form>
      )}
    </div>
  );
}
