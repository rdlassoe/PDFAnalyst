"use client";

import { AlertTriangle, CheckCircle2, FileText, FolderPlus, ImageIcon, Loader2, RotateCw, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { useToast } from "@/components/Toaster";
import { api, type Collection, type Doc } from "@/lib/api";

interface Props {
  loading?: boolean;
  docs: Doc[];
  collections: Collection[];
  selectedDocs: Set<string>;
  collectionId: string | null;
  activeDocId: string | null;
  onToggleDoc: (id: string) => void;
  onSelectCollection: (id: string | null) => void;
  onOpenDoc: (doc: Doc) => void;
  onChanged: () => void;
}

const STATUS: Record<Doc["status"], { label: string; cls: string }> = {
  uploaded: { label: "En cola", cls: "bg-panel text-ink-soft" },
  processing: { label: "Procesando", cls: "bg-amber-50 text-warn" },
  ready: { label: "Listo", cls: "bg-emerald-50 text-ok" },
  failed: { label: "Error", cls: "bg-red-50 text-bad" },
};

const fmtSize = (b: number) => (b > 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);

export default function DocumentPanel(p: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [newCol, setNewCol] = useState("");

  async function upload(files: FileList | File[]) {
    const all = Array.from(files);
    // El navegador solo prefiltra por extensión/tipo; el servidor valida por la FIRMA del archivo.
    const pdfs = all.filter((f) => /\.(pdf|png|jpe?g|webp)$/i.test(f.name) || /^(application\/pdf|image\/(png|jpeg|webp))$/.test(f.type));
    const rejected = all.length - pdfs.length;
    if (rejected > 0) {
      toast.warning(rejected === 1 ? "Se omitió 1 archivo no admitido" : `Se omitieron ${rejected} archivos no admitidos`, {
        description: "PDF Analyst admite PDF e imágenes PNG, JPEG y WebP.",
      });
    }
    if (pdfs.length === 0) {
      if (inputRef.current) inputRef.current.value = "";
      return;
    }
    setBusy(true);
    try {
      for (const f of pdfs) {
        try {
          await toast.promise(
            api.upload(f),
            {
              loading: `Subiendo «${f.name}»…`,
              success: (r) =>
                r.created
                  ? { title: `«${r.document.filename}» subido`, description: "Se está procesando en segundo plano; le avisaremos al terminar." }
                  : { title: "Ya estaba cargado", description: `«${r.document.filename}» tiene el mismo contenido que un documento existente.`, kind: "info" },
              error: (e) => ({ title: `No se pudo subir «${f.name}»`, description: e.message }),
            },
          );
        } catch {
          /* ya notificado por toast.promise */
        }
        p.onChanged();
      }
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  /** Ejecuta una acción, refresca y notifica el resultado (éxito opcional; el error siempre). */
  async function guard(fn: () => Promise<unknown>, success?: { title: string; description?: string }) {
    try {
      await fn();
      if (success) toast.success(success.title, { description: success.description });
      p.onChanged();
    } catch (e) {
      toast.error("No se pudo completar la acción", { description: (e as Error).message });
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files.length) void upload(e.dataTransfer.files);
        }}
        className={`m-3 rounded-md border border-dashed px-3 py-4 text-center text-sm ${dragging ? "border-petrol bg-petrol-tint" : "border-line bg-white"}`}
      >
        <input ref={inputRef} type="file" accept="application/pdf,.pdf,image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp" multiple hidden onChange={(e) => e.target.files && upload(e.target.files)} />
        <button className="btn-primary" disabled={busy} onClick={() => inputRef.current?.click()}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          Subir PDF o imagen
        </button>
        <p className="mt-2 text-xs text-ink-faint">o arrastre archivos aquí. Se procesan localmente.</p>
      </div>

      <div className="flex items-center gap-2 px-3 pb-2">
        <label htmlFor="col" className="text-xs font-medium text-ink-soft">
          Colección
        </label>
        <select
          id="col"
          className="min-w-0 flex-1 rounded-md border border-line bg-white px-2 py-1 text-sm"
          value={p.collectionId ?? ""}
          onChange={(e) => p.onSelectCollection(e.target.value || null)}
        >
          <option value="">Todos los documentos</option>
          {p.collections.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name} ({c.document_ids.length})
            </option>
          ))}
        </select>
        {p.collectionId && (
          <button
            className="btn-icon"
            aria-label="Eliminar colección"
            onClick={async () => {
              const col = p.collections.find((c) => c.id === p.collectionId);
              if (!col) return;
              const ok = await toast.confirm({
                title: "¿Eliminar la colección?",
                message: `Se eliminará «${col.name}». Los documentos que contiene no se borran.`,
                confirmLabel: "Eliminar colección",
                tone: "danger",
              });
              if (ok) await guard(async () => { await api.deleteCollection(col.id); p.onSelectCollection(null); }, { title: `Colección «${col.name}» eliminada` });
            }}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        )}
      </div>
      <form
        className="flex gap-2 px-3 pb-3"
        onSubmit={(e) => {
          e.preventDefault();
          const name = newCol.trim();
          if (name) void guard(async () => { const c = await api.createCollection(name); setNewCol(""); p.onSelectCollection(c.id); }, { title: `Colección «${name}» creada`, description: "Añada documentos con el icono de carpeta." });
        }}
      >
        <input
          className="min-w-0 flex-1 rounded-md border border-line bg-white px-2 py-1 text-sm"
          placeholder="Nueva colección…"
          aria-label="Nombre de la nueva colección"
          value={newCol}
          onChange={(e) => setNewCol(e.target.value)}
        />
        <button className="btn" type="submit" disabled={!newCol.trim()}>
          <FolderPlus className="h-4 w-4" /> Crear
        </button>
      </form>

      <div className="border-t border-line px-3 py-2 text-xs font-medium uppercase tracking-wide text-ink-faint">
        Documentos ({p.docs.length})
      </div>
      <ul className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {p.loading &&
          [0, 1, 2].map((i) => (
            <li key={i} className="px-2 py-2" aria-hidden>
              <div className="skeleton h-4 w-3/4" />
              <div className="skeleton mt-2 h-3 w-1/2" />
            </li>
          ))}
        {!p.loading && p.docs.length === 0 && (
          <li className="animate-fade-in px-3 py-8 text-center text-sm text-ink-faint">
            <FileText className="mx-auto mb-2 h-7 w-7 text-line" aria-hidden />
            Aún no hay documentos.
            <br />
            Suba un PDF o una imagen para empezar.
          </li>
        )}
        {p.docs.map((d) => {
          const st = STATUS[d.status];
          const inCol = p.collectionId ? p.collections.find((c) => c.id === p.collectionId)?.document_ids.includes(d.id) : false;
          return (
            <li key={d.id} className={`group rounded-md px-2 py-2 ${p.activeDocId === d.id ? "bg-petrol-tint" : "hover:bg-panel"}`}>
              <div className="flex items-start gap-2">
                <input
                  type="checkbox"
                  className="mt-1 accent-petrol"
                  aria-label={`Limitar la consulta a ${d.filename}`}
                  checked={p.selectedDocs.has(d.id)}
                  disabled={d.status !== "ready"}
                  onChange={() => p.onToggleDoc(d.id)}
                />
                <button className="min-w-0 flex-1 text-left" onClick={() => p.onOpenDoc(d)} title={d.filename}>
                  <div className="flex items-center gap-1.5">
                    {d.media_type === "image" && d.status === "ready" ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={api.thumbUrl(d.id)} alt="" className="h-8 w-8 shrink-0 rounded border border-line object-cover" />
                    ) : d.media_type === "image" ? (
                      <ImageIcon className="h-4 w-4 shrink-0 text-ink-faint" aria-hidden />
                    ) : (
                      <FileText className="h-4 w-4 shrink-0 text-ink-faint" aria-hidden />
                    )}
                    <span className="truncate text-sm font-medium">{d.filename}</span>
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-faint">
                    <span className={`badge ${st.cls}`}>
                      {d.status === "processing" && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
                      {d.status === "ready" && <CheckCircle2 className="mr-1 h-3 w-3" />}
                      {d.status === "failed" && <AlertTriangle className="mr-1 h-3 w-3" />}
                      {st.label}
                    </span>
                    {d.media_type === "image" ? <span>{d.width}×{d.height}</span> : <span>{d.page_count ?? "?"} pág.</span>}
                    <span>{fmtSize(d.size_bytes)}</span>
                    {!!d.ocr_pages && <span title="Páginas leídas con OCR">OCR {d.ocr_pages}</span>}
                    {d.media_type === "pdf" && !!d.figure_count && (
                      <span className="inline-flex items-center gap-0.5" title={`${d.figure_count} figuras detectadas (imágenes y gráficos); ${d.figures_described ?? 0} descritas por el modelo de visión`}>
                        <ImageIcon className="h-3 w-3" aria-hidden /> {d.figure_count} figura{d.figure_count > 1 ? "s" : ""}{d.figures_described ? ` (${d.figures_described} descritas)` : ""}
                      </span>
                    )}
                  </div>
                  {d.error && <p className="mt-1 text-xs text-bad">{d.error}</p>}
                </button>
                <div className="flex shrink-0 gap-0.5 opacity-60 group-hover:opacity-100 focus-within:opacity-100">
                  {p.collectionId && (
                    <button
                      className="btn-icon"
                      aria-label={inCol ? "Quitar de la colección" : "Añadir a la colección"}
                      title={inCol ? "Quitar de la colección" : "Añadir a la colección"}
                      onClick={() =>
                        guard(() => (inCol ? api.removeFromCollection(p.collectionId!, d.id) : api.addToCollection(p.collectionId!, d.id)), {
                          title: inCol ? "Quitado de la colección" : "Añadido a la colección",
                          description: d.filename,
                        })
                      }
                    >
                      <FolderPlus className={`h-4 w-4 ${inCol ? "text-petrol" : ""}`} />
                    </button>
                  )}
                  {(d.status === "failed" || d.status === "ready") && (
                    <button
                      className="btn-icon"
                      aria-label={d.status === "failed" ? "Reintentar procesamiento" : "Volver a procesar"}
                      title={d.status === "failed" ? "Reintentar" : "Volver a procesar: lee de nuevo el texto y las figuras (si hay un modelo de visión, las vuelve a describir)"}
                      onClick={async () => {
                        if (d.status === "ready") {
                          const ok = await toast.confirm({
                            title: "¿Volver a procesar este documento?",
                            message: `«${d.filename}» se leerá de nuevo (texto y figuras) y no se podrá consultar hasta terminar. Si hay un modelo de visión activo, cada figura se describirá otra vez (con un servicio externo, eso consume su cuota y envía las imágenes).`,
                            confirmLabel: "Volver a procesar",
                          });
                          if (!ok) return;
                        }
                        await guard(() => api.reprocess(d.id), { title: "Reprocesamiento en cola", description: d.filename });
                      }}
                    >
                      <RotateCw className="h-4 w-4" />
                    </button>
                  )}
                  <button
                    className="btn-icon"
                    aria-label={`Eliminar ${d.filename}`}
                    title="Eliminar"
                    onClick={async () => {
                      const ok = await toast.confirm({
                        title: "¿Eliminar este documento?",
                        message: `«${d.filename}», su texto, sus fragmentos y sus embeddings se borrarán de forma permanente. Las conversaciones anteriores conservan sus respuestas, pero sus citas a este documento dejarán de poder abrirse.`,
                        confirmLabel: "Eliminar documento",
                        tone: "danger",
                      });
                      if (ok) await guard(() => api.deleteDocument(d.id), { title: "Documento eliminado", description: d.filename });
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
