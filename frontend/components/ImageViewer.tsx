"use client";

import { AlertTriangle, ImageIcon, Loader2, ScanEye, ShieldAlert, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { useToast } from "@/components/Toaster";
import { api, type Doc, type ImageAnswer } from "@/lib/api";

interface Props {
  doc: Pick<Doc, "id" | "filename" | "width" | "height" | "description" | "warnings">;
  /** Solo en pantallas pequeñas, donde el visor es un panel deslizante. */
  onClose?: () => void;
}

/** Visor de imágenes: la imagen, su descripción automática (si hay un modelo de visión) y la pregunta «¿Qué ves en esta imagen?». */
export default function ImageViewer({ doc, onClose }: Props) {
  const toast = useToast();
  const [question, setQuestion] = useState("¿Qué ves en esta imagen?");
  const [answer, setAnswer] = useState<ImageAnswer | null>(null);
  const [busy, setBusy] = useState(false);
  const [noVision, setNoVision] = useState(false);

  useEffect(() => {
    setAnswer(null);
    setNoVision(false);
    setQuestion("¿Qué ves en esta imagen?");
  }, [doc.id]);

  async function ask() {
    const q = question.trim();
    if (!q) return;
    setBusy(true);
    setNoVision(false);
    try {
      const r = await api.describeDoc(doc.id, q);
      setAnswer(r);
    } catch (e) {
      const msg = (e as Error).message;
      if (/modelo de visi[oó]n/i.test(msg)) setNoVision(true);
      else toast.error("El modelo de visión no pudo responder", { id: "vis-ask", description: msg });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-line px-3 py-2">
        <ImageIcon className="h-4 w-4 shrink-0 text-ink-faint" aria-hidden />
        <span className="min-w-0 flex-1 truncate text-sm font-medium" title={doc.filename}>{doc.filename}</span>
        {doc.width && doc.height && <span className="shrink-0 text-xs text-ink-faint">{doc.width}×{doc.height}</span>}
        {onClose && (
          <button className="btn-icon lg:hidden" aria-label="Cerrar visor" onClick={onClose}>
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={api.fileUrl(doc.id)} alt={doc.description ?? doc.filename} className="mx-auto max-h-[55vh] max-w-full rounded-md border border-line bg-white object-contain" />

        <section className="mt-3 rounded-lg border border-line bg-white p-3" aria-label="Descripción de la imagen">
          <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-faint">
            <ScanEye className="h-3.5 w-3.5" aria-hidden /> Descripción automática
          </h3>
          {doc.description ? (
            <p className="mt-1 text-sm leading-relaxed">{doc.description}</p>
          ) : (
            <p className="mt-1 text-sm text-ink-soft">
              Esta imagen no tiene descripción. Active un modelo de visión (Gemini, Groq u Ollama) en{" "}
              <Link href="/ajustes" className="font-semibold text-petrol underline">Ajustes</Link> y vuelva a procesarla, o pregunte abajo.
            </p>
          )}
          <p className="mt-1 text-[11px] text-ink-faint">Generada por un modelo: puede equivocarse. Compárela con la imagen.</p>
          {doc.warnings.map((w) => (
            <p key={w} className="mt-1 flex items-start gap-1.5 text-xs text-warn">
              <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden /> {w}
            </p>
          ))}
        </section>

        <form
          className="mt-3 rounded-lg border border-line bg-panel p-3"
          onSubmit={(e) => {
            e.preventDefault();
            void ask();
          }}
        >
          <label htmlFor="imgq" className="mb-1 block text-xs font-medium text-ink-soft">Pregunte al modelo de visión sobre esta imagen</label>
          <div className="flex gap-2">
            <input id="imgq" className="min-w-0 flex-1 rounded-md border border-line bg-white px-2.5 py-1.5 text-sm" value={question} maxLength={2000} onChange={(e) => setQuestion(e.target.value)} />
            <button className="btn-primary shrink-0" type="submit" disabled={busy || !question.trim()}>
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanEye className="h-4 w-4" />} Preguntar
            </button>
          </div>
          {noVision && (
            <p className="mt-2 text-xs text-warn" role="status">
              No hay un modelo de visión activo. Configure uno en <Link href="/ajustes" className="font-semibold underline">Ajustes → Modelo de visión</Link>.
            </p>
          )}
          {answer && (
            <div className="mt-3 rounded-md border border-line bg-white p-2.5 text-sm" role="status">
              <p className="whitespace-pre-wrap leading-relaxed">{answer.text}</p>
              <p className="mt-1.5 flex items-start gap-1.5 text-[11px] text-ink-faint">
                {answer.external && <ShieldAlert className="mt-px h-3.5 w-3.5 shrink-0 text-warn" aria-hidden />}
                Respuesta automática de {answer.provider} · {answer.model}
                {answer.external ? " · la imagen se envió a un servicio externo" : ""}
              </p>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
