"use client";

import { ChevronLeft, ChevronRight, FileText, X, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useToast } from "@/components/Toaster";

// pdfjs-dist se importa dinámicamente: toca APIs del navegador (DOMMatrix) al cargarse.
type PdfjsLib = typeof import("pdfjs-dist");
let pdfjsPromise: Promise<PdfjsLib> | null = null;
function loadPdfjs(): Promise<PdfjsLib> {
  if (!pdfjsPromise) {
    pdfjsPromise = import("pdfjs-dist").then((lib) => {
      lib.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
      return lib;
    });
  }
  return pdfjsPromise;
}

const norm = (s: string) =>
  s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/\s+/g, " ").trim();

interface Props {
  url: string | null;
  filename?: string;
  page: number;
  /** Texto del fragmento citado: se resalta en la página. */
  highlight?: string | null;
  onPageChange: (p: number) => void;
  /** Solo en pantallas pequeñas, donde el visor es un panel deslizante. */
  onClose?: () => void;
}

export default function PdfViewer({ url, filename, page, highlight, onPageChange, onClose }: Props) {
  const toast = useToast();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fitted = useRef(false); // el ajuste al ancho se hace una vez por documento
  const [pdf, setPdf] = useState<import("pdfjs-dist").PDFDocumentProxy | null>(null);
  const [numPages, setNumPages] = useState(0);
  const [scale, setScale] = useState(1.2);
  const [error, setError] = useState<string | null>(null);
  const [matched, setMatched] = useState<number | null>(null);

  useEffect(() => {
    setPdf(null);
    fitted.current = false;
    setNumPages(0);
    setError(null);
    if (!url) return;
    let cancelled = false;
    let task: import("pdfjs-dist").PDFDocumentLoadingTask | null = null;
    loadPdfjs()
      .then((lib) => {
        task = lib.getDocument({ url });
        return task.promise;
      })
      .then((doc) => {
        if (cancelled) return void doc.destroy();
        setPdf(doc);
        setNumPages(doc.numPages);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(`No se pudo abrir el PDF: ${e?.message ?? e}`);
        toast.error("No se pudo abrir el PDF", { id: "pdf-error", description: String(e?.message ?? e) });
      });
    return () => {
      cancelled = true;
      task?.destroy();
    };
  }, [url]);

  useEffect(() => {
    if (!pdf) return;
    const safePage = Math.min(Math.max(page, 1), pdf.numPages);
    let cancelled = false;
    let renderTask: import("pdfjs-dist").RenderTask | null = null;
    (async () => {
      const lib = await loadPdfjs();
      const p = await pdf.getPage(safePage);
      if (cancelled) return;
      if (!fitted.current) {
        // Al abrir un documento, se ajusta al ancho disponible (pantallas pequeñas); en escritorio se mantiene 120 %.
        fitted.current = true;
        const w = scrollRef.current?.clientWidth;
        const fit = w ? (w - 32) / p.getViewport({ scale: 1 }).width : scale;
        if (fit < scale) {
          setScale(Math.max(0.4, +fit.toFixed(2)));
          return; // el efecto se vuelve a ejecutar con la nueva escala
        }
      }
      const viewport = p.getViewport({ scale });
      const canvas = canvasRef.current!;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      const ctx = canvas.getContext("2d")!;
      renderTask = p.render({ canvasContext: ctx, viewport, transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined });
      await renderTask.promise;

      const container = textRef.current!;
      container.innerHTML = "";
      container.style.width = `${viewport.width}px`;
      container.style.height = `${viewport.height}px`;
      const layer = new lib.TextLayer({ textContentSource: p.streamTextContent(), container, viewport });
      await layer.render();
      if (cancelled) return;

      // Resalta los trozos de texto de la página que pertenecen al fragmento citado.
      if (highlight) {
        const target = norm(highlight);
        let count = 0;
        layer.textDivs.forEach((div, i) => {
          const t = norm(layer.textContentItemsStr[i] ?? "");
          if (t.length >= 4 && target.includes(t)) {
            div.classList.add("hl");
            count++;
          }
        });
        setMatched(count);
        container.querySelector(".hl")?.scrollIntoView({ block: "center", behavior: "smooth" });
      } else setMatched(null);
    })().catch((e) => {
      if (e?.name !== "RenderingCancelledException" && !cancelled) setError(`Error al renderizar: ${e?.message ?? e}`);
    });
    return () => {
      cancelled = true;
      renderTask?.cancel();
    };
  }, [pdf, page, scale, highlight]);

  // Flechas ← → cambian de página cuando no se está escribiendo en un campo.
  useEffect(() => {
    if (!pdf) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName) || t.isContentEditable)) return;
      if (e.key === "ArrowLeft" && page > 1) onPageChange(page - 1);
      if (e.key === "ArrowRight" && page < pdf.numPages) onPageChange(page + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pdf, page, onPageChange]);

  if (!url) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-ink-faint">
        <FileText className="h-8 w-8" aria-hidden />
        <p className="text-sm">Seleccione un documento o una cita para verla aquí</p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-line bg-panel px-3 py-2 text-sm">
        {onClose && (
          <button className="btn-icon lg:hidden" aria-label="Cerrar visor" onClick={onClose}>
            <X className="h-4 w-4" />
          </button>
        )}
        <span className="min-w-0 flex-1 truncate font-medium" title={filename}>
          {filename}
        </span>
        <button className="btn-icon" aria-label="Página anterior" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
          <ChevronLeft className="h-4 w-4" />
        </button>
        <span className="tabular-nums text-ink-soft">
          {Math.min(page, numPages || page)} / {numPages || "–"}
        </span>
        <button className="btn-icon" aria-label="Página siguiente" disabled={page >= numPages} onClick={() => onPageChange(page + 1)}>
          <ChevronRight className="h-4 w-4" />
        </button>
        <button className="btn-icon" aria-label="Alejar" onClick={() => setScale((s) => Math.max(0.6, +(s - 0.2).toFixed(1)))}>
          <ZoomOut className="h-4 w-4" />
        </button>
        <button className="btn-icon" aria-label="Acercar" onClick={() => setScale((s) => Math.min(3, +(s + 0.2).toFixed(1)))}>
          <ZoomIn className="h-4 w-4" />
        </button>
      </div>
      {highlight && matched !== null && (
        <div className="border-b border-line bg-petrol-tint px-3 py-1 text-xs text-petrol-dark">
          {matched > 0 ? "Fragmento citado resaltado en la página" : "No se pudo localizar el fragmento exacto en la capa de texto (PDF escaneado o maquetación compleja)"}
        </div>
      )}
      {error && <div className="border-b border-line bg-red-50 px-3 py-2 text-sm text-bad">{error}</div>}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto bg-[#E9E7E1] p-4">
        <div className="relative mx-auto w-fit bg-white shadow-sm">
          <canvas ref={canvasRef} />
          <div ref={textRef} className="textLayer" />
        </div>
      </div>
    </div>
  );
}
