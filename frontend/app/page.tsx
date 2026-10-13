"use client";

import { FileSearch, FileText, MessageSquare, Settings } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Chat from "@/components/Chat";
import DocumentPanel from "@/components/DocumentPanel";
import PdfViewer from "@/components/PdfViewer";
import { useToast } from "@/components/Toaster";
import { api, type Citation, type Collection, type Doc, type Health } from "@/lib/api";

interface Viewing {
  docId: string;
  filename: string;
  page: number;
  highlight: string | null;
  key: string | null;
}

function Dot({ ok, label, detail }: { ok: boolean | undefined; label: string; detail?: string }) {
  return (
    <span className="flex items-center gap-1.5" title={detail}>
      <span className={`h-2 w-2 rounded-full ${ok === undefined ? "bg-ink-faint" : ok ? "bg-ok" : "bg-bad"}`} aria-hidden />
      <span>{label}</span>
      <span className="sr-only">{ok ? "operativo" : "no disponible"}</span>
    </span>
  );
}

export default function Home() {
  const toast = useToast();
  const [docs, setDocs] = useState<Doc[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [collectionId, setCollectionId] = useState<string | null>(null);
  const [viewing, setViewing] = useState<Viewing | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false); // visor como panel deslizante en pantallas pequeñas
  const [mobileTab, setMobileTab] = useState<"docs" | "chat">("chat");
  const [health, setHealth] = useState<Health | null>(null);

  // Estado anterior, para notificar transiciones (procesamiento terminado, API caída/recuperada, LLM caído/recuperado).
  const prevStatus = useRef<Map<string, Doc["status"]>>(new Map());
  const apiDown = useRef(false);
  const llmOk = useRef<boolean | undefined>(undefined);

  const reportApi = useCallback(
    (down: boolean, detail?: string) => {
      if (down && !apiDown.current) {
        apiDown.current = true;
        toast.error("Sin conexión con la API", {
          id: "api-down",
          duration: 0,
          description: `${detail ?? "No responde"}. Compruebe que el servicio está en ejecución; se reintentará automáticamente.`,
        });
      } else if (!down && apiDown.current) {
        apiDown.current = false;
        toast.dismiss("api-down");
        toast.success("Conexión restablecida", { id: "api-up", duration: 2500 });
      }
    },
    [toast],
  );

  const openDoc = useCallback((d: Pick<Doc, "id" | "filename">) => {
    setViewing({ docId: d.id, filename: d.filename, page: 1, highlight: null, key: null });
    setDrawerOpen(true);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [d, c] = await Promise.all([api.documents(), api.collections()]);
      reportApi(false);
      // Notificar documentos cuyo procesamiento terminó desde la última consulta (no en la carga inicial).
      for (const doc of d) {
        const prev = prevStatus.current.get(doc.id);
        if (prev && (prev === "uploaded" || prev === "processing")) {
          if (doc.status === "ready") {
            const detail = [doc.page_count && `${doc.page_count} páginas`, doc.chunk_count && `${doc.chunk_count} fragmentos`, doc.ocr_pages ? `${doc.ocr_pages} con OCR` : null]
              .filter(Boolean)
              .join(" · ");
            toast.success(`«${doc.filename}» está listo`, {
              id: `doc-${doc.id}`,
              description: detail || "Ya puede consultarlo.",
              action: { label: "Abrir", onClick: () => openDoc(doc) },
            });
          } else if (doc.status === "failed") {
            toast.error(`No se pudo procesar «${doc.filename}»`, {
              id: `doc-${doc.id}`,
              description: doc.error ?? "Error desconocido durante el procesamiento.",
              action: {
                label: "Reintentar",
                onClick: () =>
                  void api
                    .reprocess(doc.id)
                    .then(() => {
                      toast.info("Reprocesamiento en cola", { description: doc.filename });
                      return refresh();
                    })
                    .catch((e: Error) => toast.error("No se pudo reintentar", { description: e.message })),
              },
            });
          }
        }
        prevStatus.current.set(doc.id, doc.status);
      }
      setDocs(d);
      setCollections(c);
      setLoaded(true);
    } catch (e) {
      reportApi(true, (e as Error).message);
      setLoaded(true);
    }
  }, [toast, reportApi, openDoc]);

  const checkHealth = useCallback(async () => {
    try {
      const h = await api.health();
      reportApi(false);
      setHealth(h);
      const ok = h.generation.ok;
      if (llmOk.current === true && !ok) {
        toast.warning("El modelo de generación dejó de responder", { id: "llm-state", description: h.generation.detail });
      } else if (llmOk.current === false && ok) {
        toast.success("Modelo de generación disponible", { id: "llm-state", description: `${h.generation.provider} · ${h.generation.model}` });
      }
      llmOk.current = ok;
    } catch (e) {
      setHealth(null);
      reportApi(true, (e as Error).message);
    }
  }, [toast, reportApi]);

  useEffect(() => {
    void refresh();
    void checkHealth();
    const t = setInterval(() => void checkHealth(), 30000);
    return () => clearInterval(t);
  }, [refresh, checkHealth]);

  // Sondeo mientras haya documentos en procesamiento.
  const busy = docs.some((d) => d.status === "uploaded" || d.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => void refresh(), 2000);
    return () => clearInterval(t);
  }, [busy, refresh]);

  // Si se elimina el documento que se está viendo o seleccionando, se limpia el estado asociado.
  useEffect(() => {
    if (!loaded) return;
    const ids = new Set(docs.map((d) => d.id));
    setViewing((v) => (v && !ids.has(v.docId) ? null : v));
    setSelected((s) => {
      const n = new Set([...s].filter((id) => ids.has(id)));
      return n.size === s.size ? s : n;
    });
  }, [docs, loaded]);
  useEffect(() => {
    if (!viewing) setDrawerOpen(false);
  }, [viewing]);

  // Esc cierra el visor deslizante.
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const readyDocs = docs.filter((d) => d.status === "ready");
  const colName = collections.find((c) => c.id === collectionId)?.name;
  const scopeLabel = useMemo(() => {
    if (selected.size) return `${selected.size} documento${selected.size > 1 ? "s" : ""} seleccionado${selected.size > 1 ? "s" : ""}`;
    if (colName) return `colección «${colName}»`;
    return `todos los documentos listos (${readyDocs.length})`;
  }, [selected, colName, readyDocs.length]);

  async function openCitation(c: Citation) {
    if (!c.document_id) {
      toast.warning("El documento citado ya no existe", { id: "cite-gone", description: "Se eliminó después de generar esta respuesta." });
      return;
    }
    let highlight = c.snippet;
    if (c.chunk_id != null) {
      try {
        highlight = (await api.chunk(c.chunk_id)).text; // el fragmento exacto completo
      } catch {
        /* se usa el extracto */
      }
    }
    setViewing({ docId: c.document_id, filename: c.filename ?? "Documento", page: c.page_start ?? 1, highlight, key: c.chunk_id != null ? String(c.chunk_id) : null });
    setDrawerOpen(true);
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center gap-4 border-b border-line bg-paper px-4 py-2">
        <div className="flex items-center gap-2">
          <FileSearch className="h-5 w-5 text-petrol" aria-hidden />
          <h1 className="text-base font-semibold tracking-tight">PDF Analyst</h1>
        </div>
        <div className="ml-auto flex items-center gap-4 text-xs text-ink-soft">
          <div className="hidden items-center gap-4 md:flex" aria-label="Estado del sistema">
            <Dot ok={health?.database.ok && health.pgvector.ok} label="PostgreSQL + pgvector" detail={health?.pgvector.version ? `pgvector ${health.pgvector.version}` : undefined} />
            <Dot ok={health ? true : undefined} label={`Embeddings: ${health?.embeddings.model ?? "…"}`} detail="Modelo local" />
            <Dot ok={health?.generation.ok} label={`Generación: ${health?.generation.model ?? "…"}`} detail={health?.generation.detail} />
          </div>
          <Link href="/ajustes" className="btn" aria-label="Ajustes: proveedor de LLM y API">
            <Settings className="h-4 w-4" /> <span className="hidden sm:inline">Ajustes</span>
          </Link>
        </div>
      </header>

      {health && !health.generation.ok && (
        <div role="status" className="border-b border-line bg-amber-50 px-4 py-1.5 text-xs text-warn">
          El modelo de generación no está disponible ({health.generation.detail}). Puede seguir gestionando documentos; las preguntas
          devolverán un aviso hasta que se restablezca. <Link href="/ajustes" className="font-semibold underline">Abrir Ajustes</Link>
        </div>
      )}

      {/* Navegación entre paneles en pantallas pequeñas */}
      <nav className="grid grid-cols-2 border-b border-line bg-paper lg:hidden" aria-label="Paneles">
        {([["docs", "Documentos", FileText], ["chat", "Conversación", MessageSquare]] as const).map(([id, label, Icon]) => (
          <button
            key={id}
            onClick={() => setMobileTab(id)}
            aria-current={mobileTab === id ? "page" : undefined}
            className={`flex items-center justify-center gap-1.5 py-2 text-sm font-medium transition-colors ${mobileTab === id ? "border-b-2 border-petrol text-petrol" : "text-ink-soft hover:text-ink"}`}
          >
            <Icon className="h-4 w-4" aria-hidden /> {label}
            {id === "docs" && docs.length > 0 && <span className="badge bg-panel text-ink-soft">{docs.length}</span>}
          </button>
        ))}
      </nav>

      <main className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_minmax(0,0.9fr)]">
        <aside className={`${mobileTab === "docs" ? "block" : "hidden"} min-h-0 border-r border-line bg-paper lg:block`}>
          <DocumentPanel
            loading={!loaded}
            docs={docs}
            collections={collections}
            selectedDocs={selected}
            collectionId={collectionId}
            activeDocId={viewing?.docId ?? null}
            onToggleDoc={(id) =>
              setSelected((s) => {
                const n = new Set(s);
                n.has(id) ? n.delete(id) : n.add(id);
                return n;
              })
            }
            onSelectCollection={(id) => {
              setCollectionId(id);
              setSelected(new Set());
            }}
            onOpenDoc={openDoc}
            onChanged={() => void refresh()}
          />
        </aside>
        <section className={`${mobileTab === "chat" ? "block" : "hidden"} min-h-0 border-r border-line lg:block`}>
          <Chat
            documentIds={[...selected]}
            collectionId={selected.size ? null : collectionId}
            scopeLabel={scopeLabel}
            readyCount={readyDocs.length}
            onOpenCitation={(c) => void openCitation(c)}
            activeCitation={viewing?.key ?? null}
          />
        </section>
        <section
          className={`min-h-0 bg-paper ${drawerOpen ? "fixed inset-0 z-50 block animate-slide-in-right" : "hidden"} lg:static lg:z-auto lg:block lg:animate-none`}
          aria-label="Visor de documento"
        >
          <PdfViewer
            url={viewing ? api.fileUrl(viewing.docId) : null}
            filename={viewing?.filename}
            page={viewing?.page ?? 1}
            highlight={viewing?.highlight}
            onPageChange={(p) => setViewing((v) => (v ? { ...v, page: p, highlight: null } : v))}
            onClose={() => setDrawerOpen(false)}
          />
        </section>
      </main>
    </div>
  );
}
