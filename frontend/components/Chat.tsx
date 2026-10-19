"use client";

import { AlertTriangle, BookOpenCheck, Copy, Info, Loader2, MessageSquarePlus, Send, ShieldAlert, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { useToast } from "@/components/Toaster";
import { api, type Citation, type Conversation, type Message } from "@/lib/api";

interface Props {
  documentIds: string[];
  collectionId: string | null;
  scopeLabel: string;
  readyCount: number;
  onOpenCitation: (c: Citation) => void;
  onOpenMedia: (documentId: string, filename: string, page?: number | null, chunkId?: number) => void; // abrir una imagen/figura relacionada en el visor
  activeCitation: string | null; // id del chunk abierto en el visor
}

const CITE = /\[(C\d+(?:\s*,\s*C\d+)*)\]/gi;

function AnswerText({ msg, onOpen }: { msg: Message; onOpen: (c: Citation) => void }) {
  const byLabel = new Map(msg.citations.map((c) => [c.label.toUpperCase(), c]));
  const parts: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  const re = new RegExp(CITE.source, "gi");
  while ((m = re.exec(msg.content))) {
    parts.push(msg.content.slice(last, m.index));
    const labels = m[1].split(",").map((s) => s.trim().toUpperCase());
    parts.push(
      <span key={m.index} className="mx-0.5 whitespace-nowrap">
        {labels.map((lab) => {
          const c = byLabel.get(lab);
          return c ? (
            <button
              key={lab}
              onClick={() => onOpen(c)}
              title={`${c.filename} · p. ${c.page_start}`}
              className="mx-px rounded bg-petrol-tint px-1.5 py-0.5 align-baseline text-xs font-semibold text-petrol hover:bg-petrol hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-petrol"
            >
              {lab}
            </button>
          ) : (
            <span key={lab} className="mx-px rounded bg-red-50 px-1.5 py-0.5 text-xs text-bad">{lab}?</span>
          );
        })}
      </span>,
    );
    last = m.index + m[0].length;
  }
  parts.push(msg.content.slice(last));
  return <p className="whitespace-pre-wrap leading-relaxed">{parts.map((p, i) => <Fragment key={i}>{p}</Fragment>)}</p>;
}

export default function Chat(props: Props) {
  const toast = useToast();
  const router = useRouter();
  const [convs, setConvs] = useState<Conversation[] | null>(null);
  const [convId, setConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingMsgs, setLoadingMsgs] = useState(false);
  const [text, setText] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set()); // eliminadas con «Deshacer» aún posible
  const endRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLTextAreaElement>(null);

  const refreshConvs = useCallback(async () => setConvs(await api.conversations()), []);

  useEffect(() => {
    refreshConvs().catch((e) => toast.error("No se pudieron cargar las conversaciones", { id: "convs-load", description: e.message }));
  }, [refreshConvs, toast]);

  useEffect(() => {
    if (!convId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    setLoadingMsgs(true);
    api
      .messages(convId)
      .then((m) => !cancelled && setMessages(m))
      .catch((e) => !cancelled && toast.error("No se pudo abrir la conversación", { id: "msgs-load", description: e.message }))
      .finally(() => !cancelled && setLoadingMsgs(false));
    return () => {
      cancelled = true;
    };
  }, [convId, toast]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" }); // sin return: React interpretaría el valor como cleanup
  }, [messages, pending]);

  // El área de texto crece con el contenido (máx. ~8 líneas).
  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 176)}px`;
  }, [text]);

  async function newConversation() {
    try {
      const c = await api.createConversation();
      await refreshConvs();
      setConvId(c.id);
      areaRef.current?.focus();
    } catch (e) {
      toast.error("No se pudo crear la conversación", { description: (e as Error).message });
    }
  }

  /** Eliminación diferida: desaparece de inmediato y se confirma al cerrarse el aviso, salvo que se pulse «Deshacer». */
  function removeConversation(c: Conversation) {
    const id = `del-${c.id}`;
    setHidden((h) => new Set(h).add(c.id));
    if (convId === c.id) setConvId(null);
    const restore = () =>
      setHidden((h) => {
        const n = new Set(h);
        n.delete(c.id);
        return n;
      });
    toast.info("Conversación eliminada", {
      id,
      description: c.title,
      duration: 7000,
      action: {
        label: "Deshacer",
        onClick: () => {
          restore();
          toast.success("Conversación restaurada", { duration: 2500 });
        },
      },
      onDismiss: async (reason) => {
        if (reason === "action") return;
        try {
          await api.deleteConversation(c.id, reason === "api"); // keepalive si se está cerrando la página
          await refreshConvs().catch(() => undefined);
          restore();
        } catch (e) {
          restore();
          toast.error("No se pudo eliminar la conversación", { description: (e as Error).message });
        }
      },
    });
  }

  async function send() {
    const q = text.trim();
    if (!q || pending) return;
    setText("");
    setPending(q);
    try {
      let id = convId;
      if (!id) {
        id = (await api.createConversation()).id;
        setConvId(id);
      }
      const answer = await api.ask(id, q, props.documentIds, props.collectionId);
      setMessages(await api.messages(id));
      refreshConvs().catch(() => undefined);
      notifyAnswer(answer);
    } catch (e) {
      setText(q);
      toast.error("No se pudo obtener la respuesta", { description: (e as Error).message });
    } finally {
      setPending(null);
    }
  }

  function notifyAnswer(m: Message) {
    if (m.status === "error") {
      toast.error("El modelo no pudo responder", {
        id: "answer-error",
        description: m.content.replace(/^No se pudo generar la respuesta:\s*/, ""),
        action: { label: "Abrir Ajustes", onClick: () => router.push("/ajustes") },
      });
    } else if (m.status === "unverified") {
      toast.warning("Citas sin verificar", { description: "Contraste esta respuesta con los documentos antes de usarla." });
    } else if (m.status === "abstained") {
      toast.info("Sin evidencia suficiente", { description: "Pruebe a reformular la pregunta o a ampliar los documentos consultados.", duration: 4500 });
    }
  }

  async function copy(m: Message) {
    try {
      await navigator.clipboard.writeText(m.content);
      toast.success("Respuesta copiada", { id: "copied", duration: 2200 });
    } catch {
      toast.error("No se pudo copiar", { description: "El navegador bloqueó el acceso al portapapeles." });
    }
  }

  const visibleConvs = (convs ?? []).filter((c) => !hidden.has(c.id));

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-line bg-panel px-3 py-2">
        <select
          aria-label="Conversación"
          className="min-w-0 flex-1 rounded-md border border-line bg-white px-2 py-1 text-sm"
          value={convId ?? ""}
          onChange={(e) => setConvId(e.target.value || null)}
        >
          <option value="">Nueva conversación</option>
          {visibleConvs.map((c) => (
            <option key={c.id} value={c.id}>
              {c.title}
            </option>
          ))}
        </select>
        <button className="btn" onClick={() => void newConversation()}>
          <MessageSquarePlus className="h-4 w-4" /> Nueva
        </button>
        {convId && (
          <button
            className="btn-icon"
            aria-label="Eliminar conversación"
            title="Eliminar conversación"
            onClick={async () => {
              const c = (convs ?? []).find((x) => x.id === convId);
              if (!c) return;
              const ok = await toast.confirm({
                title: "¿Eliminar esta conversación?",
                message: `«${c.title}» y sus mensajes se borrarán. Tendrá unos segundos para deshacerlo.`,
                confirmLabel: "Eliminar",
                tone: "danger",
              });
              if (ok) removeConversation(c);
            }}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {loadingMsgs && (
          <div className="mx-auto flex max-w-3xl flex-col gap-4" aria-busy="true" aria-label="Cargando mensajes">
            <div className="skeleton h-10 w-2/5 self-end" />
            <div className="skeleton h-24 w-full" />
            <div className="skeleton h-10 w-1/3 self-end" />
            <div className="skeleton h-20 w-full" />
          </div>
        )}
        {!loadingMsgs && messages.length === 0 && !pending && (
          <div className="mx-auto mt-16 max-w-md animate-fade-in text-center text-ink-soft">
            <BookOpenCheck className="mx-auto mb-3 h-8 w-8 text-petrol" aria-hidden />
            <h2 className="text-base font-semibold text-ink">Pregunte a sus documentos</h2>
            <p className="mt-1 text-sm">
              Las respuestas se basan únicamente en el contenido de los PDF y cada afirmación cita su página de origen.
              {props.readyCount === 0 && " Suba un PDF para empezar."}
            </p>
          </div>
        )}

        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {!loadingMsgs &&
            messages.map((m) =>
              m.role === "user" ? (
                <div key={m.id} className="max-w-[85%] animate-fade-in self-end rounded-lg bg-petrol px-3.5 py-2 text-sm text-white">
                  <p className="whitespace-pre-wrap break-words">{m.content}</p>
                </div>
              ) : (
                <div key={m.id} className="group animate-fade-in rounded-lg border border-line bg-white px-4 py-3 text-sm">
                  {m.status === "abstained" && (
                    <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-ink-soft">
                      <Info className="h-3.5 w-3.5" aria-hidden /> Sin evidencia suficiente
                    </div>
                  )}
                  {m.status === "unverified" && (
                    <div className="mb-2 flex items-start gap-1.5 rounded bg-amber-50 px-2 py-1.5 text-xs text-warn">
                      <ShieldAlert className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden />
                      No se pudieron verificar las citas de esta respuesta. Contrástela con los documentos antes de usarla.
                    </div>
                  )}
                  {m.status === "error" ? (
                    <div className="flex items-start gap-1.5 text-bad">
                      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                      <p>{m.content}</p>
                    </div>
                  ) : (
                    <AnswerText msg={m} onOpen={props.onOpenCitation} />
                  )}

                  {m.citations.length > 0 && (
                    <div className="mt-3 border-t border-line pt-2">
                      <div className="mb-1 text-xs font-medium uppercase tracking-wide text-ink-faint">Fuentes verificadas</div>
                      <ul className="flex flex-col gap-1">
                        {m.citations.map((c) => (
                          <li key={c.label}>
                            <button
                              onClick={() => props.onOpenCitation(c)}
                              className={`w-full rounded-md border px-2.5 py-1.5 text-left transition-colors hover:border-petrol ${props.activeCitation === String(c.chunk_id) ? "border-petrol bg-petrol-tint" : "border-line"}`}
                            >
                              <div className="flex items-center gap-2 text-xs">
                                <span className="font-semibold text-petrol">{c.label}</span>
                                <span className="min-w-0 flex-1 truncate font-medium">{c.filename ?? "Documento eliminado"}</span>
                                {c.source === "caption" && <span className="badge shrink-0 bg-amber-50 text-warn" title="Texto generado por un modelo de visión: puede equivocarse">descripción automática</span>}
                                {c.source === "ocr" && <span className="badge shrink-0 bg-panel text-ink-soft" title="Texto leído de la imagen con OCR">OCR</span>}
                                {c.source === "figure_image" && <span className="badge shrink-0 bg-amber-50 text-warn" title="Figura adjunta al modelo de visión: la interpretó una máquina">imagen adjunta</span>}
                                <span className="shrink-0 text-ink-faint">
                                  {c.page_start != null && (c.page_start === c.page_end ? `p. ${c.page_start}` : `pp. ${c.page_start}-${c.page_end}`)}
                                </span>
                              </div>
                              {c.snippet && <p className="mt-0.5 line-clamp-2 text-xs text-ink-soft">{c.snippet}</p>}
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {!!m.meta?.related_media?.length && (
                    <div className="mt-3 border-t border-line pt-2">
                      <div className="mb-1 text-xs font-medium uppercase tracking-wide text-ink-faint">Medios relacionados</div>
                      <ul className="flex flex-wrap gap-2">
                        {m.meta.related_media.map((r) => (
                          <li key={r.chunk_id}>
                            <button
                              onClick={() => props.onOpenMedia(r.document_id, r.filename, r.figure != null ? r.page : null, r.figure != null ? r.chunk_id : undefined)}
                              title={`${r.filename}${r.similarity != null ? ` · similitud ${r.similarity.toFixed(2)}` : ""}`}
                              className="group/m block w-24 overflow-hidden rounded-md border border-line bg-white text-left transition-colors hover:border-petrol focus-visible:outline focus-visible:outline-2 focus-visible:outline-petrol"
                            >
                              {r.modality === "image" ? (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img src={r.figure != null ? api.figureUrl(r.chunk_id) : api.thumbUrl(r.document_id)} alt={r.filename} className="h-16 w-24 object-cover" />
                              ) : (
                                <div className="flex h-16 w-24 items-center justify-center bg-panel text-xs text-ink-faint">{r.modality}</div>
                              )}
                              <div className="truncate px-1.5 py-1 text-[11px] text-ink-soft">{r.figure != null && r.page ? `p. ${r.page} · ` : ""}{r.filename}</div>
                            </button>
                          </li>
                        ))}
                      </ul>
                      <p className="mt-1 text-[11px] text-ink-faint">Coinciden por similitud con su pregunta; el sistema no afirma nada sobre su contenido. Ábralos para comprobarlo.</p>
                    </div>
                  )}
                  <div className="mt-2 flex items-center gap-2 text-[11px] text-ink-faint">
                    {m.meta?.generation?.model && (
                      <span className="min-w-0 flex-1 truncate">
                        {m.meta.generation.provider} · {m.meta.generation.model}
                        {m.meta.generation.vision && ` · respondió mirando ${m.meta.generation.vision.images} imagen${m.meta.generation.vision.images > 1 ? "es" : ""}`}
                        {m.meta.generation.external && (m.meta.generation.vision ? " · pregunta, fragmentos e imágenes enviados a un servicio externo" : " · contexto enviado a un servicio externo")}
                      </span>
                    )}
                    {m.status !== "error" && (
                      <button
                        className="ml-auto inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-ink-faint opacity-60 transition hover:bg-panel hover:text-ink hover:opacity-100 focus-visible:opacity-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-petrol group-hover:opacity-100"
                        onClick={() => void copy(m)}
                        aria-label="Copiar respuesta"
                      >
                        <Copy className="h-3 w-3" /> Copiar
                      </button>
                    )}
                  </div>
                </div>
              ),
            )}
          {pending && (
            <>
              <div className="max-w-[85%] animate-fade-in self-end rounded-lg bg-petrol px-3.5 py-2 text-sm text-white">{pending}</div>
              <div className="flex items-center gap-2 text-sm text-ink-soft" role="status">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Buscando evidencia y redactando…
              </div>
            </>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-line bg-paper p-3">
        <div className="mx-auto max-w-3xl">
          <div className="mb-1.5 text-xs text-ink-faint">Consultando: {props.scopeLabel}</div>
          <div className="flex items-end gap-2">
            <textarea
              ref={areaRef}
              aria-label="Pregunta"
              rows={1}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder="Escriba su pregunta…  (Enter para enviar, Mayús+Enter para salto de línea)"
              className="min-h-[2.75rem] flex-1 resize-none rounded-md border border-line bg-white px-3 py-2.5 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-petrol"
            />
            <button className="btn-primary h-[2.75rem]" disabled={!text.trim() || !!pending || props.readyCount === 0} onClick={() => void send()}>
              {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} Enviar
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
