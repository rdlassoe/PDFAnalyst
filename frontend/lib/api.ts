export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type DocStatus = "uploaded" | "processing" | "ready" | "failed";

export interface Doc {
  id: string;
  filename: string;
  size_bytes: number;
  page_count: number | null;
  title: string | null;
  status: DocStatus;
  error: string | null;
  created_at: string;
  chunk_count: number | null;
  ocr_pages: number | null;
  warnings: string[];
}

export interface Collection {
  id: string;
  name: string;
  description: string | null;
  document_ids: string[];
}

export interface Citation {
  label: string;
  chunk_id: number | null;
  document_id: string | null;
  filename: string | null;
  page_start: number | null;
  page_end: number | null;
  score: number | null;
  snippet: string | null;
}

export type MsgStatus = "ok" | "abstained" | "unverified" | "error";

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  status: MsgStatus;
  created_at: string;
  citations: Citation[];
  meta: { generation?: { provider?: string; model?: string; external?: boolean } };
}

export interface Conversation {
  id: string;
  title: string;
  updated_at: string;
}

export interface Health {
  ok: boolean;
  database: { ok: boolean };
  pgvector: { ok: boolean; version?: string };
  embeddings: { provider: string; model: string };
  generation: { ok: boolean; provider?: string; model?: string; external?: boolean; detail?: string };
}

export interface Chunk {
  id: number;
  document_id: string;
  filename: string;
  page_start: number;
  page_end: number;
  text: string;
}

export type LlmKind = "ollama" | "openai-compatible";

export interface LlmProfile {
  id: string;
  name: string;
  kind: LlmKind;
  base_url: string;
  model: string;
  has_key: boolean;
  key_hint: string | null;
  key_error: boolean;
  allow_external: boolean;
  external: boolean;
  editable: boolean;
  active: boolean;
}

export interface LlmPreset {
  id: string;
  label: string;
  kind: LlmKind;
  base_url: string;
  needs_key: boolean;
  key_url: string | null;
  note: string;
}

export interface LlmSettings {
  active: string;
  profiles: LlmProfile[];
  presets: LlmPreset[];
}

export interface LlmTestResult {
  ok: boolean;
  detail: string;
  models: string[];
}

export interface LlmProfileInput {
  name: string;
  kind: LlmKind;
  base_url: string;
  model: string;
  api_key?: string | null;
  clear_api_key?: boolean;
  allow_external?: boolean;
}

/** Réplica de la regla del servidor solo para decidir si mostrar el aviso de consentimiento (el servidor es quien decide). */
export function isLocalUrl(url: string): boolean {
  try {
    const h = new URL(url).hostname.toLowerCase().replace(/^\[|\]$/g, "");
    if (["localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama", "db"].includes(h) || h.endsWith(".localhost")) return true;
    const m = h.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
    if (m) {
      const a = +m[1], b = +m[2];
      return a === 10 || a === 127 || (a === 192 && b === 168) || (a === 172 && b >= 16 && b <= 31);
    }
    return false;
  } catch {
    return false;
  }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, init);
  } catch {
    throw new ApiError(0, "No se pudo conectar con la API de PDF Analyst");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* respuesta sin JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/health"),
  documents: () => req<Doc[]>("/documents"),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<{ document: Doc; created: boolean }>("/documents", { method: "POST", body: fd });
  },
  deleteDocument: (id: string) => req<void>(`/documents/${id}`, { method: "DELETE" }),
  reprocess: (id: string) => req<unknown>(`/documents/${id}/reprocess`, { method: "POST" }),
  fileUrl: (id: string) => `${API}/documents/${id}/file`,
  chunk: (id: number) => req<Chunk>(`/chunks/${id}`),
  collections: () => req<Collection[]>("/collections"),
  createCollection: (name: string) => req<Collection>("/collections", json("POST", { name })),
  deleteCollection: (id: string) => req<void>(`/collections/${id}`, { method: "DELETE" }),
  addToCollection: (cid: string, did: string) => req<Collection>(`/collections/${cid}/documents/${did}`, { method: "PUT" }),
  removeFromCollection: (cid: string, did: string) =>
    req<Collection>(`/collections/${cid}/documents/${did}`, { method: "DELETE" }),
  conversations: () => req<Conversation[]>("/conversations"),
  createConversation: () => req<Conversation>("/conversations", { method: "POST" }),
  deleteConversation: (id: string, keepalive = false) => req<void>(`/conversations/${id}`, { method: "DELETE", keepalive }),
  messages: (id: string) => req<Message[]>(`/conversations/${id}/messages`),
  llmSettings: () => req<LlmSettings>("/settings/llm"),
  createLlmProfile: (body: LlmProfileInput) => req<LlmSettings>("/settings/llm/profiles", json("POST", body)),
  updateLlmProfile: (id: string, body: Partial<LlmProfileInput>) => req<LlmSettings>(`/settings/llm/profiles/${id}`, json("PATCH", body)),
  deleteLlmProfile: (id: string) => req<LlmSettings>(`/settings/llm/profiles/${id}`, { method: "DELETE" }),
  setActiveLlm: (profile_id: string) => req<LlmSettings>("/settings/llm/active", json("PUT", { profile_id })),
  testLlm: (body: { profile_id?: string | null; kind?: LlmKind; base_url?: string; api_key?: string | null }) =>
    req<LlmTestResult>("/settings/llm/test", json("POST", body)),
  ask: (id: string, question: string, documentIds: string[], collectionId: string | null) =>
    req<Message>(`/conversations/${id}/messages`, json("POST", {
      question,
      document_ids: documentIds.length ? documentIds : null,
      collection_id: collectionId,
    })),
};
