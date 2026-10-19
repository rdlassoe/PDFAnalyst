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
  media_type: "pdf" | "image" | "audio" | "video";
  mime: string | null;
  width: number | null;
  height: number | null;
  duration_ms: number | null;
  thumbnail_url: string | null;
  description: string | null; // descripción automática de una imagen (si hay modelo de visión)
  figure_count: number | null; // figuras detectadas dentro de un PDF
  figures_described: number | null;
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
  modality: string | null;
  source: string | null; // caption = descripción automática de una imagen; ocr; asr; pdf_text
  start_ms: number | null;
  end_ms: number | null;
}

export interface RelatedMedia {
  chunk_id: number;
  document_id: string;
  filename: string;
  modality: string;
  similarity: number | null;
  start_ms: number | null;
  end_ms: number | null;
  page: number | null;
  figure: number | null; // n.º de figura dentro de un PDF (null = imagen suelta)
}

export interface ImageAnswer {
  text: string;
  model: string;
  provider: string;
  external: boolean;
  automatic: boolean;
}

export type MsgStatus = "ok" | "abstained" | "unverified" | "error";

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  status: MsgStatus;
  created_at: string;
  citations: Citation[];
  meta: {
    generation?: { provider?: string; model?: string; external?: boolean; vision?: { images: number; labels: string[] }; vision_error?: string };
    related_media?: RelatedMedia[];
  };
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
  embeddings: {
    ok: boolean;
    provider: string | null;
    model: string | null;
    external: boolean | null;
    dimension?: number | null;
    documents_total?: number;
    documents_indexed?: number;
    detail?: string;
  };
  generation: { ok: boolean; provider?: string; model?: string; external?: boolean; detail?: string };
}

export interface Chunk {
  id: number;
  document_id: string;
  filename: string;
  page_start: number;
  page_end: number;
  text: string;
  modality?: string;
  source?: string;
  figure_url?: string | null; // el fragmento pertenece a una figura de un PDF
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

// ---- Modelo de visión (describe imágenes; Gemini, Groq, Ollama…). Mismos perfiles que el LLM pero independientes: otro modelo, desactivado por defecto («none»).
export interface VisionPreset extends LlmPreset {
  model: string;
}

export interface VisionSettings {
  active: string; // id de perfil o «none»
  profiles: LlmProfile[];
  presets: VisionPreset[];
  answer_with_images: boolean; // responder MIRANDO las figuras con este modelo
}

export interface VisionImageTest {
  ok: boolean;
  detail: string;
  description: string | null;
  external: boolean | null;
  model?: string;
}

export type EmbedKind = "sentence-transformers" | "ollama" | "openai-compatible";

export interface EmbedProfile {
  id: string;
  name: string;
  kind: EmbedKind;
  base_url: string;
  model: string;
  has_key: boolean;
  key_hint: string | null;
  key_error: boolean;
  allow_external: boolean;
  external: boolean;
  editable: boolean;
  active: boolean;
  query_prefix: string;
  passage_prefix: string;
  dimensions: number | null;
  /** Umbral coseno de relevancia propio de este modelo (null = el global del servidor). */
  min_similarity: number | null;
  /** El servicio (local) embebe imágenes, audio y vídeo. */
  supports_media: boolean;
}

export interface EmbedPreset {
  id: string;
  label: string;
  kind: Exclude<EmbedKind, "sentence-transformers">;
  base_url: string;
  model: string;
  query_prefix: string;
  passage_prefix: string;
  dimensions: number | null;
  min_similarity: number | null;
  supports_media: boolean;
  needs_key: boolean;
  key_url: string | null;
  note: string;
}

export interface EmbedModelUsage {
  id: number;
  provider: string;
  name: string;
  dimension: number;
  vectors: number;
  active: boolean;
}

export interface EmbedSettings {
  active: string;
  profiles: EmbedProfile[];
  presets: EmbedPreset[];
  models: EmbedModelUsage[];
  coverage: { documents_total: number; documents_indexed: number };
}

export interface EmbedPreview {
  name: string;
  kind: EmbedKind;
  model: string;
  host: string;
  external: boolean;
  allow_external: boolean;
  documents_total: number;
  documents_to_index: number;
  already_indexed: number;
  chunks_to_index: number;
  chars_to_send: number;
}

export interface EmbedActivation {
  documents: number;
  already_indexed: number;
  queued: number;
  model: string;
  dimension: number;
}

export interface EmbedTestResult {
  ok: boolean;
  detail: string;
  dimension: number | null;
  latency_ms: number | null;
}

export interface EmbedProfileInput {
  name: string;
  kind: Exclude<EmbedKind, "sentence-transformers">;
  base_url: string;
  model: string;
  api_key?: string | null;
  clear_api_key?: boolean;
  allow_external?: boolean;
  query_prefix?: string;
  passage_prefix?: string;
  dimensions?: number | null;
  min_similarity?: number | null;
  supports_media?: boolean;
}

/** Réplica de la regla del servidor solo para decidir si mostrar el aviso de consentimiento (el servidor es quien decide). */
export function isLocalUrl(url: string): boolean {
  try {
    const h = new URL(url).hostname.toLowerCase().replace(/^\[|\]$/g, "");
    if (["localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama", "embedder", "db"].includes(h) || h.endsWith(".localhost")) return true;
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
  thumbUrl: (id: string) => `${API}/documents/${id}/thumbnail`,
  figureUrl: (chunkId: number) => `${API}/chunks/${chunkId}/figure`,
  describeDoc: (id: string, question: string) => req<ImageAnswer>(`/documents/${id}/describe`, json("POST", { question })),
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
  visionSettings: () => req<VisionSettings>("/settings/vision"),
  createVisionProfile: (body: LlmProfileInput) => req<VisionSettings>("/settings/vision/profiles", json("POST", body)),
  updateVisionProfile: (id: string, body: Partial<LlmProfileInput>) => req<VisionSettings>(`/settings/vision/profiles/${id}`, json("PATCH", body)),
  deleteVisionProfile: (id: string) => req<VisionSettings>(`/settings/vision/profiles/${id}`, { method: "DELETE" }),
  setActiveVision: (profile_id: string) => req<VisionSettings>("/settings/vision/active", json("PUT", { profile_id })),
  setVisionAnswer: (enabled: boolean) => req<VisionSettings>("/settings/vision/answer", json("PUT", { enabled })),
  testVision: (body: { profile_id?: string | null; kind?: LlmKind; base_url?: string; api_key?: string | null }) =>
    req<LlmTestResult>("/settings/vision/test", json("POST", body)),
  testVisionImage: (profile_id: string) => req<VisionImageTest>("/settings/vision/test-image", json("POST", { profile_id })),
  embedSettings: () => req<EmbedSettings>("/settings/embeddings"),
  createEmbedProfile: (body: EmbedProfileInput) => req<EmbedSettings>("/settings/embeddings/profiles", json("POST", body)),
  updateEmbedProfile: (id: string, body: Partial<EmbedProfileInput>) =>
    req<EmbedSettings>(`/settings/embeddings/profiles/${id}`, json("PATCH", body)),
  deleteEmbedProfile: (id: string) => req<EmbedSettings>(`/settings/embeddings/profiles/${id}`, { method: "DELETE" }),
  previewEmbed: (profile_id: string) => req<EmbedPreview>("/settings/embeddings/preview", json("POST", { profile_id })),
  activateEmbed: (profile_id: string) =>
    req<EmbedSettings & { activation: EmbedActivation }>("/settings/embeddings/active", json("PUT", { profile_id })),
  testEmbed: (body: { profile_id?: string | null; kind?: string; base_url?: string; model?: string; api_key?: string | null; query_prefix?: string; dimensions?: number | null }) =>
    req<EmbedTestResult>("/settings/embeddings/test", json("POST", body)),
  purgeEmbedModel: (id: number) => req<EmbedSettings>(`/settings/embeddings/models/${id}/vectors`, { method: "DELETE" }),
  ask: (id: string, question: string, documentIds: string[], collectionId: string | null) =>
    req<Message>(`/conversations/${id}/messages`, json("POST", {
      question,
      document_ids: documentIds.length ? documentIds : null,
      collection_id: collectionId,
    })),
};
