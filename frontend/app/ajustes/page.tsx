"use client";

import { ArrowLeft, Cpu, Database, Settings } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import EmbeddingSettingsPanel from "@/components/EmbeddingSettingsPanel";
import LlmSettingsPanel from "@/components/LlmSettingsPanel";
import { api, type Health } from "@/lib/api";

export default function AjustesPage() {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="min-h-screen">
      <header className="flex items-center gap-3 border-b border-line bg-paper px-4 py-2">
        <Link href="/" className="btn" aria-label="Volver a la aplicación">
          <ArrowLeft className="h-4 w-4" /> Volver
        </Link>
        <div className="flex items-center gap-2">
          <Settings className="h-5 w-5 text-petrol" aria-hidden />
          <h1 className="text-base font-semibold tracking-tight">Ajustes</h1>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 py-6">
        <section aria-labelledby="llm-h">
          <h2 id="llm-h" className="flex items-center gap-2 text-lg font-semibold">
            <Cpu className="h-5 w-5 text-petrol" aria-hidden /> Modelo de lenguaje (generación de respuestas)
          </h2>
          <p className="mb-4 mt-1 text-sm text-ink-soft">
            Elija qué LLM redacta las respuestas. Por defecto se usa Ollama en su equipo. Puede añadir otras APIs (Groq, Gemini,
            OpenAI u otra compatible), probarlas y cambiar el proveedor predeterminado en cualquier momento; el cambio se aplica a la
            siguiente pregunta, sin reiniciar y sin reprocesar documentos.
          </p>
          {health && !health.generation.ok && (
            <p role="status" className="mb-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-warn">
              El proveedor activo no responde ahora mismo: {health.generation.detail}
            </p>
          )}
          <LlmSettingsPanel />
        </section>

        <section aria-labelledby="emb-h" className="mt-10 border-t border-line pt-6">
          <h2 id="emb-h" className="flex items-center gap-2 text-lg font-semibold">
            <Database className="h-5 w-5 text-petrol" aria-hidden /> Modelo de embeddings (búsqueda e indexación)
          </h2>
          <p className="mb-4 mt-1 text-sm text-ink-soft">
            Convierte el texto de los documentos y sus preguntas en vectores para buscar. Es independiente del modelo de lenguaje. Modelo activo:{" "}
            <strong>{health?.embeddings.model ?? "…"}</strong>
            {health?.embeddings.external ? " (servicio externo)" : health?.embeddings.model ? " (local)" : ""}. Puede usar otro, incluso un servicio externo,
            sabiendo que <strong>un servicio externo recibe el texto íntegro de sus documentos</strong>.
          </p>
          <EmbeddingSettingsPanel />
        </section>
      </main>
    </div>
  );
}
