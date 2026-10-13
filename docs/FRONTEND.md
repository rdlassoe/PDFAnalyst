# Frontend de PDF Analyst

Guía para quien use o modifique la interfaz. Stack: Next.js 15 (App Router) · React 19 · TypeScript estricto · Tailwind CSS 3 ·
Lucide Icons · PDF.js (`pdfjs-dist`). Todo es código abierto (ver `docs/LICENSES.md`).

## 1. Estructura

```
frontend/
├── app/
│   ├── layout.tsx          # <html lang="es"> + <Providers> (notificaciones)
│   ├── page.tsx            # pantalla principal: documentos · conversación · visor
│   ├── ajustes/page.tsx    # Ajustes: proveedor de LLM
│   └── globals.css         # Tailwind + clases de componentes (.btn, .btn-primary, .btn-icon, .badge, .skeleton)
├── components/
│   ├── Toaster.tsx         # notificaciones flotantes + diálogo de confirmación (useToast)
│   ├── Providers.tsx       # monta <ToastProvider> (componente de cliente)
│   ├── DocumentPanel.tsx   # subir/listar/eliminar documentos, colecciones
│   ├── Chat.tsx            # conversaciones, mensajes, citas, copiar, eliminar con «Deshacer»
│   ├── PdfViewer.tsx       # visor PDF.js con resaltado del fragmento citado
│   └── LlmSettingsPanel.tsx# perfiles de proveedor LLM (alta, edición, prueba, activación)
├── lib/api.ts              # cliente tipado de la API + tipos + isLocalUrl
├── tailwind.config.ts      # paleta, animaciones
└── Dockerfile              # build multi-etapa (standalone)
```

No hay gestor de estado global: cada pantalla guarda su estado con hooks; `page.tsx` orquesta documentos, selección, visor y salud.

## 2. Identidad visual

Sobria y profesional (requisito del proyecto): fondo blanco cálido (`paper`), paneles gris claro (`panel`), texto gris carbón (`ink`),
azul petróleo (`petrol`) como único color de acción, bordes discretos (`line`). Verde/ámbar/rojo solo para estados. Sin degradados,
neones ni brillos; las animaciones son breves (≤ 220 ms) y funcionales. Con `prefers-reduced-motion` se reducen a ~1 ms.

## 3. Notificaciones y confirmaciones (`components/Toaster.tsx`)

Se usa con el hook `useToast()` dentro de cualquier componente de cliente (el proveedor está en `layout.tsx`).

```tsx
const toast = useToast();

toast.success("Documento subido", { description: "informe.pdf" });
toast.error("No se pudo guardar", { description: err.message });
toast.warning("Citas sin verificar");
toast.info("Sin evidencia suficiente", { duration: 4500 });

// Estado de carga que se transforma en el resultado (mismo aviso, sin parpadeo)
await toast.promise(api.upload(file), {
  loading: "Subiendo…",
  success: (r) => ({ title: "Subido", description: r.document.filename }),   // también admite { kind: "info" }
  error: (e) => ({ title: "No se pudo subir", description: e.message }),
});

// Acción en el aviso + callback único al desaparecer (con el motivo)
toast.info("Conversación eliminada", {
  id: "del-123", duration: 7000,
  action: { label: "Deshacer", onClick: restore },
  onDismiss: (reason) => { if (reason !== "action") commitDelete(); },
});

// Confirmación modal (devuelve Promise<boolean>)
if (await toast.confirm({ title: "¿Eliminar?", message: "…", confirmLabel: "Eliminar", tone: "danger" })) { … }
```

| Opción | Significado |
|--------|-------------|
| `id` | Si ya existe un aviso con ese `id`, se **reemplaza en el sitio** (útil para estados y para evitar duplicados). |
| `duration` | ms visibles; `0` = permanece hasta cerrarse. Por defecto: éxito 4 s · info 4,5 s · aviso 7 s · error 9 s · cargando ∞. |
| `description` | Segunda línea (detalle). |
| `action` | Un botón en el aviso; al pulsarlo se ejecuta y el aviso se cierra con motivo `"action"`. |
| `onDismiss(reason)` | Se llama **una sola vez**. Motivos: `timeout`, `close` (botón ×), `action`, `replaced` (reemplazado o desbordado), `api` (cerrado por código o por cierre de página). |

Comportamiento: máximo 5 avisos visibles (se retiran los más antiguos); la barra de progreso es una animación CSS cuyo fin cierra el
aviso, y se **pausa** al pasar el ratón o enfocar; región `aria-live="polite"` y `role="alert"` en los errores; el diálogo de
confirmación tiene `role="alertdialog"`, foco inicial en «Cancelar», `Esc` cancela, trampa de foco entre sus dos botones y restaura
el foco al cerrar.

**Convenciones**

1. No usar `alert()`/`confirm()` del navegador: usar `toast.confirm`.
2. Todo error de red se notifica con el detalle del servidor (`ApiError.message`); las acciones correctas se confirman con un aviso breve.
3. Acciones destructivas: confirmar con `tone: "danger"` explicando las consecuencias. Si es posible, preferir **eliminación diferida con
   «Deshacer»** (patrón de `Chat.removeConversation`).
4. Para avisos que describen un **estado** (API caída, modelo caído) usar un `id` fijo, para que se actualicen en lugar de acumularse.

### Eliminación diferida (conversaciones)

1. El usuario confirma → la conversación se oculta de inmediato (estado local `hidden`) y se muestra el aviso con «Deshacer» (7 s).
2. «Deshacer» la vuelve a mostrar y **no** llama a la API. Cualquier otro cierre (expira, ×, desbordamiento) ejecuta `DELETE /conversations/{id}`.
3. Si se cierra o recarga la página con un aviso pendiente, el proveedor lo cierra con motivo `api` y el borrado se envía con
   `fetch(..., { keepalive: true })`. Si aun así el navegador lo cancelara, la conversación simplemente no se borra (fallo seguro).
4. Si el borrado falla, la conversación reaparece y se muestra un error.

## 4. Pantallas

### Principal (`/`)

- **Cabecera:** estado de PostgreSQL+pgvector, embeddings y generación (punto verde/rojo) y botón **Ajustes**.
- **Documentos:** subida con arrastrar y soltar (solo PDF; los demás archivos se omiten con aviso), estado por documento
  (en cola · procesando · listo · error), casilla para limitar la consulta, colecciones, reintento y eliminación con confirmación.
  Al terminar el procesamiento de un documento se notifica con acción «Abrir» (o «Reintentar» si falló). Esqueletos mientras carga.
- **Conversación:** respuestas con etiquetas `[C1]` clicables y lista de «Fuentes verificadas»; estados visibles (sin evidencia, citas sin
  verificar, error); **Copiar** respuesta; área de texto que crece (Enter envía, Mayús+Enter salto de línea); proveedor y modelo usados.
- **Visor PDF:** abre la página citada y resalta los fragmentos de la capa de texto que pertenecen al chunk citado (si es un PDF escaneado
  o la maquetación impide localizarlo, lo indica). Zoom, páginas, flechas ← → para navegar. Se ajusta al ancho en pantallas pequeñas.
- Si se elimina el documento que se está viendo o seleccionando, el estado se limpia solo.

### Ajustes (`/ajustes`)

Gestión de proveedores LLM: lista (con insignias *Predeterminado*, *Local/Servicio externo*, *Definido en .env*), **Probar** conexión,
**Usar por defecto**, alta/edición con plantillas (Ollama, Groq, Gemini, OpenAI, LM Studio…), **Cargar modelos** (consulta al proveedor), clave
API (campo de contraseña; una clave guardada nunca se muestra, solo «…1234»; vacío = conservar; casilla para eliminarla) y el aviso de
consentimiento para servicios externos. Detalle de seguridad en README §«Proveedor de generación» y `docs/OPERATIONS.md` §7.

## 5. Diseño adaptable

| Ancho | Diseño |
|-------|--------|
| ≥ 1024 px (`lg`) | Tres columnas: documentos (300 px) · conversación · visor. |
| < 1024 px | Pestañas «Documentos / Conversación» bajo la cabecera; el visor es un **panel deslizante** a pantalla completa (se abre al pulsar una cita o un documento; botón × y `Esc` lo cierran). Altura con `h-dvh`. |
| < 768 px | Se ocultan las etiquetas de estado de la cabecera (queda el botón de Ajustes). |

## 6. Desarrollo

```bash
cd frontend && npm install && npm run dev        # http://localhost:3000, necesita la API en NEXT_PUBLIC_API_URL
npm run typecheck                                # tsc --noEmit (también se ejecuta dentro de `next build`)
docker compose build frontend && docker compose up -d frontend   # reconstruir la imagen tras cambios
```

`NEXT_PUBLIC_API_URL` (por defecto `http://localhost:8000`) se **incorpora en el build**: para cambiarla hay que reconstruir
(`PUBLIC_API_URL` en `.env`, usada por el compose como argumento de build). Añada el origen del frontend a `CORS_ORIGINS` si no es
`http://localhost:3000`.

Notas de implementación que evitan errores ya vistos:

- Un `useEffect` **no debe devolver** el resultado de una llamada (p. ej. `scrollIntoView`): React lo interpreta como función de limpieza
  y falla con «u is not a function». Usar cuerpo de bloque.
- `pdfjs-dist` se importa dinámicamente (toca APIs del navegador al cargarse).
- `localhost` puede resolver primero a IPv6; los puertos se publican en `127.0.0.1` (IPv4) y el navegador cae a IPv4 sin problema.

## 7. Verificación

**No hay pruebas automáticas del frontend** (limitación conocida). Se verificó manualmente en el navegador integrado y con el chequeo de
tipos de `next build`. Lista de comprobación tras cambios en la interfaz:

- [ ] Eliminar una conversación **de prueba**: aparece el diálogo; tras confirmar, el aviso con «Deshacer»; «Deshacer» la restaura (la API
      aún la lista); si se deja expirar, la API ya no la lista.
- [ ] Subir: un `.txt` (aviso de omitido), un PDF corrupto (error con motivo), el mismo PDF dos veces (info «Ya estaba cargado»), un PDF nuevo
      (éxito y, al terminar, «está listo» con «Abrir»).
- [ ] Parar y arrancar la API: aparece «Sin conexión con la API» (persistente) y luego «Conexión restablecida».
- [ ] Preguntar con el proveedor caído: aviso de error con «Abrir Ajustes».
- [ ] Ajustes: probar, cargar modelos, guardar/activar un perfil local; un servicio externo sin permiso no se puede activar; eliminar con confirmación.
- [ ] Vista móvil (375 px): pestañas, visor deslizante ajustado al ancho, sin desbordamiento horizontal; `Esc` cierra el visor.
- [ ] Consola del navegador sin errores inesperados.

> Al probar contra una instalación con datos reales, use conversaciones, documentos y proveedores **desechables**: el borrado es real.
