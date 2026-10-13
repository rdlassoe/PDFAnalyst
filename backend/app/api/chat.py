from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Conversation, Document, Message, MessageCitation
from ..schemas import AskIn, CitationOut, ConversationOut, HitOut, MessageOut, SearchIn, SearchOut
from ..services.answering import answer_question
from ..services.embeddings import EmbeddingError
from ..services.retrieval import retrieve

router = APIRouter(tags=["chat"])


def _citation_out(c: MessageCitation, filename: str | None) -> CitationOut:
    snippet = (c.snapshot or "")[:400] or None
    return CitationOut(
        label=c.label,
        chunk_id=c.chunk_id,
        document_id=c.document_id,
        filename=filename,
        page_start=c.page_start,
        page_end=c.page_end,
        score=c.score,
        snippet=snippet,
    )


def message_out(session: Session, m: Message) -> MessageOut:
    cits = []
    if m.citations:
        names = dict(
            session.execute(
                select(Document.id, Document.filename).where(Document.id.in_([c.document_id for c in m.citations if c.document_id]))
            ).all()
        )
        cits = [_citation_out(c, names.get(c.document_id)) for c in m.citations]
    return MessageOut(
        id=m.id, role=m.role, content=m.content, status=m.status, created_at=m.created_at, citations=cits, meta=m.meta or {}
    )


@router.post("/search", response_model=SearchOut)
def search(body: SearchIn, session: Session = Depends(get_session)):
    """Recuperación sin LLM (depuración/evaluación): muestra qué evidencia encontraría el sistema."""
    try:
        r = retrieve(
            session, body.query, document_ids=body.document_ids, collection_id=body.collection_id,
            top_k=body.top_k, hybrid=body.hybrid,
        )
    except EmbeddingError as exc:
        raise HTTPException(503, str(exc)) from exc
    return SearchOut(
        hits=[HitOut(**{k: getattr(h, k) for k in HitOut.model_fields}) for h in r.hits],
        considered=r.candidates_considered,
        abstain_reason=r.abstain_reason,
        terms=r.terms,
    )


@router.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(session: Session = Depends(get_session)):
    c = Conversation()
    session.add(c)
    session.commit()
    return c


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(session: Session = Depends(get_session)):
    return session.scalars(select(Conversation).order_by(Conversation.updated_at.desc()).limit(100)).all()


@router.delete("/conversations/{cid}", status_code=204)
def delete_conversation(cid: uuid.UUID, session: Session = Depends(get_session)):
    c = session.get(Conversation, cid)
    if c is None:
        raise HTTPException(404, "Conversación no encontrada")
    session.delete(c)
    session.commit()


@router.get("/conversations/{cid}/messages", response_model=list[MessageOut])
def list_messages(cid: uuid.UUID, session: Session = Depends(get_session)):
    c = session.get(Conversation, cid)
    if c is None:
        raise HTTPException(404, "Conversación no encontrada")
    return [message_out(session, m) for m in c.messages]


@router.post("/conversations/{cid}/messages", response_model=MessageOut)
def ask(cid: uuid.UUID, body: AskIn, session: Session = Depends(get_session)):
    c = session.get(Conversation, cid)
    if c is None:
        raise HTTPException(404, "Conversación no encontrada")
    try:
        result = answer_question(
            session, c, body.question, document_ids=body.document_ids, collection_id=body.collection_id
        )
    except EmbeddingError as exc:
        raise HTTPException(503, f"Modelo de embeddings no disponible: {exc}") from exc
    return message_out(session, result.assistant_message)
