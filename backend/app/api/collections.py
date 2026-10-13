from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Collection, CollectionDocument, Document
from ..schemas import CollectionIn, CollectionOut

router = APIRouter(prefix="/collections", tags=["collections"])


def _out(session: Session, c: Collection) -> CollectionOut:
    ids = session.scalars(select(CollectionDocument.document_id).where(CollectionDocument.collection_id == c.id)).all()
    return CollectionOut(id=c.id, name=c.name, description=c.description, document_ids=list(ids))


def _get(session: Session, cid: uuid.UUID) -> Collection:
    c = session.get(Collection, cid)
    if c is None:
        raise HTTPException(404, "Colección no encontrada")
    return c


@router.post("", response_model=CollectionOut, status_code=201)
def create_collection(body: CollectionIn, session: Session = Depends(get_session)):
    c = Collection(name=body.name.strip(), description=body.description)
    session.add(c)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(409, "Ya existe una colección con ese nombre")
    return _out(session, c)


@router.get("", response_model=list[CollectionOut])
def list_collections(session: Session = Depends(get_session)):
    return [_out(session, c) for c in session.scalars(select(Collection).order_by(Collection.name))]


@router.delete("/{cid}", status_code=204)
def delete_collection(cid: uuid.UUID, session: Session = Depends(get_session)):
    session.delete(_get(session, cid))
    session.commit()


@router.put("/{cid}/documents/{doc_id}", response_model=CollectionOut)
def add_document(cid: uuid.UUID, doc_id: uuid.UUID, session: Session = Depends(get_session)):
    c = _get(session, cid)
    if session.get(Document, doc_id) is None:
        raise HTTPException(404, "Documento no encontrado")
    if session.get(CollectionDocument, (cid, doc_id)) is None:
        session.add(CollectionDocument(collection_id=cid, document_id=doc_id))
        session.commit()
    return _out(session, c)


@router.delete("/{cid}/documents/{doc_id}", response_model=CollectionOut)
def remove_document(cid: uuid.UUID, doc_id: uuid.UUID, session: Session = Depends(get_session)):
    c = _get(session, cid)
    link = session.get(CollectionDocument, (cid, doc_id))
    if link:
        session.delete(link)
        session.commit()
    return _out(session, c)
