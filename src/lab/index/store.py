# -----------------------------------------------------------------------------
# File:     src/lab/index/store.py
# Purpose:  Qdrant collection for the chunks: build, metadata and vector search.
# Author:   Alexandre
# Created:  2026-09-28
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Coleção vetorial no Qdrant (SPEC 4.4).

Onde fica o índice:

- ``QDRANT_URL`` (e ``QDRANT_API_KEY``) no ambiente: Qdrant Cloud;
- sem isso: Qdrant **local em arquivo** (``LAB_QDRANT_PATH``, padrão
  ``.cache/qdrant``), que a própria biblioteca executa, sem servidor;
- testes: ``":memory:"``.

Cada ponto é um :class:`~lab.index.chunking.Chunk` com o vetor do
``embed_text`` e o chunk inteiro no payload, mais índices de payload em ``law``
e ``article_id`` para filtrar (ex.: filtro por norma, #44); no modo local esses
índices não existem, e o filtro funciona do mesmo jeito, só sem acelerar. O id do ponto é um
UUID derivado do id do chunk, então reindexar sobrescreve em vez de duplicar.

O nome da coleção junta granularidade e modelo (``lab_article_bge-m3``): mudar
qualquer um dos dois cria outra coleção, e resultados de runs antigas continuam
apontando para a coleção que usaram.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from lab.index.chunking import Chunk

_NAMESPACE = uuid.UUID("7f0c3f0e-6a55-4d6b-9e3b-5a1d2c0b9e11")


class IndexNotReadyError(RuntimeError):
    """Índice ausente ou incompatível (ex.: coleção não criada, dimensão errada)."""


class Hit(BaseModel):
    """Um chunk recuperado e a similaridade (cosseno) com a pergunta."""

    chunk: Chunk
    score: float


def open_client(url: str | None, api_key: str | None, path: Path) -> QdrantClient:
    """Cliente do Qdrant: Cloud se houver ``url``, senão local em ``path``."""
    if url:
        return QdrantClient(url=url, api_key=api_key or None)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def open_index(url: str | None, api_key: str | None, path: Path, collection: str) -> VectorIndex:
    """Abrir a coleção ``collection`` (Cloud ou local). Feche com ``close`` ou ``with``."""
    return VectorIndex(open_client(url, api_key, path), collection, remote=bool(url))


def collection_name(granularity: str, model: str) -> str:
    """``lab_<granularidade>_<modelo>`` só com caracteres seguros (``lab_article_bge-m3``)."""
    slug = re.sub(r"[^a-z0-9.-]+", "-", model.split("/")[-1].lower()).strip("-")
    return f"lab_{granularity}_{slug}"


def point_id(chunk_id: str) -> str:
    """UUID estável para o id do chunk (o Qdrant só aceita inteiro ou UUID)."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class VectorIndex:
    """Uma coleção do Qdrant com os chunks do corpus."""

    def __init__(self, client: QdrantClient, collection: str, *, remote: bool = False) -> None:
        self.client = client
        self.collection = collection
        self.remote = remote  # índices de payload só têm efeito no servidor

    def close(self) -> None:
        """Liberar o cliente (no modo local, solta o lock do arquivo)."""
        self.client.close()

    def __enter__(self) -> VectorIndex:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection).count if self.exists() else 0

    def rebuild(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Recriar a coleção com ``chunks`` e seus vetores. Devolve quantos pontos gravou.

        :raises ValueError: número de vetores diferente do de chunks, ou vazio.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks para {len(vectors)} vetores")
        if not chunks:
            raise ValueError("nenhum chunk para indexar")
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=len(vectors[0]), distance=models.Distance.COSINE
            ),
        )
        if self.remote:
            for field in ("law", "article_id"):
                self.client.create_payload_index(
                    self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        points = [
            models.PointStruct(id=point_id(c.id), vector=list(v), payload=c.model_dump())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[start : start + 128])
        return len(points)

    def search(
        self, vector: Sequence[float], k: int, *, laws: Sequence[str] | None = None
    ) -> list[Hit]:
        """Os ``k`` chunks mais próximos de ``vector``, do mais ao menos similar.

        :param laws: se informado, só chunks dessas normas.
        :raises IndexNotReadyError: a coleção não existe (rode ``lab index``).
        """
        if not self.exists():
            raise IndexNotReadyError(f"coleção {self.collection!r} não existe; rode `lab index`")
        query_filter = None
        if laws:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="law", match=models.MatchAny(any=list(laws)))]
            )
        response = self.client.query_points(
            self.collection,
            query=list(vector),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [Hit(chunk=Chunk.model_validate(p.payload), score=p.score) for p in response.points]


__all__ = [
    "Hit",
    "IndexNotReadyError",
    "VectorIndex",
    "collection_name",
    "open_client",
    "open_index",
    "point_id",
]
