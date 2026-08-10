from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.services.schemas.retrieval import RetrievalResult
from app.services.schemas.vector import VectorRecord
from app.services.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant-backed implementation of the VectorStore contract."""

    def __init__(
        self,
        client: QdrantClient,
        collection_prefix: str = "lexisai",
    ):
        self._client = client
        self._collection_prefix = collection_prefix

    def upsert(
        self,
        index_id: str,
        records: list[VectorRecord],
    ) -> None:
        if not records:
            return

        self._ensure_collection(
            index_id=index_id,
            dimensions=len(records[0].vector),
        )

        points = [
            models.PointStruct(
                id=str(record.chunk_id),
                vector=record.vector,
                payload={
                    "chunk_id": str(record.chunk_id),
                    "document_id": str(record.document_id),
                    "organization_id": str(
                        record.organization_id,
                    ),
                    "page_number": record.page_number,
                    "chunk_index": record.chunk_index,
                    "content": record.content,
                },
            )
            for record in records
        ]

        self._client.upsert(
            collection_name=self._collection_name(index_id),
            points=points,
        )

    def search(
        self,
        index_id: str,
        query_vector: list[float],
        organization_id: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        collection_name = self._collection_name(index_id)

        if not self._client.collection_exists(
            collection_name,
        ):
            return []

        conditions = [
            models.FieldCondition(
                key="organization_id",
                match=models.MatchValue(
                    value=organization_id,
                ),
            ),
        ]

        if document_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id,
                    ),
                )
            )

        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=conditions,
            ),
            limit=top_k,
            with_payload=True,
        )

        results: list[RetrievalResult] = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                RetrievalResult(
                    document_id=uuid.UUID(
                        str(payload["document_id"]),
                    ),
                    chunk_id=uuid.UUID(
                        str(payload["chunk_id"]),
                    ),
                    page_number=int(
                        payload["page_number"],
                    ),
                    chunk_index=int(
                        payload["chunk_index"],
                    ),
                    content=str(
                        payload["content"],
                    ),
                    score=float(point.score),
                )
            )

        return results

    def delete(
        self,
        index_id: str,
        chunk_ids: list[str],
    ) -> None:
        if not chunk_ids:
            return

        collection_name = self._collection_name(index_id)

        if not self._client.collection_exists(
            collection_name,
        ):
            return

        self._client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=chunk_ids,
            ),
        )

    def delete_by_document_id(
        self,
        index_id: str,
        document_id: str,
    ) -> None:
        collection_name = self._collection_name(index_id)

        if not self._client.collection_exists(
            collection_name,
        ):
            return

        self._client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(
                                value=document_id,
                            ),
                        ),
                    ],
                ),
            ),
        )

    def _ensure_collection(
        self,
        index_id: str,
        dimensions: int,
    ) -> None:
        collection_name = self._collection_name(index_id)

        if self._client.collection_exists(
            collection_name,
        ):
            return

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    def _collection_name(
        self,
        index_id: str,
    ) -> str:
        normalized_index_id = str(index_id).replace(
            "-",
            "",
        )

        return (
            f"{self._collection_prefix}_"
            f"{normalized_index_id}"
        )