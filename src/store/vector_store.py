import os
from typing import List, Dict, Any, Optional
from chromadb import PersistentClient, Collection
from chromadb.utils import embedding_functions
from src.config import settings


class VectorStore:
    """ChromaDB 向量存储封装"""

    def __init__(self):
        os.makedirs(settings.chromadb_path, exist_ok=True)
        self._client = PersistentClient(path=settings.chromadb_path)
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.siliconflow_api_key,
            api_base=settings.siliconflow_base_url,
            model_name=settings.siliconflow_embedding_model,
        )
        self._collection = self._client.get_or_create_collection(
            name="knowledge_base", embedding_function=ef
        )
        self._doc_names: dict[str, str] = self._rebuild_doc_map()

    def _rebuild_doc_map(self) -> dict[str, str]:
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        name_map = {}
        for meta in metadatas:
            if meta and "source" in meta:
                name = meta["source"]
                name_map[name] = name
        return name_map

    @property
    def doc_names(self) -> list[str]:
        return sorted(self._doc_names.keys())

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        if not ids:
            return 0
        self._collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas
        )
        for meta in metadatas:
            name = meta.get("source", "")
            if name:
                self._doc_names[name] = name
        return len(ids)

    def query(
        self,
        query_text: str,
        top_k: int = None,
        where: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if top_k is None:
            top_k = settings.top_k
        kwargs = {"query_texts": [query_text], "n_results": top_k}
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def delete_by_source(self, source_name: str) -> int:
        """删除指定文档的全部片段"""
        result = self._collection.get(
            where={"source": source_name}, include=["metadatas"]
        )
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        self._doc_names.pop(source_name, None)
        return len(ids_to_delete)

    def doc_count(self) -> int:
        return self._collection.count()
