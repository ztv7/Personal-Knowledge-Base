import numpy as np
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings


class HybridSplitter:
    """递归切分 + Embedding 语义边界检测混合切分器"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        similarity_threshold: float = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.similarity_threshold = (
            similarity_threshold or settings.similarity_threshold
        )
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", ". ", " ", ""],
        )

    def split(self, text: str) -> List[str]:
        """先递归切分，再通过语义相似度合并过于相似的相邻片段"""
        raw_chunks = self._recursive_splitter.split_text(text)
        if len(raw_chunks) <= 1:
            return [t for t in raw_chunks if t.strip()]

        # 合并过短的尾部片段到前一个 chunk
        merged = self._merge_short_chunks(raw_chunks)
        return [m for m in merged if m.strip()]

    def split_with_semantic(
        self, text: str, embedding_fn
    ) -> List[str]:
        """
        完整混合切分:
        1. RecursiveCharacterTextSplitter 粗切
        2. 计算相邻 chunk embedding 余弦相似度
        3. 相似度高于阈值的相邻 chunk 合并 (说明语义连续)
        4. 相似度低于阈值的保留边界 (说明语义切换)
        """
        raw_chunks = self._recursive_splitter.split_text(text)
        if len(raw_chunks) <= 1:
            return [t for t in raw_chunks if t.strip()]

        # 获取所有 chunk 的 embedding
        embeddings = embedding_fn(raw_chunks)
        emb_matrix = np.array(embeddings)

        # 计算相邻相似度，决定合并/保留
        merged_chunks = []
        current = raw_chunks[0]
        current_emb = emb_matrix[0]

        for i in range(1, len(raw_chunks)):
            sim = self._cosine_similarity(current_emb, emb_matrix[i])
            if sim >= self.similarity_threshold:
                # 语义连续，合并
                current += "\n" + raw_chunks[i]
                # 更新 current_emb 为合并后两段的均值
                current_emb = (current_emb + emb_matrix[i]) / 2.0
            else:
                # 语义断点，保留当前 chunk
                merged_chunks.append(current)
                current = raw_chunks[i]
                current_emb = emb_matrix[i]

        merged_chunks.append(current)
        # 合并过短片段
        result = self._merge_short_chunks(merged_chunks)
        return [r for r in result if r.strip()]

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """将长度不足 chunk_size/4 的片段合并到前一个"""
        min_len = max(self.chunk_size // 4, 50)
        if len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            if len(chunks[i]) < min_len and result:
                result[-1] += "\n" + chunks[i]
            else:
                result.append(chunks[i])
        return result

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)
