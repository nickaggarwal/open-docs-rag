import asyncio
import numpy as np
import faiss
import pytest

from app.faiss_vector_store import FAISSVectorStore

class DummyVectorStore(FAISSVectorStore):
    async def generate_embedding(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        idx = sum(ord(c) for c in text) % self.dimension
        vec[idx] = 1.0
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    async def generate_embeddings(self, texts):
        arr = []
        for t in texts:
            v = np.zeros(self.dimension, dtype=np.float32)
            idx = sum(ord(c) for c in t) % self.dimension
            v[idx] = 1.0
            faiss.normalize_L2(v.reshape(1, -1))
            arr.append(v)
        return np.stack(arr).astype(np.float32)

@pytest.mark.asyncio
async def test_cosine_search(tmp_path):
    store = DummyVectorStore(index_path=str(tmp_path / "faiss"))
    await store.add_documents([
        {"text": "foo bar", "metadata": {"url": "doc1"}},
        {"text": "foo baz", "metadata": {"url": "doc2"}},
    ])

    assert isinstance(store.index, faiss.IndexFlatIP)

    results = await store.search("foo bar", k=2)
    assert len(results) == 2
    assert results[0]["metadata"]["url"] == "doc1"
    assert results[0]["score"] >= results[1]["score"]
