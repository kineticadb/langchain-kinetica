"""Integration tests for Kinetica vector store."""

import os
from typing import override

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr

from langchain_kinetica.vectorstores import (
    DistanceStrategy,
    KineticaSettings,
    KineticaVectorstore,
)

# Fake embedding dimension
DIMENSIONS = 3

# Schema to be created for tests
SCHEMA_NAME = "langchain_test"

# Setting OPENAI_API_KEY will enable optional OpenAI embedding tests
OPENAI_API_KEY = None
if os.getenv("OPENAI_API_KEY") is not None:
    OPENAI_API_KEY = SecretStr(os.environ["OPENAI_API_KEY"])


class FakeEmbeddingsWithAdaDimension(Embeddings):
    """Fake embeddings functionality for testing."""

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return simple embeddings."""
        return [[1.0] * (DIMENSIONS - 1) + [float(i)] for i in range(len(texts))]

    @override
    def embed_query(self, text: str) -> list[float]:
        """Return simple embeddings."""
        return [1.0] * (DIMENSIONS - 1) + [0.0]


@pytest.fixture
def create_config() -> KineticaSettings:
    """Create Kinetica configuration for tests."""
    return KineticaSettings()


def test_kinetica(create_config: KineticaSettings) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"text": text} for text in texts]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        collection_name="test_kinetica",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].page_content == "foo"


def test_kinetica_embeddings(create_config: KineticaSettings) -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings, strict=False))
    docsearch = KineticaVectorstore.from_embeddings(
        config=create_config,
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithAdaDimension(),
        collection_name="test_kinetica_embeddings",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_kinetica_with_metadatas(create_config: KineticaSettings) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        collection_name="test_kinetica_with_metadatas",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_kinetica_with_metadatas_with_scores(create_config: KineticaSettings) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        collection_name="test_kinetica_with_metadatas_with_scores",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_kinetica_with_filter_match(create_config: KineticaSettings) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        collection_name="test_kinetica_with_filter_match",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_score(
        "foo", k=1, emb_filter={"page": "0"}
    )
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_kinetica_with_filter_distant_match(create_config: KineticaSettings) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        collection_name="test_kinetica_with_filter_distant_match",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_score(
        "foo", k=1, emb_filter={"page": "2"}
    )
    assert output == [(Document(page_content="baz", metadata={"page": "2"}), 2.0)]


@pytest.mark.skip(reason="Filter condition has IN clause")
def test_kinetica_with_filter_in_set(create_config: KineticaSettings) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        collection_name="test_kinetica_with_filter_in_set",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_score(
        "foo", k=2, emb_filter={"page": {"IN": ["0", "2"]}}
    )
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 0.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406),
    ]


def test_kinetica_relevance_score(create_config: KineticaSettings) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        collection_name="test_kinetica_relevance_score",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.29289321881345254),
        (Document(page_content="baz", metadata={"page": "2"}), -0.4142135623730949),
    ]


@pytest.mark.skipif(
    OPENAI_API_KEY is None, reason="OPENAI_API_KEY is None, skipping test"
)
def test_kinetica_max_marginal_relevance_search(
    create_config: KineticaSettings,
) -> None:
    """Test end to end construction and search."""
    openai = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    texts = ["foo", "bar", "baz"]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        embedding=openai,
        distance_strategy=DistanceStrategy.COSINE,
        collection_name="test_kinetica_max_marginal_relevance_search",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
    assert output == [Document(page_content="foo")]


def test_kinetica_max_marginal_relevance_search_with_score(
    create_config: KineticaSettings,
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        embedding=FakeEmbeddingsWithAdaDimension(),
        distance_strategy=DistanceStrategy.EUCLIDEAN,
        collection_name="test_kinetica_max_marginal_relevance_search_with_score",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.max_marginal_relevance_search_with_score("foo", k=1, fetch_k=3)
    assert output == [(Document(page_content="foo"), 0.0)]


@pytest.mark.skipif(
    OPENAI_API_KEY is None, reason="OPENAI_API_KEY is None, skipping test"
)
def test_kinetica_with_openai_embeddings(create_config: KineticaSettings) -> None:
    """Test end to end construction and search."""
    openai = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    texts = ["foo", "bar", "baz"]
    metadatas = [{"text": text} for text in texts]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=openai,
        collection_name="kinetica_openai_test",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"text": "foo"})]


def test_kinetica_retriever_search_threshold(create_config: KineticaSettings) -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        distance_strategy=DistanceStrategy.EUCLIDEAN,
        collection_name="test_kinetica_retriever_search_threshold",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.999},
    )
    output = retriever.invoke("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
    ]


def test_kinetica_retriever_search_threshold_custom_normalization_fn(
    create_config: KineticaSettings,
) -> None:
    """Test searching with threshold and custom normalization function."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = KineticaVectorstore.from_texts(
        config=create_config,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        distance_strategy=DistanceStrategy.EUCLIDEAN,
        collection_name="test_kinetica_retriever_search_threshold_custom_normalization_fn",
        schema_name=SCHEMA_NAME,
        pre_delete_collection=False,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    output = retriever.invoke("foo")
    assert output == []
