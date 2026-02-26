"""Integration tests for KineticaStore."""
import logging
import os

import pytest
from gpudb import GPUdb
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from langchain_kinetica.storage import KineticaStore

LOG = logging.getLogger(__name__)

SCHEMA_NAME = "test"

if os.getenv("OPENAI_API_KEY") is not None:
    OPENAI_API_KEY = SecretStr(os.environ["OPENAI_API_KEY"])


@pytest.fixture(scope="module")
def fx_kdbc() -> GPUdb:
    """Create Kinetica configuration for tests."""
    return GPUdb.get_connection(enable_ssl_cert_verification=True)

def test_kinetica_store(fx_kdbc: GPUdb) -> None:
    """Test KineticaStore functionality."""
    store = KineticaStore(
        kdbc=fx_kdbc,
        schema_name=SCHEMA_NAME,
        collection_name="test_kv_store_mget_mset",
        delete_existing_collection=True)

    # # Test mset
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    store.mset(key_value_pairs)

    # # Test mget
    values = store.mget(["key1", "key2", "non_existent_key"])
    assert values == [b"value1", b"value2", None]

    # Test mdelete
    store.mdelete(["key1"])
    values_after_delete = store.mget(["key1", "key2"])
    assert values_after_delete == [None, b"value2"]


def test_empty_mget(fx_kdbc: GPUdb) -> None:
    """Test mget method of KineticaStore with non-existent keys."""
    store = KineticaStore(
        kdbc=fx_kdbc,
        schema_name=SCHEMA_NAME,
        collection_name="test_kv_store_empty_mget",
        delete_existing_collection=True)

    # Test mget with non-existent keys
    values = store.mget(["non_existent_key1", "non_existent_key2"])
    assert values == [None, None]


def test_yield_keys(fx_kdbc: GPUdb) -> None:
    """Test yield_keys method of KineticaStore."""
    store = KineticaStore(
        kdbc=fx_kdbc,
        schema_name="test",
        collection_name="test_kv_store_yeild_keys",
        delete_existing_collection=True)

    # Set up test data
    key_value_pairs = [
        ("key1", b"value1"),
        ("key2", b"value2"),
        ("another_key", b"value3")]
    store.mset(key_value_pairs)

    # Test yield_keys without prefix
    keys = list(store.yield_keys())
    assert set(keys) == {"key1", "key2", "another_key"}

    # Test yield_keys with prefix
    keys_with_prefix = list(store.yield_keys(prefix="key"))
    assert set(keys_with_prefix) == {"key1", "key2"}


@pytest.mark.skipif(
    condition=OPENAI_API_KEY is None,
    reason="OPENAI_API_KEY is None, skipping test"
)
def test_cached_embeddings(fx_kdbc: GPUdb) -> None:
    """Test caching embeddings in KineticaStore."""
    underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    cache_store = KineticaStore(
        kdbc=fx_kdbc,
        schema_name=SCHEMA_NAME,
        collection_name="langchain_embedding_cache",
        delete_existing_collection=True,
    )

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embeddings,
        document_embedding_cache=cache_store,
        namespace=underlying_embeddings.model,
    )

    texts = [
        "Kinetica is a high-performance analytical database.",
        "LangChain makes building LLM applications easy.",
        "Vector similarity search enables semantic retrieval.",
    ]

    # First call — embeddings are computed and cached in Kinetica
    embeddings_1 = cached_embeddings.embed_documents(texts)
    LOG.info("Computed %d embeddings (first call — cache miss).", len(embeddings_1))

    # Second call — embeddings are served from the Kinetica cache
    embeddings_2 = cached_embeddings.embed_documents(texts)
    LOG.info("Retrieved %d embeddings (second call — cache hit).", len(embeddings_2))

    cached_keys = list(cache_store.yield_keys())

    # Inspect cached keys
    LOG.info("Cached embedding keys (first 3):")
    for key in cached_keys[:3]:
        LOG.info("  %s", key)

    assert len(cached_keys) == len(texts)

