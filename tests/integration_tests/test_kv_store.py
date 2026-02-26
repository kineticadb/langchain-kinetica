"""Integration tests for KineticaStore."""
import logging

import pytest
from gpudb import GPUdb

from langchain_kinetica.storage import KineticaStore

LOG = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def fx_kdbc() -> GPUdb:
    """Create Kinetica configuration for tests."""
    return GPUdb.get_connection(enable_ssl_cert_verification=True)

def test_kinetica_store(fx_kdbc: GPUdb) -> None:
    """Test KineticaStore functionality."""
    store = KineticaStore(
        kdbc=fx_kdbc,
        schema_name="test",
        collection_name="test_kv_store",
        delete_existing_collection=True)

    # # Test mset
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    store.mset(key_value_pairs)

    # # Test mget
    values = store.mget(["key1", "key2"])
    assert values == [b"value1", b"value2"]

    # Test mdelete
    store.mdelete(["key1"])
    values_after_delete = store.mget(["key1", "key2"])
    assert values_after_delete == [None, b"value2"]
