"""KineticaStore: A ByteStore implementation using Kinetica as the underlying store."""
import logging
from collections.abc import Iterator, Sequence

from gpudb import GPUdb, GPUdbTable
from langchain_core.stores import ByteStore

_LANGCHAIN_DEFAULT_SCHEMA_NAME = "langchain"
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain_store"

LOG = logging.getLogger(__name__)

class KineticaStore(ByteStore):
    """BaseStore implementation using Kinetica as the underlying store.

    Examples:
        Create a KineticaStore instance and perform operations on it:

        .. code-block:: python

            # Instantiate the RedisStore with a Redis connection
            from langchain_community.storage import RedisStore
            from langchain_community.utilities.redis import get_client

            client = get_client('kinetica://localhost:9191')
            kinetica_store = KineticaStore(client=client)

            # Set values for keys
            kinetica_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = kinetica_store.mget(["key1", "key2"])
            # [b"value1", b"value2"]

            # Delete keys
            kinetica_store.mdelete(["key1"])

            # Iterate over keys
            for key in kinetica_store.yield_keys():
                print(key)  # noqa: T201
    """

    def __init__(
        self,
        *,
        kdbc: GPUdb = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        schema_name: str = _LANGCHAIN_DEFAULT_SCHEMA_NAME,
        delete_existing_collection: bool = False,
    ) -> None:
        """Initialize the KineticaStore with a Kinetica connection.

        Must provide either a Kinetica client or a kinetica_url with optional
            client_kwargs.

        Args:
            kdbc: An instance of GPUdb to interact with Kinetica.
            collection_name: The name of the collection to use for storing key-value
                pairs.
            schema_name: The name of the schema to use for storing key-value pairs.
            delete_existing_collection: Whether to delete the existing collection if
                it exists.
        """
        self._kdbc = kdbc
        if self._kdbc is None:
            self._kdbc = GPUdb.get_connection()

        self._collection_name = collection_name
        self._schema_name = schema_name

        self.table_name = self._collection_name
        if self._schema_name is not None and len(self._schema_name) > 0:
            self.table_name = f"{self._schema_name}.{self._collection_name}"

        if delete_existing_collection:
            self.__drop_table_if_exists()


        self._table_schema = [
            ["key", "string"],
            ["value", "bytes"],
        ]

        self._kv_table = self.__create_tables_if_not_exists()


    def __create_tables_if_not_exists(self) -> GPUdbTable:
        """Create the table to store the texts and embeddings."""
        LOG.info("Creating table: %s", self.table_name)
        return GPUdbTable(
            _type=self._table_schema,
            name=self.table_name,
            db=self._kdbc,
            options={"is_replicated": "true"},
        )


    def __drop_table_if_exists(self) -> None:
        """Delete the table."""
        LOG.info("Deleting table: %s", self.table_name)
        self._kdbc.clear_table(
            table_name=f"{self.table_name}",
            options={"no_error_if_not_exists": "true"}
        )


    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys to retrieve values for.

        Returns:
            A list of values corresponding to the given keys, with None for missing
            keys.
        """
        if not keys:
            return []

        # Keys are escaped to prevent SQL injection; Kinetica does not support
        # parameterized queries via execute_sql_and_decode.
        escaped_keys = [k.replace("'", "''") for k in keys]
        keys_str = ", ".join(f"'{k}'" for k in escaped_keys)
        query = (
            f"SELECT key, value" # noqa: S608 - hardcoded-sql-expression
            f" FROM {self.table_name}"
            f" WHERE key IN ({keys_str})"
        )
        resp = self._kdbc.execute_sql_and_decode(query)

        records = resp.get("records", {})
        result_dict = dict(
            zip(records.get("key", []), records.get("value", []), strict=False)
        )
        return [result_dict.get(key) for key in keys]


    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        """Set the given key-value pairs.

        Args:
            key_value_pairs: A sequence of tuples, where each tuple contains a key
                (string) and its corresponding value (bytes).
        """
        # need to cast tuples to lists for GPUdbTable.insert_records
        _key_value_pairs = [list(_tuple) for _tuple in key_value_pairs]

        self._kv_table.insert_records(_key_value_pairs,
                                      options={"update_on_existing_pk": "true"})

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys.

        Args:
            keys: A sequence of keys to delete from the store.
        """
        if not keys:
            return

        # Keys are escaped to prevent SQL injection; Kinetica does not support
        # parameterized queries via execute_sql.
        escaped_keys = [k.replace("'", "''") for k in keys]
        keys_str = ", ".join(f"'{k}'" for k in escaped_keys)
        self._kdbc.execute_sql(
            f"DELETE FROM {self.table_name} WHERE key IN ({keys_str})"  # noqa: S608
        )


    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """Yield keys in the store."""
        # if prefix:
        #     pattern = self._get_prefixed_key(prefix)
        # else:
        #     pattern = self._get_prefixed_key("*")
        # scan_iter = cast(Iterator[bytes], self.client.scan_iter(match=pattern))
        # for key in scan_iter:
        #     decoded_key = key.decode("utf-8")
        #     if self.namespace:
        #         relative_key = decoded_key[len(self.namespace) + 1 :]
        #         yield relative_key
        #     else:
        #         yield decoded_key
        return None
