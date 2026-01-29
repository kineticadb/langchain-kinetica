"""Kinetica Document Loader API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpudb import GPUdb, GPUdbSqlIterator
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

if TYPE_CHECKING:
    from collections.abc import Iterator


class KineticaLoader(BaseLoader):
    """Load from `Kinetica` API.

    Each document represents one row of the result. The `page_content_columns`
    are written into the `page_content` of the document. The `metadata_columns`
    are written into the `metadata` of the document. By default, all columns
    are written into the `page_content` and none into the `metadata`.

    """

    def __init__(
        self,
        query: str,
        kdbc: GPUdb | None = None,
        parameters: dict[str, Any] | None = None,
        page_content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
    ) -> None:
        """Initialize Kinetica document loader.

        Args:
            query: The query to run in Kinetica.
            kdbc (GPUdb, optional): An optional GPUdb connection instance. If not
                provided, the connection will be established using environment
                variables.
            parameters: Optional. Parameters to pass to the query.
            page_content_columns: Optional. Columns written to Document `page_content`.
            metadata_columns: Optional. Columns written to Document `metadata`.
        """
        self.query = query
        self.parameters = parameters
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns if metadata_columns is not None else []

        if kdbc is None:
            kdbc = GPUdb.get_connection()
        self.kdbc = kdbc

    def _execute_query(self) -> list[dict[str, Any]]:
        with GPUdbSqlIterator(self.kdbc, self.query) as records:
            column_names = records.type_map.keys()
            return [dict(zip(column_names, record, strict=False)) for record in records]

    def _get_columns(
        self, query_result: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        page_content_columns = self.page_content_columns
        metadata_columns = self.metadata_columns

        if page_content_columns is None and query_result:
            page_content_columns = list(query_result[0].keys())
        if metadata_columns is None:
            metadata_columns = []
        return page_content_columns or [], metadata_columns

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load data into document objects."""
        query_result = self._execute_query()
        if isinstance(query_result, Exception):
            print(f"An error occurred during the query: {query_result}")  # noqa: T201
            return []
        page_content_columns, metadata_columns = self._get_columns(query_result)
        if "*" in page_content_columns:
            page_content_columns = list(query_result[0].keys())
        for row in query_result:
            page_content = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in page_content_columns
            )
            metadata = {k: v for k, v in row.items() if k in metadata_columns}
            doc = Document(page_content=page_content, metadata=metadata)
            yield doc

    def load(self) -> list[Document]:
        """Load data into document objects."""
        return list(self.lazy_load())
