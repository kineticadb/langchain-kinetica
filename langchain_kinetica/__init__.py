"""An integration package connecting Kinetica and LangChain."""

from importlib import metadata

from langchain_kinetica.chat_models import (
    ChatKinetica,
    KineticaSqlOutputParser,
    KineticaSqlResponse,
)
from langchain_kinetica.document_loaders import KineticaLoader
from langchain_kinetica.vectorstores import (
    DistanceStrategy,
    KineticaSettings,
    KineticaVectorstore,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatKinetica",
    "DistanceStrategy",
    "KineticaLoader",
    "KineticaSettings",
    "KineticaSqlOutputParser",
    "KineticaSqlResponse",
    "KineticaVectorstore",
    "__version__",
]
