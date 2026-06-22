"""Package exceptions."""


class GraphRAGError(Exception):
    """Base package exception."""


class ModelBackendError(GraphRAGError):
    """Raised when a configured model backend cannot be used."""


class StorageError(GraphRAGError):
    """Raised when a storage backend operation fails."""


class IngestionError(GraphRAGError):
    """Raised for ingestion failures."""


class RetrievalError(GraphRAGError):
    """Raised for retrieval failures."""

