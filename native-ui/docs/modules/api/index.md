# API Module

The `api/` module provides the backend API client.

## Module Structure

```
api/
├── client.rs    # HTTP client
└── models.rs    # API data structures
```

## ApiClient

The `ApiClient` struct provides async HTTP operations:

- Chat requests
- Collections management
- Papers management
- Insights management
- **Shards**: list, create/update, update, delete, semantic search (pinned message bookmarks)
- Ingest operations

See [Client Documentation](client.md).

## Models

API data structures:

- ChatRequest, ChatResponse
- Collection
- ApiPaper
- Insight

See [Models Documentation](models.md).

## Related Documentation

- [API Client](client.md)
- [Models](models.md)

