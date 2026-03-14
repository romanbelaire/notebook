# API Client

The `api/client.rs` module provides the HTTP client for backend communication.

## ApiClient

```rust
pub struct ApiClient {
    pub client: Client,
    pub base_url: String,
}
```

### Initialization

```rust
let api_client = ApiClient::new(base_url);
```

### Methods

- `post_chat()`: Send chat request
- `list_collections()`: List collections
- `set_context_pool()`: Set context pool
- `list_papers()`: List papers
- `ingest_document()`: Ingest document
- `get_task_status()`: Get task status
- `list_insights()`: List insights
- `create_insight()`: Create insight
- `update_insight()`: Update insight
- **Shards (pinned message bookmarks)**  
  - `list_shards()`: List pinned shards  
  - `create_or_update_shard(id, text, contexts, title?, conversation_id?, parent_id?)`: Pin/update shard  
  - `update_shard(id, text?, contexts?, title?)`: Update shard metadata  
  - `delete_shard(id)`: Unpin shard  
  - `search_shards(query, k)`: Semantic search over pinned shards  
- `get_pdf()`: Get PDF
- `get_note_content()`: Get note content

## Async Operations

API operations are async and return results via channels:

```rust
let (sender, receiver) = mpsc::channel();
tokio::spawn(async move {
    let result = api_client.post_chat(query, history, model_id).await;
    sender.send(result).ok();
});
```

## Related Documentation

- [API Module](index.md)
- [Models](models.md)

