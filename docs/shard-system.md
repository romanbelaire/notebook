# Shard system

Shards are the core unit of content: a blob of text (e.g. one chat message) with optional metadata, embeddings, and links. **Pinned** shards are persisted on the backend and shown in the sidebar (Insights); they support semantic search via FAISS.

## Backend: ShardStore (`app/shard_store.py`)

`ShardStore` is the persistent store for pinned shards with FAISS for semantic search.

### Behaviour

- **Pin/unpin and list** are synchronous (metadata only; stored in `db/shards.pkl`).
- **Unpin** marks a shard as unpinned; actual removal and index rebuild happen on **cleanup** (app close or explicit `cleanup_unpinned()`).
- **FAISS indexing** runs in a background task after upsert/update so the request returns immediately. Index is written to `db/shards_index.faiss`.
- Embeddings use `all-MiniLM-L6-v2`; up to `MAX_CONTEXTS` (2) context strings are concatenated with text for encoding.

### Class: ShardStore

- `__init__(db_dir="db")`: Create store; loads metadata and optional existing FAISS index.
- `upsert_shard(shard_id, text, contexts, *, title=..., conversation_id=..., parent_id=...)`: Add or update a pinned shard. Saves metadata immediately; triggers async index rebuild. Returns `shard_id`. Raises `ValueError` if text is empty.
- `delete_shard(shard_id)`: Unpin (mark unpinned). Raises `KeyError` if id not found.
- `cleanup_unpinned()`: Remove all unpinned shards and rebuild FAISS index. Call on application close.
- `get_shard(shard_id)`: Return one shard by id, or `None` if not found or unpinned.
- `update_shard(shard_id, *, text=..., contexts=..., title=...)`: Update metadata; re-indexing is async.
- `list_all(pinned_only=True)`: Return list of (pinned) shards.
- `search(query, k=5)`: Return up to `k` pinned shards most relevant to `query` (L2 distance). Returns `List[Tuple[dict, float]]`.

Shard records are dicts with at least: `id`, `text`, `contexts`, `title`, `created_at`, `conversation_id`, `parent_id`, `pinned`.

## HTTP API (rag_server)

Shard CRUD and search are exposed by the FastAPI app:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/shards?pinned=true` | List pinned shards |
| POST | `/shard` | Create or update pinned shard (body: id, text, contexts, title?, …). Returns `{ "id": "…" }`. |
| PUT | `/shard/{shard_id}` | Update shard (body: text?, contexts?, title?) |
| DELETE | `/shard/{shard_id}` | Unpin shard |
| GET | `/shard/search?query=…&k=5` | Semantic search over pinned shards |

On server shutdown, `cleanup_unpinned()` is called so unpinned shards are removed and the index is rebuilt.

## Client (native-ui)

- **Data model**: Shards are defined in `native-ui/src/state/shard.rs` (`Shard`, `ShardMetadata`, `ShardSource`). Conversations store `Vec<Shard>`; see [state/shard](native-ui/docs/modules/state/shard.md).
- **API**: `ApiClient` in `native-ui/src/api/client.rs` provides `list_shards`, `create_or_update_shard`, `update_shard`, `delete_shard`, `search_shards`. See [api/client](native-ui/docs/modules/api/client.md).
