Yes. This is exactly the right moment to formalize it.

I’ll write this as a **clean engineering spec** you can hand to Cursor.
Concise, structured, implementation-oriented. No philosophy.

---

# Constellar — Project Specification (v1)

## 1. Vision

Constellar is a graph-based interface for managing LLM context explicitly.

Instead of linear chat transcripts, conversations form a **branching DAG of shards** (context units).
Users manually activate, hide, compress, and attach shards before sending a message.

Core principle:

> What is activated is exactly what the model sees.
> No silent summarization. No hidden heuristics.

---

## 2. Core Design Principles

1. **Explicit context control** — No automatic pruning or summarization.
2. **Deterministic compilation** — Same activated graph → same prompt.
3. **Reproducibility** — Every API call is snapshot-able.
4. **Non-destructive compression** — Raw shards are never deleted.
5. **Graph-native UI, linear-native compiler** — Internal DAG, serialized output.

---

## 3. Core Data Model

### 3.1 Shard

A shard is the atomic unit of context.

```ts
type Shard = {
  id: string
  content: string
  role: "system" | "user" | "assistant" | "external" | "summary"
  parentIds: string[]      // supports branching DAG
  createdAt: timestamp
  visible: boolean         // user-controlled inclusion
  metadata?: {
    title?: string
    tags?: string[]
    tokenCount?: number
    compressedFrom?: string[] // if summary shard
  }
}
```

Notes:

* Multiple parents allow merges.
* `visible` controls compilation inclusion.
* No shard is ever deleted — only hidden or archived.

---

### 3.2 Conversation Graph

```ts
type ConversationGraph = {
  shards: Map<string, Shard>
  rootId: string
}
```

Graph is a DAG.
Branches occur when multiple children reference same parent.

---

### 3.3 Active Leaf

The user is always positioned at a specific leaf shard.

```ts
type ActiveState = {
  currentLeafId: string
}
```

---

## 4. Compiler

The compiler converts activated graph state into a linear prompt.

### 4.1 Compilation Rule (v1 — Simple & Deterministic)

Given `currentLeafId`:

1. Traverse backward to root following parent pointers.
2. Collect all shards where `visible == true`.
3. Preserve chronological order from root → leaf.
4. Serialize according to role sections.

No automatic compression.
No heuristics.
No token-based pruning.

---

### 4.2 Serialization Template (v1)

Output format:

```
=== SYSTEM ===
(system shards in order)

=== CONTEXT ===
(all visible non-system shards except current user draft)

=== USER MESSAGE ===
(current draft content)
```

Alternative (if role-based chat API used):

Return structured message array:

```ts
[
  { role: "system", content: "..." },
  { role: "user", content: "..." },
  { role: "assistant", content: "..." },
  ...
  { role: "user", content: "NEW MESSAGE" }
]
```

---

## 5. Branching Behavior

When a user replies to any previous shard:

* Create new shard with `parentIds = [selectedShardId]`
* This creates a new branch
* No mutation of existing branch

---

## 6. Hiding Shards

If a shard is hidden:

* `visible = false`
* It remains in graph
* Compiler excludes it from traversal output

Important: Hidden shards do not affect children automatically.
Visibility is per shard, not inherited.

---

## 7. External Documents

External documents are imported as shards:

```ts
role: "external"
```

They are treated identically in compilation if visible.

Optional future:

* Chunk large docs into multiple shards.

---

## 8. Summary Shards (Manual Only — v1.1)

User may select multiple shards and click:

> "Summarize into new shard"

System:

* Sends selected shards to LLM
* Creates new shard:

  * role: "summary"
  * compressedFrom: [ids]
* Optionally auto-hide original shards

Crucial:
Original shards remain recoverable.

---

## 9. Token Accounting

Before sending:

1. Compile prompt.
2. Tokenize.
3. Display token count.
4. If exceeds model limit:

   * Refuse send.
   * Prompt user to hide/compress shards.

No silent truncation.

---

## 10. Snapshot Logging (Reproducibility)

Each send generates:

```ts
type Snapshot = {
  id: string
  timestamp: number
  compiledShardIds: string[]
  serializedPrompt: string
  model: string
  temperature: number
  response: string
}
```

This allows:

* Re-running identical calls
* Auditing context
* Research reproducibility

---

## 11. UI Requirements

### Must Have

* Graph visualization (nodes + branches)
* Current active path highlight
* Toggle visibility per shard
* Token counter
* Preview compiled prompt before send

### Nice to Have

* Collapsible branches
* Diff view between branches
* Summary shard creation button
* “Activate only this path” shortcut



## 13. Future Extensions (Research-Oriented)

1. **Activation Policies**

   * Embedding-based shard suggestion
   * Relevance scoring
2. **Iterative Expansion Mode**

   * Model requests additional shards
4. **Branch Comparison**

   * Evaluate responses across branches


