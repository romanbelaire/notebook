use crate::api::models::{
    ChatRequest, ChatResponse, ApiChatMessage, Insight, CreateShardRequest, UpdateShardRequest,
    CreateGraphResponse, GetGraphResponse, GraphSendRequest, GraphSendResponse, GraphListResponse,
    GraphCompileRequest, GraphCompileResponse, GraphShardPatchRequest, ArxivImportResponse,
};
use crate::ui::chat_window::ChatMessage;
use anyhow::Result;
use reqwest::Client;
use reqwest::multipart;
use std::path::Path;
use std::time::Duration;
use futures::StreamExt;

pub struct ApiClient {
    pub client: Client,
    pub base_url: String,
}

impl ApiClient {
    pub fn new(base_url: Option<String>) -> Self {
        let base_url = base_url
            .or_else(|| std::env::var("API_URL").ok())
            .unwrap_or_else(|| "http://localhost:8000".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minutes timeout
            .build()
            .expect("Failed to create HTTP client");

        Self { client, base_url }
    }

    /// Old /chat endpoint. Prefer graph send (`send_graph`) for conversation flow.
    #[deprecated(since = "0.1.0", note = "Use send_graph for conversation; /chat endpoints are legacy")]
    pub async fn post_chat(
        &self,
        query: &str,
        history: &[ChatMessage],
        provider: &str,
        model_id: Option<&str>,
        openai_model: Option<&str>,
    ) -> Result<ChatResponse> {
        let api_history: Vec<ApiChatMessage> = history.iter().map(|m| m.into()).collect();

        let request = ChatRequest {
            query: query.to_string(),
            history: api_history,
            provider: provider.to_string(),
            model_id: model_id.map(|s| s.to_string()),
            openai_model: openai_model.map(|s| s.to_string()),
        };

        let url = format!("{}/chat", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        let chat_response: ChatResponse = response.json().await?;
        Ok(chat_response)
    }

    /// Old /chat/stream endpoint. Prefer graph send for conversation flow.
    #[deprecated(since = "0.1.0", note = "Use send_graph for conversation; /chat endpoints are legacy")]
    pub async fn post_chat_stream(
        &self,
        query: &str,
        history: &[ChatMessage],
        provider: &str,
        model_id: Option<&str>,
        openai_model: Option<&str>,
        on_chunk: impl Fn(String) -> Result<()>,
    ) -> Result<ChatResponse> {
        let api_history: Vec<ApiChatMessage> = history.iter().map(|m| m.into()).collect();

        let request = ChatRequest {
            query: query.to_string(),
            history: api_history,
            provider: provider.to_string(),
            model_id: model_id.map(|s| s.to_string()),
            openai_model: openai_model.map(|s| s.to_string()),
        };

        // Try streaming endpoint first, fallback to regular endpoint
        let stream_url = format!("{}/chat/stream", self.base_url);
        let response = self
            .client
            .post(&stream_url)
            .header("Accept", "text/event-stream")
            .json(&request)
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() && resp.headers().get("content-type")
                .and_then(|h| h.to_str().ok())
                .map(|ct| ct.contains("text/event-stream"))
                .unwrap_or(false) => {
                // Streaming response
                let mut stream = resp.bytes_stream();
                let mut buffer = String::new();
                let mut accumulated_answer = String::new();
                
                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            buffer.push_str(&text);
                            
                            // Process complete SSE events
                            while let Some(newline_pos) = buffer.find('\n') {
                                let line = buffer[..newline_pos].trim().to_string();
                                buffer = buffer[newline_pos + 1..].to_string();
                                
                                if line.starts_with("data: ") {
                                    let data = &line[6..];
                                    if data == "[DONE]" {
                                        break;
                                    }
                                    
                                    // Try to parse as JSON chunk
                                    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(data) {
                                        if let Some(text) = json_value.get("text").and_then(|v| v.as_str()) {
                                            accumulated_answer.push_str(text);
                                            on_chunk(text.to_string())?;
                                        } else if let Some(answer) = json_value.get("answer").and_then(|v| v.as_str()) {
                                            // Full answer in one chunk
                                            accumulated_answer = answer.to_string();
                                            on_chunk(answer.to_string())?;
                                        }
                                    } else {
                                        // Plain text chunk
                                        accumulated_answer.push_str(data);
                                        on_chunk(data.to_string())?;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            anyhow::bail!("Stream error: {}", e);
                        }
                    }
                }
                
                // Return final response (we'd need to accumulate contexts/citations too)
                // For now, return with accumulated answer
                Ok(ChatResponse {
                    answer: accumulated_answer,
                    contexts: Vec::new(),  // Would need to be accumulated from stream
                    citations: Vec::new(),  // Would need to be accumulated from stream
                })
            }
            _ => {
                // Fallback to regular non-streaming endpoint
                #[allow(deprecated)]
                self.post_chat(query, history, provider, model_id, openai_model).await
            }
        }
    }

    pub async fn list_collections(&self) -> Result<Vec<crate::api::models::Collection>> {
        let url = format!("{}/collections", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        let collections: Vec<crate::api::models::Collection> = response.json().await?;
        Ok(collections)
    }

    pub async fn set_context_pool(
        &self,
        collection_id: Option<i32>,
        model_id: Option<&str>,
    ) -> Result<()> {
        let request = crate::api::models::ContextPoolRequest {
            collection_id,
            model_id: model_id.map(|s| s.to_string()),
        };

        let url = format!("{}/context_pool", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        Ok(())
    }

    pub async fn list_papers(&self) -> Result<Vec<crate::api::models::ApiPaper>> {
        let url = format!("{}/papers", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        let papers: Vec<crate::api::models::ApiPaper> = response.json().await?;
        Ok(papers)
    }

    pub async fn create_collection(&self, name: &str) -> Result<crate::api::models::Collection> {
        let request = crate::api::models::CreateCollectionRequest {
            name: name.to_string(),
        };

        let url = format!("{}/collections", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        let collection: crate::api::models::Collection = response.json().await?;
        Ok(collection)
    }

    pub async fn update_collection(&self, id: i32, name: &str) -> Result<()> {
        let request = crate::api::models::CreateCollectionRequest {
            name: name.to_string(),
        };

        let url = format!("{}/collections/{}", self.base_url, id);
        let response = self
            .client
            .put(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        Ok(())
    }

    pub async fn delete_collection(&self, id: i32) -> Result<()> {
        let url = format!("{}/collections/{}", self.base_url, id);
        let response = self
            .client
            .delete(&url)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        Ok(())
    }

    pub async fn add_papers_to_collection(&self, collection_id: i32, paper_ids: &[i32]) -> Result<()> {
        let request = crate::api::models::CollectionPaperIdsRequest {
            paper_ids: paper_ids.to_vec(),
        };
        let url = format!("{}/collections/{}/add", self.base_url, collection_id);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        Ok(())
    }

    pub async fn remove_papers_from_collection(&self, collection_id: i32, paper_ids: &[i32]) -> Result<()> {
        let request = crate::api::models::CollectionPaperIdsRequest {
            paper_ids: paper_ids.to_vec(),
        };
        let url = format!("{}/collections/{}/remove", self.base_url, collection_id);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        Ok(())
    }

    pub async fn ingest_pdfs(&self, pdf_dir: &str) -> Result<String> {
        let url = format!("{}/ingest", self.base_url);
        let request = serde_json::json!({
            "pdf_dir": pdf_dir
        });

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        let result: serde_json::Value = response.json().await?;
        let task_id = result.get("task_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No task_id in response"))?;
        Ok(task_id.to_string())
    }

    pub async fn import_arxiv_text(&self, input_text: &str) -> Result<ArxivImportResponse> {
        let url = format!("{}/import/arxiv", self.base_url);
        let request = serde_json::json!({
            "input_text": input_text
        });
        let response = self.client.post(&url).json(&request).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let result: ArxivImportResponse = response.json().await?;
        Ok(result)
    }

    pub async fn import_arxiv_bibtex(&self, path: &Path) -> Result<ArxivImportResponse> {
        let url = format!("{}/import/arxiv/bibtex", self.base_url);
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid bibtex filename"))?
            .to_string();
        let file_bytes = std::fs::read(path)?;
        let part = multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("application/x-bibtex")?;
        let form = multipart::Form::new().part("file", part);
        let response = self.client.post(&url).multipart(form).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let result: ArxivImportResponse = response.json().await?;
        Ok(result)
    }

    pub async fn get_task_status(&self, task_id: &str) -> Result<serde_json::Value> {
        let url = format!("{}/task/{}", self.base_url, task_id);
        let response = self
            .client
            .get(&url)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        let result: serde_json::Value = response.json().await?;
        Ok(result)
    }

    pub async fn list_shards(&self) -> Result<Vec<Insight>> {
        let url = format!("{}/shards", self.base_url);
        let response = self.client.get(&url).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let shards: Vec<Insight> = response.json().await?;
        Ok(shards)
    }

    pub async fn create_or_update_shard(
        &self,
        shard_id: &str,
        text: &str,
        contexts: Vec<String>,
        title: Option<String>,
        conversation_id: Option<String>,
        parent_id: Option<String>,
        notes: Option<Vec<String>>,
    ) -> Result<String> {
        let request = CreateShardRequest {
            id: shard_id.to_string(),
            text: text.to_string(),
            contexts,
            title,
            conversation_id,
            parent_id,
            notes,
        };
        let url = format!("{}/shard", self.base_url);
        let response = self.client.post(&url).json(&request).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let result: serde_json::Value = response.json().await?;
        let id = result
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No id in response"))?;
        Ok(id.to_string())
    }

    pub async fn update_shard(
        &self,
        id: &str,
        text: Option<String>,
        contexts: Option<Vec<String>>,
        title: Option<String>,
        notes: Option<Vec<String>>,
    ) -> Result<()> {
        let request = UpdateShardRequest { text, contexts, title, notes };
        let url = format!("{}/shard/{}", self.base_url, id);
        let response = self.client.put(&url).json(&request).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        Ok(())
    }

    pub async fn delete_shard(&self, id: &str) -> Result<()> {
        let url = format!("{}/shard/{}", self.base_url, id);
        let response = self.client.delete(&url).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        Ok(())
    }

    pub async fn search_shards(&self, query: &str, k: usize) -> Result<Vec<(Insight, f64)>> {
        let url = format!("{}/shard/search?query={}&k={}", self.base_url, query, k);
        let response = self.client.get(&url).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let results: Vec<(Insight, f64)> = response.json().await?;
        Ok(results)
    }

    pub async fn create_note(&self, content_md: &str, title: Option<&str>) -> Result<i32> {
        #[derive(serde::Serialize)]
        struct NoteRequest {
            content_md: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            title: Option<String>,
        }

        let request = NoteRequest {
            content_md: content_md.to_string(),
            title: title.map(|s| s.to_string()),
        };

        let url = format!("{}/note", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        #[derive(serde::Deserialize)]
        struct NoteResponse {
            paper_id: i32,
        }

        let note_response: NoteResponse = response.json().await?;
        Ok(note_response.paper_id)
    }

    pub async fn delete_note(&self, paper_id: i32) -> Result<()> {
        let url = format!("{}/note/{}", self.base_url, paper_id);
        let response = self
            .client
            .delete(&url)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        Ok(())
    }

    pub async fn get_note_content(&self, paper_id: i32) -> Result<String> {
        // Notes are stored as papers, so we fetch the paper content
        let url = format!("{}/papers/{}", self.base_url, paper_id);
        let response = self
            .client
            .get(&url)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        // The response is the PDF file or note content as bytes
        // For notes, we expect markdown/text content
        let bytes = response.bytes().await?;
        let content = String::from_utf8_lossy(&bytes).to_string();
        Ok(content)
    }

    /// Compute vector embedding for the given text
    /// Returns a 384-dimensional vector (all-MiniLM-L6-v2)
    pub async fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(serde::Serialize)]
        struct EmbedRequest {
            text: String,
        }

        #[derive(serde::Deserialize)]
        struct EmbedResponse {
            embedding: Vec<f32>,
        }

        let request = EmbedRequest {
            text: text.to_string(),
        };

        let url = format!("{}/embed", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }

        let embed_response: EmbedResponse = response.json().await?;
        Ok(embed_response.embedding)
    }

    // ---------------------------------------------------------------------------
    // Constellar graph API
    // ---------------------------------------------------------------------------

    /// Create a new conversation graph. Returns graph_id and root_id.
    pub async fn create_graph(&self) -> Result<CreateGraphResponse> {
        let url = format!("{}/graph", self.base_url);
        let response = self.client.post(&url).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let body: CreateGraphResponse = response.json().await?;
        Ok(body)
    }

    /// Load full graph and active state.
    pub async fn get_graph(&self, graph_id: &str) -> Result<GetGraphResponse> {
        let url = format!("{}/graph/{}", self.base_url, graph_id);
        let response = self.client.get(&url).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let body: GetGraphResponse = response.json().await?;
        Ok(body)
    }

    /// Send user message from current leaf; returns response, new_leaf_id, token_count.
    pub async fn send_graph(&self, graph_id: &str, request: &GraphSendRequest) -> Result<GraphSendResponse> {
        let url = format!("{}/graph/{}/send", self.base_url, graph_id);
        let response = self.client.post(&url).json(request).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let body: GraphSendResponse = response.json().await?;
        Ok(body)
    }

    /// List graph IDs (newest first).
    pub async fn list_graphs(&self) -> Result<GraphListResponse> {
        let url = format!("{}/graphs", self.base_url);
        let response = self.client.get(&url).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let body: GraphListResponse = response.json().await?;
        Ok(body)
    }

    /// Compile graph for given leaf and draft; returns messages, compiled_shard_ids, token_count.
    pub async fn compile_graph(
        &self,
        graph_id: &str,
        current_leaf_id: &str,
        user_draft: &str,
    ) -> Result<GraphCompileResponse> {
        let url = format!("{}/graph/{}/compile", self.base_url, graph_id);
        let payload = GraphCompileRequest {
            current_leaf_id: current_leaf_id.to_string(),
            user_draft: user_draft.to_string(),
        };
        let response = self.client.post(&url).json(&payload).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        let body: GraphCompileResponse = response.json().await?;
        Ok(body)
    }

    /// Patch shard fields (visibility, notes).
    pub async fn patch_shard(
        &self,
        graph_id: &str,
        shard_id: &str,
        visible: Option<bool>,
        notes: Option<Vec<String>>,
    ) -> Result<()> {
        let url = format!("{}/graph/{}/shard/{}", self.base_url, graph_id, shard_id);
        let payload = GraphShardPatchRequest { visible, notes };
        let response = self.client.patch(&url).json(&payload).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        Ok(())
    }

    /// Patch shard visibility.
    pub async fn patch_shard_visibility(
        &self,
        graph_id: &str,
        shard_id: &str,
        visible: bool,
    ) -> Result<()> {
        self.patch_shard(graph_id, shard_id, Some(visible), None).await
    }

    /// Remove shard from graph (e.g. DELETE /graph/{graph_id}/shard/{shard_id}).
    pub async fn remove_shard_from_graph(&self, graph_id: &str, shard_id: &str) -> Result<()> {
        let url = format!("{}/graph/{}/shard/{}", self.base_url, graph_id, shard_id);
        let response = self.client.delete(&url).send().await?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API error {}: {}", status, error_text);
        }
        Ok(())
    }
}

