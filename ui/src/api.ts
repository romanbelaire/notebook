import axios from "axios";

// Base URL from environment variable (defined in Vite), fallback to localhost
const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// Long generations can exceed 60 s; allow 5 min by default, overridable via env.
const REQ_TIMEOUT = Number(import.meta.env.VITE_API_TIMEOUT_MS ?? 300_000);

const api = axios.create({
  baseURL: BASE_URL,
  timeout: REQ_TIMEOUT,
});

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  contexts?: string[];
  citations?: Record<string, unknown>[];
}

export interface ChatResponse {
  answer: string;
  contexts: string[];
  citations: Record<string, unknown>[];
}

export async function postChat(query: string, history: ChatMessage[], modelId?: string) {
  const { data } = await api.post<ChatResponse>("/chat", {
    query,
    history,
    model_id: modelId,
  });
  return data;
}

export async function ingestPdfs(pdfDir: string) {
  const { data } = await api.post<{ task_id: string }>("/ingest", { pdf_dir: pdfDir });
  return data.task_id;
}

export async function getTaskStatus(taskId: string) {
  const { data } = await api.get<{ status: string; result?: unknown; error?: string }>(`/task/${taskId}`);
  return data;
}

export interface Insight {
  id: string;
  title: string;
  text: string;
  contexts: string[];
  created_at: string;
}

export async function listInsights() {
  const { data } = await api.get<Insight[]>("/insight");
  return data;
}

export async function createInsight(text: string, contexts: string[], title?: string) {
  const { data } = await api.post<{ id: string }>("/insight", { text, contexts, title });
  return data.id;
}

export async function deleteInsight(id: string) {
  await api.delete(`/insight/${id}`);
}

export async function updateInsight(id: string, text: string, title?: string) {
  await api.put(`/insight/${id}`, { text, title });
}

export async function searchInsights(query: string, k = 5) {
  const { data } = await api.get<[Insight, number][]>("/insight/search", { params: { query, k } });
  return data;
}

// ---------------------------------------------------------------------------
// Papers & Collections
// ---------------------------------------------------------------------------

export interface Paper {
  id: number;
  filename: string;
  title?: string;
  authors?: string;
  year?: string;
  added_at: string;
}

export interface Collection {
  id: number;
  name: string;
  created_at: string;
  papers?: Paper[];
}

export async function listPapers() {
  const { data } = await api.get<Paper[]>("/papers");
  return data;
}

export async function listCollections() {
  const { data } = await api.get<Collection[]>("/collections");
  return data;
}

export async function createCollection(name: string) {
  const { data } = await api.post<Collection>("/collections", { name });
  return data;
}

export async function addPapersToCollection(collectionId: number, paperIds: number[]) {
  await api.post(`/collections/${collectionId}/add`, { paper_ids: paperIds });
}

export async function removeFromCollection(collectionId: number, paperIds: number[]) {
  await api.post(`/collections/${collectionId}/remove`, { paper_ids: paperIds });
}

export async function renameCollection(collectionId: number, name: string) {
  await api.put(`/collections/${collectionId}`, { name });
}

export async function removeCollection(collectionId: number) {
  await api.delete(`/collections/${collectionId}`);
}

// ---------------------------------------------------------------------------
// Notes
// ---------------------------------------------------------------------------

export async function createNote(contentMd: string, title?: string) {
  const { data } = await api.post<{ paper_id: number }>("/note", { content_md: contentMd, title });
  return data.paper_id;
}

export async function deleteNote(paperId: number) {
  await api.delete(`/note/${paperId}`);
}

// ---------------------------------------------------------------------------
// Context Pool
// ---------------------------------------------------------------------------

export async function setContextPool(collectionId: number | null, modelId?: string) {
  await api.post("/context_pool", { collection_id: collectionId, model_id: modelId });
}

// ---------------------------------------------------------------------------
// Models & Settings
// ---------------------------------------------------------------------------

export async function listModels() {
  const { data } = await api.get<string[]>("/models");
  return data;
}

export async function checkModelAvailable(name: string) {
  const { data } = await api.get<{ available: boolean; gated?: boolean }>("/models/check", { params: { name } });
  return data;
}

export async function saveApiKey(provider: "openai" | "claude" | "hf", key: string) {
  await api.post("/keys", { provider, key });
}

// ---------------------------------------------------------------------------
// System Prompts
// ---------------------------------------------------------------------------

export interface SystemPrompt {
  id: string;
  name: string;
  content: string;
  created_at: string;
}

export interface PromptsResponse {
  prompts: SystemPrompt[];
  selected_id: string;
}

export async function listSystemPrompts() {
  const { data } = await api.get<PromptsResponse>("/prompts");
  return data;
}

export async function createSystemPrompt(name: string, content: string) {
  const { data } = await api.post<SystemPrompt>("/prompts", { name, content });
  return data;
}

export async function updateSystemPrompt(id: string, name: string, content: string) {
  const { data } = await api.put<SystemPrompt>(`/prompts/${id}`, { name, content });
  return data;
}

export async function selectSystemPrompt(id: string) {
  await api.post("/prompts/select", { prompt_id: id });
}

export async function deleteSystemPrompt(id: string) {
  await api.delete(`/prompts/${id}`);
}

// ---------------------------------------------------------------------------
// Debug utilities – dangerous and *only* for local experimentation.
// ---------------------------------------------------------------------------

export async function clearDatabase() {
  await api.post("/debug/clear_db");
} 