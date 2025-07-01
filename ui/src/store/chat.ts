import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  contexts?: string[];
  citations?: Record<string, unknown>[];
  muted?: boolean;
}

interface Conversation {
  id: string;
  title: string; // human readable preview
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
  history: ChatMessage[]; // messages of active conversation OR draft if activeId null
  isSending: boolean;
  // actions
  setSending: (v: boolean) => void;
  addMessage: (msg: ChatMessage) => void;
  switchConversation: (id: string) => void;
  createConversation: () => string;
  finalizeConversation: (title?: string) => string; // convert draft → saved conversation
  deleteConversation: (id: string) => void;
  updateConversationTitle: (id: string, title: string) => void;
  updateMessage: (idx: number, updater: (msg: ChatMessage) => ChatMessage) => void;
  deleteMessage: (idx: number) => void;
}

const genId = () => ((crypto as any).randomUUID ? (crypto as any).randomUUID() : Date.now().toString());

const newConversation = (title: string, messages: ChatMessage[] = []): Conversation => {
  return {
    id: genId(),
    title: title || "New chat",
    messages,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
};

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeId: null,
      history: [],
      isSending: false,
      setSending: (v: boolean) => set({ isSending: v }),
      addMessage: (msg: ChatMessage) => {
        set((state) => {
          if (state.activeId) {
            // Append to existing conversation
            const idx = state.conversations.findIndex((c) => c.id === state.activeId);
            if (idx === -1) return {};
            const conv = { ...state.conversations[idx] };
            conv.messages = [...conv.messages, msg];
            conv.updatedAt = Date.now();
            const updatedConvs = [...state.conversations];
            updatedConvs[idx] = conv;
            return { conversations: updatedConvs, history: conv.messages };
          }
          // Draft stage – keep in history only
          return { history: [...state.history, msg] };
        });
      },
      finalizeConversation: (title?: string) => {
        const { activeId, history, conversations } = get();
        if (activeId) return activeId; // already finalized
        const conv = newConversation(title || "New chat", history);
        set({
          conversations: [conv, ...conversations],
          activeId: conv.id,
          history: conv.messages,
        });
        return conv.id;
      },
      switchConversation: (id: string) => {
        const conv = get().conversations.find((c) => c.id === id);
        if (!conv) return;
        set({ activeId: id, history: conv.messages });
      },
      createConversation: () => {
        const conv = newConversation("New chat");
        set((state) => ({
          conversations: [conv, ...state.conversations],
          activeId: conv.id,
          history: [],
        }));
        return conv.id;
      },
      deleteConversation: (id: string) => {
        set((state) => {
          const conversations = state.conversations.filter((c) => c.id !== id);
          let { activeId, history } = state;
          if (state.activeId === id) {
            activeId = conversations.length ? conversations[0].id : null;
            history = activeId ? conversations.find((c) => c.id === activeId)!.messages : [];
          }
          return { conversations, activeId, history };
        });
      },
      updateConversationTitle: (id: string, title: string) => {
        set((state) => {
          const conversations = state.conversations.map((c) =>
            c.id === id ? { ...c, title } : c
          );
          return { conversations };
        });
      },
      updateMessage: (idx: number, updater: (msg: ChatMessage) => ChatMessage) => {
        set((state) => {
          // Update in history
          const history = [...state.history];
          if (idx < 0 || idx >= history.length) return {};
          history[idx] = updater(history[idx]);

          // Also update in saved conversation if active
          if (state.activeId) {
            const conversations = state.conversations.map((c) => {
              if (c.id !== state.activeId) return c;
              const msgs = [...c.messages];
              if (idx < msgs.length) msgs[idx] = updater(msgs[idx]);
              return { ...c, messages: msgs, updatedAt: Date.now() };
            });
            return { history, conversations };
          }
          return { history };
        });
      },
      deleteMessage: (idx: number) => {
        set((state) => {
          if (idx < 0 || idx >= state.history.length) return {};
          const history = state.history.filter((_, i) => i !== idx);

          let conversations = state.conversations;
          if (state.activeId) {
            conversations = state.conversations.map((c) => {
              if (c.id !== state.activeId) return c;
              const msgs = c.messages.filter((_, i) => i !== idx);
              return { ...c, messages: msgs, updatedAt: Date.now() };
            });
          }
          return { history, conversations };
        });
      },
    }),
    {
      name: "chat-conversations",
      partialize: (state) => ({ conversations: state.conversations, activeId: state.activeId }),
    }
  )
); 