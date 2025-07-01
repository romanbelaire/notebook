import { useState } from "react";
import { useChatStore } from "../store/chat";
import InsightsPanel from "./InsightsPanel";
import ChatInfoDialog from "./ChatInfoDialog";
import PlusIcon from "../assets/plus.svg?react";
import PencilIcon from "../assets/pencil.svg?react";

export default function Sidebar({ open = true }: { open: boolean }) {
  return (
    <aside
      className={
        `w-full h-full flex flex-col p-4 gap-4 text-defaultText transition-opacity duration-300 ` +
        (open ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none')
      }
    >
      <h2 className="font-semibold text-lg mb-2">Recent Chats</h2>
      <ConversationList />
      <hr className="my-4 border-t border-primaryBg" />
      <h2 className="font-semibold text-lg mb-2">Pinned Insights</h2>
      <InsightsPanel />
    </aside>
  );
}

function ConversationList() {
  const { conversations, activeId, switchConversation, createConversation } = useChatStore((state) => ({
    conversations: state.conversations,
    activeId: state.activeId,
    switchConversation: state.switchConversation,
    createConversation: state.createConversation,
  }));

  const MAX_SHOW = 5;
  const items = [...conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_SHOW);

  const [infoOpenId, setInfoOpenId] = useState<string | null>(null);

  return (
    <div className="space-y-2 text-sm">
      <button
        className="w-full bg-buttonBg opacity-80 hover:opacity-100 text-defaultText rounded px-3 py-1 text-left border border-primaryBg flex items-center gap-2"
        onClick={() => {
          switchConversation(createConversation());
        }}
      >
        <PlusIcon className="w-4 h-4" />
        New Chat
      </button>
      {items.map((c) => (
        <button
          key={c.id}
          className={
            "group w-full text-left px-2 py-1 rounded border border-primaryBg transition-opacity flex items-center text-defaultText " +
            (c.id === activeId
              ? "bg-buttonBg opacity-100"
              : "bg-buttonBg/60 hover:bg-buttonBg opacity-80 hover:opacity-100")
          }
          title={c.title}
          onClick={() => switchConversation(c.id)}
        >
          <span className="flex-1 truncate">{c.title}</span>
          <PencilIcon
            className="w-4 h-4 ml-2 opacity-0 group-hover:opacity-100"
            onClick={(e) => {
              e.stopPropagation();
              setInfoOpenId(c.id);
            }}
          />
        </button>
      ))}

      {infoOpenId && <ChatInfoDialog conversationId={infoOpenId} onClose={() => setInfoOpenId(null)} />}
    </div>
  );
} 