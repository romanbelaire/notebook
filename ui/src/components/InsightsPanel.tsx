import { useEffect, useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listInsights, deleteInsight } from "../api";
import type { Insight } from "../api";
import { useInsightsStore } from "../store/insights";
import PencilIcon from "../assets/pencil.svg?react";
import DotsIcon from "../assets/dots-6-vertical.svg?react";
import TrashIcon from "../assets/trash.svg?react";
import CloseIcon from "../assets/close.svg?react";
import { useDraggable } from "@dnd-kit/core";

type RowProps = {
  ins: Insight;
  deleteConfirmId: string | null;
  setDeleteConfirmId: (id: string | null) => void;
  setModalInsight: (ins: Insight) => void;
  deleteMutation: { mutate: (id: string) => void };
};

function InsightRow({ ins, deleteConfirmId, setDeleteConfirmId, setModalInsight, deleteMutation }: RowProps) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: ins.id,
    data: { type: "insight", title: ins.title ?? "", body: ins.text ?? "" },
  });

  // Capture the element's original viewport position so the fixed overlay
  // starts exactly where the row is, eliminating the 0,0 offset jump.
  const nodeRef = useRef<HTMLDivElement | null>(null);

  const combinedRef = (node: HTMLDivElement | null) => {
    nodeRef.current = node;
    setNodeRef(node);
  };

  const [origin, setOrigin] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (isDragging && nodeRef.current) {
      const rect = nodeRef.current.getBoundingClientRect();
      setOrigin({ x: rect.left, y: rect.top });
    } else if (!isDragging) {
      setOrigin(null);
    }
  }, [isDragging]);

  const style = isDragging && transform && origin
    ? {
        position: 'fixed' as const,
        top: origin.y,
        left: origin.x,
        transform: `translate3d(${transform.x}px, ${transform.y}px, 0)`,
        zIndex: 10000,
        pointerEvents: 'none',
        width: '14rem',
      }
    : transform
    ? { transform: `translate3d(${transform.x}px, ${transform.y}px, 0)` }
    : undefined;

  return (
    <div
      ref={combinedRef}
      style={style}
      {...listeners}
      {...attributes}
      className={
        "group w-full px-2 py-1 rounded border border-primaryBg transition-opacity flex items-center bg-buttonBg/60 hover:bg-buttonBg opacity-80 hover:opacity-100 " +
        (isDragging ? "opacity-40" : "cursor-grab")
      }
      title={ins.title || ins.text.slice(0, 60)}
      onClick={(e) => {
        if (deleteConfirmId === null) {
          setModalInsight(ins);
        }
      }}
    >
      {deleteConfirmId === ins.id ? (
        <div className="flex items-center gap-1 w-full justify-end">
          <button
            className="flex px-2 py-1 h-5 items-center justify-center text-xs rounded bg-[#db363c]/50 hover:bg-[#db363c] focus:outline-none"
            onClick={(e) => {
              e.stopPropagation();
              deleteMutation.mutate(ins.id);
              setDeleteConfirmId(null);
            }}
          >
            Delete
          </button>
          <button
            className="w-5 h-5 p-0 flex items-center justify-center bg-transparent border-0 hover:bg-white/10 focus:outline-none"
            onClick={(e) => {
              e.stopPropagation();
              setDeleteConfirmId(null);
            }}
          >
            <CloseIcon className="w-4 h-4 pointer-events-none" />
          </button>
        </div>
      ) : (
        <>
          <span className="flex-1 truncate">
            {(ins.title || ins.text).slice(0, 40)}
            {(ins.title || ins.text).length > 40 ? "…" : ""}
          </span>
          {/* edit icon */}
          <PencilIcon
            className="w-4 h-4 ml-2 opacity-0 group-hover:opacity-100 flex-none"
            onClick={(e) => {
              e.stopPropagation();
              setModalInsight(ins);
            }}
          />
          {/* trash icon */}
          <button
            className="w-4 h-4 ml-2 p-0 text-defaultText flex items-center justify-center bg-transparent border-0 opacity-0 group-hover:opacity-100 hover:bg-white/10 focus:outline-none"
            onClick={(e) => {
              e.stopPropagation();
              setDeleteConfirmId(ins.id);
            }}
            title="Delete insight"
          >
            <TrashIcon className="w-4 h-4 pointer-events-none" />
          </button>
          {/* drag handle icon */}
          <DotsIcon
            className="w-4 h-4 ml-2 opacity-60 group-hover:opacity-100 flex-none pointer-events-none"
          />
        </>
      )}
    </div>
  );
}

export default function InsightsPanel() {
  const { insights, setInsights, removeInsight, setModalInsight } = useInsightsStore();
  const queryClient = useQueryClient();
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const { data: fetched } = useQuery<Insight[], Error>({
    queryKey: ["insights"],
    queryFn: listInsights,
  });

  useEffect(() => {
    if (fetched) setInsights(fetched);
  }, [fetched, setInsights]);

  const deleteMutation = useMutation<void, Error, string>({
    mutationFn: deleteInsight,
    onSuccess: (_, id) => {
      removeInsight(id);
      queryClient.invalidateQueries({ queryKey: ["insights"] });
    },
  });

  useEffect(() => {
    // No-op: just to mark mount to TS
  }, []);

  if (insights.length === 0) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400">No insights pinned yet.</div>
    );
  }

  return (
    <>
      <div className="space-y-2 overflow-y-auto max-h-60 text-sm">
        {insights.map((ins) => (
          <InsightRow
            key={ins.id}
            ins={ins}
            deleteConfirmId={deleteConfirmId}
            setDeleteConfirmId={setDeleteConfirmId}
            setModalInsight={setModalInsight}
            deleteMutation={deleteMutation}
          />
        ))}
      </div>

      {/* Modal handled by InsightModal component globally */}
    </>
  );
} 