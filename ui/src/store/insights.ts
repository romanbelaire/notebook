import { create } from "zustand";
import type { SetState } from "zustand";
import type { Insight } from "../api";

interface InsightsState {
  insights: Insight[];
  modalInsight: Insight | null;
  setInsights: (items: Insight[]) => void;
  addInsight: (item: Insight) => void;
  removeInsight: (id: string) => void;
  updateInsightTitle: (id: string, title: string) => void;
  updateInsightText: (id: string, text: string) => void;
  setModalInsight: (ins: Insight | null) => void;
}

export const useInsightsStore = create<InsightsState>((set: SetState<InsightsState>) => ({
  insights: [],
  modalInsight: null,
  setInsights: (items: Insight[]) => set({ insights: items }),
  addInsight: (item: Insight) => set((state) => ({ insights: [item, ...state.insights] })),
  removeInsight: (id: string) =>
    set((state) => ({ insights: state.insights.filter((ins) => ins.id !== id) })),
  updateInsightTitle: (id: string, title: string) =>
    set((state) => {
      const updatedInsights = state.insights.map((ins) => (ins.id === id ? { ...ins, title } : ins));
      const updatedModal = state.modalInsight && state.modalInsight.id === id ? { ...state.modalInsight, title } : state.modalInsight;
      return { insights: updatedInsights, modalInsight: updatedModal } as any;
    }),
  updateInsightText: (id: string, text: string) =>
    set((state) => {
      const updatedInsights = state.insights.map((ins) => (ins.id === id ? { ...ins, text } : ins));
      const updatedModal = state.modalInsight && state.modalInsight.id === id ? { ...state.modalInsight, text } : state.modalInsight;
      return { insights: updatedInsights, modalInsight: updatedModal } as any;
    }),
  setModalInsight: (ins: Insight | null) => set({ modalInsight: ins }),
}) as InsightsState as any); 