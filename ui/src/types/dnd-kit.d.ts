declare module "@dnd-kit/core" {
  import { ReactNode } from "react";
  export interface DndContextProps {
    children: ReactNode;
    sensors?: unknown;
    onDragEnd?: (e: any) => void;
  }
  export const DndContext: React.FC<DndContextProps>;
  export function useDraggable(options: any): any;
  export function useDroppable(options: any): any;
  export function useDndMonitor(callbacks: any): void;
  export function useSensor(sensor: any, config?: any): any;
  export function useSensors(...sensors: any[]): any;
  export const PointerSensor: any;
} 