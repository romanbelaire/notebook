import Quill from "quill";
// svg asset representing a drag handle
import dotsSvg from "../assets/dots-6-vertical.svg?raw";

// Quill's built-in BlockEmbed is the best base class for a card-like block element
// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
const BlockEmbed = Quill.import("blots/block/embed");

type InsightValue = {
  title: string;
  body: string;
};

/**
 * An "insight" is rendered as a coloured call-out card that stores a title
 * and body as dataset attributes so the information is preserved in the Delta.
 *
 * The blot is self-registering; merely importing this file once is enough to
 * make Quill aware of it.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
class InsightBlot extends (BlockEmbed as any) {
  static blotName = "insight";
  static tagName = "div";
  static className = "insight-blot";

  // Quill calls create() whenever a blot is inserted.
  // eslint-disable-next-line @typescript-eslint/explicit-module-boundary-types
  static create(value: InsightValue) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const node: HTMLElement = (BlockEmbed as any).create.call(this);

    // Persist data in attributes so it round-trips through Delta → HTML.
    node.setAttribute("data-title", value.title);
    node.setAttribute("data-body", value.body);

    // Compose inner markup. Keep it simple – devs can style via global CSS.
    node.innerHTML = `
      <div style="display:flex; gap:8px; width:100%;">
        <div style="flex:1; overflow:hidden;">
          <div class="insight-title" style="font-weight:600; margin-bottom:4px;">${
            value.title ?? ""
          }</div>
          <div class="insight-body bg-chat-assistant-bg">${value.body ?? ""}</div>
        </div>
        <div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
          <button class="insight-delete-btn" aria-label="Delete insight" draggable="false" style="background:transparent;border:none;color:#f87171;font-size:14px;font-weight:bold;cursor:pointer;line-height:1;">&times;</button>
          ${dotsSvg.replace(
            "<svg",
            '<svg class="insight-drag-handle" style="cursor:grab;width:16px;height:16px; text-defaultText"'
          )}
        </div>
      </div>
    `;

    node.style.borderLeft = "4px solid rgb(from var(--color-accent-text) r g b / 0.6)";
    node.style.background = "rgb(from var(--color-primary-bg) r g b / 0.6)";
    node.style.color = "rgb(from var(--color-default-text) r g b / 0.6)";
    node.style.padding = "8px";
    node.style.borderRadius = "4px";
    node.style.margin = "8px 0";

    // Make the entire blot draggable so it can serve as a drag source
    node.setAttribute("draggable", "true");

    // Provide drag payload identical to sidebar insights so users can drag
    // cards back out or reorder.
    node.addEventListener("dragstart", (e) => {
      if (!e.dataTransfer) return;
      e.dataTransfer.effectAllowed = "copy";
      e.dataTransfer.setData(
        "text/plain",
        JSON.stringify({ type: "insight", title: value.title, body: value.body })
      );
    });

    // Attach delete handler – remove blot from editor when × clicked
    const delBtn = node.querySelector(".insight-delete-btn") as HTMLButtonElement | null;
    if (delBtn) {
      delBtn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        try {
          // Use Quill.find to locate blot instance and remove it cleanly
          const blot: any = (Quill as any).find(node);
          if (!blot) return;

          // Obtain Quill instance from DOM (root .ql-editor has __quill ref)
          const root = node.closest('.ql-editor') as HTMLElement & { __quill?: any } | null;
          const quill = root?.__quill;

          if (quill && typeof quill.deleteText === 'function') {
            const index = quill.getIndex(blot);
            quill.deleteText(index, 1, 'user');
          } else {
            // Fallback: directly remove blot from Parchment
            blot.remove();
          }
        } catch (err) {
          // Surface error loudly per project guidelines
          // eslint-disable-next-line no-alert
          alert(String(err));
          throw err;
        }
      });
    }

    return node;
  }

  static value(node: HTMLElement): InsightValue {
    return {
      title: node.getAttribute("data-title") ?? "",
      body: node.getAttribute("data-body") ?? "",
    };
  }
}

// Register globally. The second argument "true" allows overwriting in hot-reload.
// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
Quill.register(InsightBlot, true);

export default InsightBlot; 