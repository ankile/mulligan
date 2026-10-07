import { useEffect, useRef } from "react";

/** True when the event targets a form control — review shortcuts must not
 *  fire while the operator is typing. */
export function isTypingTarget(event: KeyboardEvent): boolean {
  const target = event.target as HTMLElement | null;
  return Boolean(
    target &&
      (target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable)
  );
}

/**
 * Single window keydown listener whose handler is kept fresh via a ref — the
 * listener is registered exactly once, so handler closures can freely read
 * the latest state without re-subscribing on every render. Shared by the
 * outcome and stage reviews.
 */
export function useWindowKeydown(handler: (event: KeyboardEvent) => void): void {
  const handlerRef = useRef<(event: KeyboardEvent) => void>(() => {});
  useEffect(() => {
    handlerRef.current = handler;
  });
  useEffect(() => {
    const listener = (event: KeyboardEvent) => handlerRef.current(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);
}
