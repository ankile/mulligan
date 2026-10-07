import { useState, useCallback, useEffect } from "react";

function getParam(key: string): string | null {
  return new URLSearchParams(window.location.search).get(key);
}

// Params that are working STATE, not addressable pages: writing them with
// pushState makes every episode selection/filter change a history entry, so
// one Back press mid-review discards unsaved marks and lands on a stale
// episode. These always replace instead.
const REPLACE_KEYS = new Set([
  "episode",
  "queue",
  "status",
  "arm",
  // Stage-review working state (prefixed so a URL crossing the outcome/stage
  // boundary never mis-filters; Back must not discard in-progress labeling).
  "sstatus",
  "sconf",
  "sflag",
  "sarm",
  "schema",
  "prediction",
  "blind",
  // Joined-view policy visibility filter.
  "hide",
]);

// pushState/replaceState fire no event, so hook instances holding the same
// key in OTHER components would keep their stale mount-time value (e.g. the
// App-level mainline/all toggle updating the URL while EvalSessions kept
// filtering by the old lens). Every accepted write broadcasts this event so all
// instances re-read.
const PARAMS_EVENT = "searchparamschange";

type NavigationGuard = (current: URLSearchParams, next: URLSearchParams) => boolean;
const navigationGuards = new Set<NavigationGuard>();

/** Atomic navigation: guards run before URL state or any subscriber changes. */
export function setSearchParams(updates: Record<string, string | null>): boolean {
  const current = new URLSearchParams(window.location.search);
  const params = new URLSearchParams(current);
  for (const [key, value] of Object.entries(updates)) {
    if (value === null) params.delete(key);
    else params.set(key, value);
  }
  for (const guard of navigationGuards) {
    if (!guard(current, params)) return false;
  }
  const qs = params.toString();
  const url = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
  if (Object.keys(updates).every((key) => REPLACE_KEYS.has(key))) {
    window.history.replaceState(null, "", url);
  } else {
    window.history.pushState(null, "", url);
  }
  window.dispatchEvent(new Event(PARAMS_EVENT));
  return true;
}

/**
 * Protect a mounted editor from parent-route unmounts. Back/Forward fires
 * after the browser changes the URL, so restore the editor URL and stop the
 * event before any URL-hook subscriber can render the destination.
 */
export function useSearchParamNavigationGuard(guard: NavigationGuard) {
  useEffect(() => {
    navigationGuards.add(guard);
    let currentUrl = window.location.href;
    const remember = () => { currentUrl = window.location.href; };
    const onPopState = (event: PopStateEvent) => {
      const current = new URL(currentUrl);
      const next = new URL(window.location.href);
      if (!guard(current.searchParams, next.searchParams)) {
        event.stopImmediatePropagation();
        window.history.pushState(null, "", currentUrl);
        return;
      }
      remember();
    };
    window.addEventListener(PARAMS_EVENT, remember);
    window.addEventListener("popstate", onPopState, true);
    return () => {
      navigationGuards.delete(guard);
      window.removeEventListener(PARAMS_EVENT, remember);
      window.removeEventListener("popstate", onPopState, true);
    };
  }, [guard]);
}

/** Re-read a param from the URL on popstate (back/forward) or any setSearchParams write. */
function useSyncOnPopState(key: string, setValue: (v: string | null) => void) {
  useEffect(() => {
    const handler = () => setValue(getParam(key));
    window.addEventListener("popstate", handler);
    window.addEventListener(PARAMS_EVENT, handler);
    return () => {
      window.removeEventListener("popstate", handler);
      window.removeEventListener(PARAMS_EVENT, handler);
    };
  }, [key, setValue]);
}

/** Sync a required string param with the URL. Falls back to `defaultValue` when absent. */
export function useSearchParam(key: string, defaultValue: string): [string, (v: string) => void] {
  const [value, setValue] = useState(() => getParam(key) ?? defaultValue);

  const setRaw = useCallback(
    (v: string | null) => setValue(v ?? defaultValue),
    [defaultValue],
  );
  useSyncOnPopState(key, setRaw);

  const set = useCallback(
    (v: string) => {
      setSearchParams({ [key]: v === defaultValue ? null : v });
    },
    [key, defaultValue],
  );

  return [value, set];
}

/** Sync an optional string param with the URL. Returns `null` when absent. */
export function useSearchParamNullable(key: string): [string | null, (v: string | null) => void] {
  const [value, setValue] = useState(() => getParam(key));

  useSyncOnPopState(key, setValue);

  const set = useCallback(
    (v: string | null) => {
      setSearchParams({ [key]: v });
    },
    [key],
  );

  return [value, set];
}

/** Sync an optional numeric param with the URL. Returns `null` when absent or NaN. */
export function useSearchParamNumber(key: string): [number | null, (v: number | null) => void] {
  const [value, setValue] = useState(() => {
    const raw = getParam(key);
    if (raw === null) return null;
    const n = Number(raw);
    return Number.isNaN(n) ? null : n;
  });

  const setFromUrl = useCallback((raw: string | null) => {
    if (raw === null) { setValue(null); return; }
    const n = Number(raw);
    setValue(Number.isNaN(n) ? null : n);
  }, []);
  useSyncOnPopState(key, setFromUrl);

  const set = useCallback(
    (v: number | null) => {
      setSearchParams({ [key]: v === null ? null : String(v) });
    },
    [key],
  );

  return [value, set];
}

/** Remove multiple search params from the URL at once. */
export function clearSearchParams(...keys: string[]) {
  const updates: Record<string, null> = {};
  for (const key of keys) {
    updates[key] = null;
  }
  setSearchParams(updates);
}
