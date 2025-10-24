// --- hooks/useUrlState.ts ---
import { useEffect } from "react";

export function useUrlState(stateObj: object) {
  // push state to URL when stateObj changes
  useEffect(() => {
    const params = new URLSearchParams();
    params.set("s", btoa(unescape(encodeURIComponent(JSON.stringify(stateObj)))));
    const url = `${location.pathname}?${params.toString()}`;
    history.replaceState(null, "", url);
  }, [JSON.stringify(stateObj)]);

  // read state from URL once
  function readInitial<T>(fallback: T): T {
    try {
      const raw = new URLSearchParams(location.search).get("s");
      if (!raw) return fallback;
      return JSON.parse(decodeURIComponent(escape(atob(raw)))) as T;
    } catch {
      return fallback;
    }
  }

  return { readInitial };
}
