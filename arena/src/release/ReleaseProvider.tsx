import { useMemo } from "react";
import type { ReactNode } from "react";
import { ReleaseContext } from "./client";
import { createReleaseAdapter } from "./adapter";
import type { UISnapshot, SimStatistics } from "./adapter";
import type { Release } from "./types";
export function ReleaseProvider({
  data,
  ui,
  sim,
  children,
}: {
  data: Release;
  ui: UISnapshot;
  sim: SimStatistics;
  children: ReactNode;
}) {
  const adapter = useMemo(() => createReleaseAdapter(data, ui, sim), [data, ui, sim]);
  return (
    <ReleaseContext.Provider value={adapter}>
      {children}
    </ReleaseContext.Provider>
  );
}
