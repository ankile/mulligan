import { createContext, useContext } from "react";
import { getFunctionName } from "convex/server";
import type * as Live from "convex/react";
import { createReleaseAdapter } from "./adapter";
export const ReleaseContext = createContext<ReturnType<
  typeof createReleaseAdapter
> | null>(null);
function useAdapter() {
  const a = useContext(ReleaseContext);
  if (!a) throw new Error("Missing release provider");
  return a;
}
export const useQuery: typeof Live.useQuery = (query, ...args) => {
  const adapter = useAdapter();
  return args[0] === "skip"
    ? undefined
    : adapter.resolve(getFunctionName(query), args[0]);
};
export const useQueries: typeof Live.useQueries = (queries) => {
  const adapter = useAdapter();
  return Object.fromEntries(
    Object.entries(queries).map(([key, { query, args }]) => [
      key,
      adapter.resolve(getFunctionName(query), args),
    ]),
  );
};
export const useMutation: typeof Live.useMutation = () => {
  const fail = () => {
    throw new Error("The public Arena is read-only");
  };
  return Object.assign(fail, { withOptimisticUpdate: fail });
};
export const usePaginatedQuery: typeof Live.usePaginatedQuery = () => {
  throw new Error("Review pagination is unavailable in the public release");
};
