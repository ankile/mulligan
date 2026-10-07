// The self-deploy build (vite.config.ts) loads this module wherever the shared screens import
// src/lib/arenaClient.ts, so every query and mutation goes to the configured Convex deployment.
export { useQuery, useQueries, useMutation, usePaginatedQuery } from "convex/react";
