// The release build serves every query from the frozen export through this boundary.
export { useQuery, useQueries, useMutation, usePaginatedQuery } from "../release/client";
