// The Convex runtime exposes deployment environment variables via process.env
// (set with `npx convex env set NAME value`). This is the only Node-ism
// available in the default Convex runtime, so declare it narrowly instead of
// pulling in full @types/node.
declare const process: {
  env: Record<string, string | undefined>;
};
