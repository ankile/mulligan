import { getAuthUserId } from "@convex-dev/auth/server";
import type { Id } from "./_generated/dataModel";
import type { MutationCtx, QueryCtx } from "./_generated/server";

declare const process: {
  env: Record<string, string | undefined>;
};

/**
 * Access control for arena writes.
 *
 * Two principals may reach these mutation handlers:
 *  - Allowlisted humans, signed in with Hugging Face OAuth. The allowlist is
 *    the ARENA_EDITOR_SUBS env var: comma-separated HF OIDC `sub` values (the
 *    stable account ids stored as authAccounts.providerAccountId). Usernames
 *    are display-only — HF usernames are MUTABLE, so keying authorization on
 *    them would let a rename orphan or impersonate an editor.
 *  - The machine HTTP API, after authenticating a per-machine key and scope,
 *    presenting the server-only ARENA_SERVICE_TOKEN bridge value.
 *
 * Both env vars are required for their respective paths; missing config fails
 * closed with a descriptive error.
 */

function editorSubAllowlist(): string[] {
  return (process.env.ARENA_EDITOR_SUBS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

/** The signed-in user's HF OIDC sub (authAccounts.providerAccountId), or null. */
async function viewerSub(
  ctx: QueryCtx | MutationCtx,
  userId: Id<"users">
): Promise<string | null> {
  // .first(), not .unique(): this feeds the NON-THROWING viewerIsEditor path
  // (surfaced by users:viewer, an app-wide query) — a duplicate authAccounts
  // row must degrade to "not an editor", never take down the viewer query.
  const account = await ctx.db
    .query("authAccounts")
    .withIndex("userIdAndProvider", (q) =>
      q.eq("userId", userId).eq("provider", "huggingface")
    )
    .first();
  return account?.providerAccountId ?? null;
}

/** Require a signed-in, allowlisted human editor. Returns their HF username
 * (the display/audit string; authorization itself keys on the OIDC sub). */
export async function requireEditor(
  ctx: QueryCtx | MutationCtx
): Promise<string> {
  const userId = await getAuthUserId(ctx);
  if (userId === null) {
    throw new Error("Not signed in — this action requires authentication");
  }
  const sub = await viewerSub(ctx, userId);
  if (!sub) {
    throw new Error("Signed-in user has no Hugging Face account id on record");
  }
  const allowlist = editorSubAllowlist();
  if (allowlist.length === 0) {
    throw new Error(
      "ARENA_EDITOR_SUBS is not configured on this deployment — no editors are allowlisted"
    );
  }
  if (!allowlist.includes(sub)) {
    throw new Error(`HF account "${sub}" is not an allowlisted editor`);
  }
  const user = await ctx.db.get(userId);
  const username = user?.username;
  if (!username) {
    // The username is the audit string recorded on every review row; an
    // allowlisted sub with no username on record would silently write opaque
    // attributions — fail loud instead (matches the pre-hardening behavior).
    throw new Error("Signed-in user has no Hugging Face username on record");
  }
  return username;
}

/**
 * Require either the server-only HTTP Action bridge token or an allowlisted
 * signed-in editor. Returns the acting principal for audit purposes.
 */
export async function requireEditorOrService(
  ctx: QueryCtx | MutationCtx,
  serviceToken: string | undefined
): Promise<string> {
  if (serviceToken !== undefined) {
    const expected = process.env.ARENA_SERVICE_TOKEN;
    if (!expected) {
      throw new Error(
        "ARENA_SERVICE_TOKEN is not configured on this deployment"
      );
    }
    if (serviceToken !== expected) {
      throw new Error("Invalid service token");
    }
    return "service";
  }
  return requireEditor(ctx);
}

/** Non-throwing check used by the UI to decide whether to show edit affordances. */
export async function viewerIsEditor(ctx: QueryCtx): Promise<boolean> {
  const userId = await getAuthUserId(ctx);
  if (userId === null) return false;
  const sub = await viewerSub(ctx, userId);
  if (!sub) return false;
  return editorSubAllowlist().includes(sub);
}
