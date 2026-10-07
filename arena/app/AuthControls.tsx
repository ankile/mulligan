import { useAuthActions } from "@convex-dev/auth/react";
import { useQuery } from "../src/lib/arenaClient";
import { api } from "../convex/_generated/api";

export default function AuthControls() {
  const { signIn, signOut } = useAuthActions();
  const viewer = useQuery(api.users.viewer);

  if (viewer === undefined) return null;

  if (viewer === null) {
    return (
      <button
        onClick={() => void signIn("huggingface")}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white border border-warm-200 text-sm font-medium text-ink shadow-sm hover:shadow transition-all duration-150 cursor-pointer"
      >
        <span aria-hidden>🤗</span>
        Sign in with Hugging Face
      </button>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        {viewer.image && (
          <img
            src={viewer.image}
            alt=""
            className="w-7 h-7 rounded-full border border-warm-200"
          />
        )}
        <span className="text-sm font-medium text-ink">
          {viewer.username ?? viewer.name}
        </span>
        {viewer.isEditor && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-teal/10 text-teal font-medium">
            editor
          </span>
        )}
      </div>
      <button
        onClick={() => void signOut()}
        className="text-sm text-ink-muted hover:text-ink transition-colors duration-150 cursor-pointer"
      >
        Sign out
      </button>
    </div>
  );
}
