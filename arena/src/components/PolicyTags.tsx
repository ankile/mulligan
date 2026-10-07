import { useId, useState } from "react";
import { useMutation, useQuery } from "../lib/arenaClient";
import { api } from "../release/api";
import { SUGGESTED_METHODS, type PolicyTags as Tags } from "../../convex/policyTags";

export function PolicyTagBadges({ policy }: { policy: Tags }) {
  return <div className="flex flex-wrap gap-1.5 text-[11px] font-medium">
    {policy.round !== undefined && <span className="rounded-full px-2 py-0.5 bg-gold-light text-gold">Round {policy.round}</span>}
    {policy.method && <span className="rounded-full px-2 py-0.5 bg-teal-light text-teal">{policy.method}</span>}
    {policy.tags?.map((tag) => <span key={tag} className="rounded-full px-2 py-0.5 bg-warm-100 text-ink-muted">{tag}</span>)}
  </div>;
}

export function PolicyTagEditor({ policy }: { policy: Tags & { model_id: string } }) {
  const [editing, setEditing] = useState(false);
  return <div className="mt-4">
    {editing ? <TagForm policy={policy} onClose={() => setEditing(false)} /> :
      <button className="text-xs text-teal hover:underline cursor-pointer" onClick={() => setEditing(true)}>Edit tags</button>}
  </div>;
}

function TagForm({ policy, onClose }: { policy: Tags & { model_id: string }; onClose: () => void }) {
  const id = useId();
  const options = useQuery(api.policies.tagOptions);
  const save = useMutation(api.policies.setTags);
  const [round, setRound] = useState(policy.round?.toString() ?? "");
  const [method, setMethod] = useState(policy.method ?? "");
  const [tags, setTags] = useState((policy.tags ?? []).join("\n"));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputClass = "w-full rounded-lg border border-warm-200 bg-white px-3 py-2 text-xs";
  return <form className="space-y-3" onSubmit={async (event) => {
    event.preventDefault();
    setPending(true);
    setError(null);
    // A failed network save must leave the draft editable and show its error.
    try {
      await save({ model_id: policy.model_id, round: round === "" ? null : Number(round), method: method || null, tags: tags.split("\n") });
      onClose();
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    } finally {
      setPending(false);
    }
  }}>
    <fieldset disabled={pending} className="space-y-3">
      <label className="block text-xs text-ink-muted">Round
        <input type="number" min="0" step="1" placeholder="Unassigned" value={round} onChange={(e) => setRound(e.target.value)} className={inputClass} />
      </label>
      <label className="block text-xs text-ink-muted">Method
        <input list={`${id}-methods`} maxLength={100} placeholder="Choose or enter a method" value={method} onChange={(e) => setMethod(e.target.value)} className={inputClass} />
      </label>
      <datalist id={`${id}-methods`}>{[...new Set([...SUGGESTED_METHODS, ...(options?.methods ?? [])])].map((m) => <option key={m} value={m} />)}</datalist>
      <label className="block text-xs text-ink-muted">Additional tags, one per line
        <textarea rows={3} value={tags} onChange={(e) => setTags(e.target.value)} className={inputClass} placeholder="e.g. relative action" />
      </label>
      <div className="flex gap-3 text-xs">
        <button type="submit" className="rounded-lg bg-teal px-3 py-2 text-white cursor-pointer disabled:opacity-50">{pending ? "Saving..." : "Save tags"}</button>
        <button type="button" onClick={onClose} className="text-ink-muted cursor-pointer">Cancel</button>
      </div>
    </fieldset>
    {error && <p role="alert" className="text-xs text-rose-bar">{error}</p>}
  </form>;
}
