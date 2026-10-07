// Keyboard-shortcut help modal. The shell is shared; each review surface
// passes its own keymap table and title.

export function HelpOverlay({
  title,
  keys,
  onClose,
}: {
  title: string;
  keys: [string, string][];
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 px-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl border border-warm-200 shadow-lg p-6 max-w-md w-full"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="font-display text-lg text-ink mb-3">{title}</h3>
        <dl className="space-y-1.5">
          {keys.map(([key, description]) => (
            <div key={key} className="flex items-baseline gap-3 text-xs">
              <dt className="font-mono text-ink w-28 flex-shrink-0">{key}</dt>
              <dd className="text-ink-muted font-body">{description}</dd>
            </div>
          ))}
        </dl>
        <button
          onClick={onClose}
          className="mt-4 px-3 py-1.5 rounded-lg border border-warm-200 text-xs text-ink-muted hover:bg-warm-50 cursor-pointer"
        >
          Close
        </button>
      </div>
    </div>
  );
}
