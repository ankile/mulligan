import { navigateToAppTab } from "../lib/appNavigation";

const TABS = [
  { id: "leaderboard", label: "Leaderboard" },
  { id: "sessions", label: "Eval Sessions" },
  { id: "pairings", label: "Pairings" },
  { id: "explorer", label: "Data Explorer" },
  { id: "coverage", label: "Coverage" },
];

export function AppTabNavigation({ activeTab }: { activeTab: string }) {
  return (
    <div className="flex flex-wrap gap-1 bg-warm-100 rounded-xl p-1 w-fit">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => navigateToAppTab(tab.id)}
          className={`px-5 py-2 rounded-lg text-sm font-medium transition-all duration-150 cursor-pointer ${
            activeTab === tab.id ? "bg-white text-ink shadow-sm" : "text-ink-muted hover:text-ink"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
