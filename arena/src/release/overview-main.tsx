import { createRoot } from "react-dom/client";
import { validateRelease } from "./types";
import Overview from "./Overview";
import { trackOutboundClicks } from "./trackClicks";
trackOutboundClicks();
const root = createRoot(document.getElementById("root")!);
fetch("/data/release.json").then(r => {
  if (!r.ok) throw new Error(`Release returned ${r.status}`);
  return r.json();
}).then(raw => root.render(<Overview data={validateRelease(raw)} />))
.catch(error => root.render(<main><h1>Release unavailable</h1><p>{String(error)}</p></main>));
