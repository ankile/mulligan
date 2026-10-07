/**
 * Count page views and followed links to other hosts and downloads on the hosted Mulligan snapshot.
 * Events go to the paper site's counter (mulligan-paper-website, `python -m analytics report`); copies
 * deployed elsewhere send nothing.
 */
const HOSTED = "https://arena.mulligan.page";
const ENDPOINT = "https://mulligan.page/api/collect";

const send = (event: Record<string, string>) =>
  navigator.sendBeacon(ENDPOINT, JSON.stringify({ page: location.pathname + location.search, ...event }));

function trackClick(event: MouseEvent) {
  const link = (event.target as Element).closest("a[href]") as HTMLAnchorElement | null;
  if (!link) return;
  const url = new URL(link.href, location.href);
  if (url.origin === location.origin && !link.hasAttribute("download") && !url.pathname.endsWith(".pdf")) return;
  const label = link.textContent!.replace(/\s+/g, " ").trim() || link.getAttribute("aria-label") || "";
  send({ type: "click", href: url.href, label: label.slice(0, 120) });
}

export function trackVisits() {
  if (location.origin !== HOSTED) return;
  const referrer = document.referrer && new URL(document.referrer).host;
  const params = new URLSearchParams(location.search);
  const source = params.get("utm_source") || params.get("ref") || "";
  // Drop the tag from the address bar before counting the page, so a copied link does not carry it to new visitors.
  if (params.has("ref") || params.has("utm_source")) {
    params.delete("ref");
    params.delete("utm_source");
    history.replaceState(history.state, "", location.pathname + (params.size ? `?${params}` : "") + location.hash);
  }
  send({ type: "view", referrer: referrer !== location.host ? referrer : "", source });
  document.addEventListener("click", trackClick, true);
  // Middle-click opens a new tab without a click event.
  document.addEventListener("auxclick", (event) => { if (event.button === 1) trackClick(event); }, true);
}
