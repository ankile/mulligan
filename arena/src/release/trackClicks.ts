/**
 * Count followed links to other hosts and downloads on the hosted Mulligan snapshot. Clicks go to
 * the paper site's counter (mulligan-paper-website, `python -m clicks report`); copies deployed
 * elsewhere send nothing.
 */
const HOSTED = "https://arena.mulligan.page";
const ENDPOINT = "https://mulligan.page/api/click";

function trackClick(event: MouseEvent) {
  const link = (event.target as Element).closest("a[href]") as HTMLAnchorElement | null;
  if (!link) return;
  const url = new URL(link.href, location.href);
  if (url.origin === location.origin && !link.hasAttribute("download") && !url.pathname.endsWith(".pdf")) return;
  const label = link.textContent!.replace(/\s+/g, " ").trim() || link.getAttribute("aria-label") || "";
  navigator.sendBeacon(ENDPOINT, JSON.stringify({ href: url.href, label: label.slice(0, 120), page: location.pathname + location.search }));
}

export function trackOutboundClicks() {
  if (location.origin !== HOSTED) return;
  document.addEventListener("click", trackClick, true);
  // Middle-click opens a new tab without a click event.
  document.addEventListener("auxclick", (event) => { if (event.button === 1) trackClick(event); }, true);
}
