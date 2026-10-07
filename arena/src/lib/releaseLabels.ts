/**
 * Release-only display fields carried on the frozen adapter's session and round
 * objects (the Convex types do not declare them). Live sessions return undefined.
 */
export function releaseLabel(value: unknown): string | undefined {
  const label = (value as { label?: unknown } | null | undefined)?.label;
  return typeof label === "string" ? label : undefined;
}
