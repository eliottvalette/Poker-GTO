/** Shared catalog routing for the UI and Web Worker. No secret credentials are used. */
export type PolicyCatalog = {
  version: number;
  exports: Record<string, { bundle_id: string; iteration: number }>;
};

export function policyStorageBase(): string | null {
  const source = process.env.NEXT_PUBLIC_POLICY_SOURCE ?? "local";
  if (source === "local") return null;
  if (source !== "supabase") throw new Error(`Unsupported policy source: ${source}`);
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const bucket = process.env.NEXT_PUBLIC_SUPABASE_POLICY_BUCKET;
  if (!url || !bucket || !/^[a-z0-9][a-z0-9-]*$/.test(bucket)) throw new Error("Supabase policy URL/bucket is missing or invalid");
  const origin = new URL(url);
  if (origin.protocol !== "https:" || origin.pathname !== "/") throw new Error("Supabase URL must be an HTTPS project origin");
  return `${origin.origin}/storage/v1/object/public/${bucket}`;
}

export async function fetchPolicyCatalog(assetOrigin?: string): Promise<PolicyCatalog> {
  const base = policyStorageBase();
  if (!base) {
    const response = await fetch(new URL("/policy/index.json", assetOrigin ?? globalThis.location.origin), { cache: "no-store", signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error(`Local policy catalog unavailable: HTTP ${response.status}`);
    return response.json();
  }
  const entries = await Promise.all((["hu", "3max"] as const).map(async track => {
    const response = await fetch(`${base}/${track}/current.json?refresh=${Math.floor(Date.now()/60000)}`, { cache: "no-store", signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error(`Supabase ${track} policy unavailable: HTTP ${response.status}`);
    const pointer = await response.json();
    if (pointer.version !== 1 || pointer.track !== track || !/^[a-f0-9]{64}$/.test(pointer.release_id)
        || !Number.isSafeInteger(pointer.iteration) || pointer.iteration < 1) throw new Error(`Invalid Supabase ${track} pointer`);
    return [track, { bundle_id: pointer.release_id, iteration: pointer.iteration }];
  }));
  return { version: 1, exports: Object.fromEntries(entries) };
}

export function policyArtifactBase(track: string, release: string, assetOrigin?: string): string {
  const storage = policyStorageBase();
  return storage ? `${storage}/${track}/releases/${release}/average_${track}`
    : new URL(`/policy/releases/${release}/average_${track}`, assetOrigin ?? globalThis.location.origin).href;
}
