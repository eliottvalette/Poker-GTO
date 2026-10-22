"use client";

import { useEffect, useState } from "react";
import { policyStorageBase } from "@/lib/policy-catalog";

type Status = { track: "hu" | "3max"; iteration: number; updated_at: string; iteration_seconds: number };

export default function SelfPlayStatus() {
  const [rows, setRows] = useState<Status[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    let alive = true;
    let controller = new AbortController();
    async function load() {
      controller.abort();
      controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 15000);
      try {
        const base = policyStorageBase();
        if (!base) throw new Error("Continuous training status requires Supabase configuration.");
        const next = await Promise.all((["hu", "3max"] as const).map(async track => {
          const response = await fetch(`${base}/selfplay/${track}/status.json?refresh=${Date.now()}`, { cache: "no-store", signal: controller.signal });
          if (!response.ok) throw new Error(`${track}: self-play status unavailable (HTTP ${response.status}).`);
          const value = await response.json();
          if (value.engine !== "continuous_selfplay_v1" || value.track !== track || value.iteration_limit !== null
            || !Number.isSafeInteger(value.iteration) || value.iteration < 1
            || !Number.isFinite(Date.parse(value.updated_at)) || !Number.isFinite(value.iteration_seconds)
            || value.iteration_seconds < 0) throw new Error(`${track}: invalid self-play status.`);
          return value as Status;
        }));
        if (alive) { setRows(next); setError(""); }
      } catch (e) { if (alive) setError(e instanceof Error ? e.message : String(e)); }
      finally { clearTimeout(timer); }
    }
    void load();
    const timer = setInterval(() => { if (!document.hidden) void load(); }, 60000);
    return () => { alive = false; controller.abort(); clearInterval(timer); };
  }, []);
  return <div className="mx-auto max-w-6xl space-y-2 py-3">
    <h2 className="text-xl font-semibold">Continuous self-play</h2>
    <p className="text-sm text-muted-foreground">No iteration limit. Checkpoints resume after restarts. Experimental policies; poker strength is not certified.</p>
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    <div className="flex flex-wrap gap-6">{rows.map(row => <div key={row.track}>
      <strong>{row.track === "hu" ? "HU" : "3-Max"} · Iteration {row.iteration.toLocaleString()}</strong>
      <p className="text-xs text-muted-foreground">{row.iteration_seconds.toFixed(1)} s per last iteration · {new Date(row.updated_at).toLocaleString()}</p>
      {Date.now() - Date.parse(row.updated_at) > Math.max(180000, row.iteration_seconds * 3000) && <p className="text-sm text-amber-500">No recent completed iteration; check the training service.</p>}
    </div>)}</div>
  </div>;
}
