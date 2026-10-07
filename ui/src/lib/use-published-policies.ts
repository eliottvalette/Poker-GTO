"use client";
import { useEffect, useState } from "react";
import { loadPublishedPolicies, type LoadedAveragePolicy } from "./onnx-policy";

/** All views share one model bank; atomic catalog changes select new exports. */
export function usePublishedPolicies() {
  const [models, setModels] = useState<Record<number, LoadedAveragePolicy>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let disposed = false;
    let running = false;
    let fingerprint = "";
    const sessions = new Set<LoadedAveragePolicy>();
    async function refresh() {
      if (running || disposed) return;
      running = true;
      try {
        const response = await fetch("/policy/index.json", { cache: "no-store" });
        if (!response.ok) throw new Error(`Policy catalog unavailable: HTTP ${response.status}; publish an export with migrate.py`);
        const catalog = await response.json();
        const nextFingerprint = JSON.stringify(catalog);
        if (nextFingerprint !== fingerprint) {
          const loaded = await loadPublishedPolicies(catalog);
          if (disposed) { await Promise.all(Object.values(loaded).map(model => model.release())); return; }
          Object.values(loaded).forEach(model => sessions.add(model));
          // Retain old sessions until unmount so ongoing queries can finish safely.
          setModels(loaded);
          fingerprint = nextFingerprint;
        }
        if (!disposed) setError(null);
      } catch (cause) {
        if (!disposed) setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        running = false;
        if (!disposed) setLoading(false);
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 15000);
    const focus = () => void refresh();
    window.addEventListener("focus", focus);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      window.removeEventListener("focus", focus);
      void Promise.all([...sessions].map(model => model.release())).catch(console.error);
    };
  }, []);
  return { models, error, loading };
}
