"use client";
import { useEffect, useState } from "react";
import type { LoadedAveragePolicy } from "./onnx-policy";
import { PolicyBank } from "./policy-bank";

/** Poll small pointers; only download and validate changed model bundles. */
export function usePublishedPolicies() {
  const [models, setModels] = useState<Record<number, LoadedAveragePolicy>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let disposed = false;
    const bank = new PolicyBank();
    async function refresh() {
      try {
        const changed = await bank.refresh();
        if (!disposed) {
          if (changed) setModels({ ...bank.models });
          setError(null);
        }
      } catch (cause) {
        if (!disposed) setError(`Policy refresh failed; keeping the last validated models if available. ${cause instanceof Error ? cause.message : String(cause)}`);
      } finally { if (!disposed) setLoading(false); }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 60000);
    const focus = () => void refresh();
    window.addEventListener("focus", focus);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      window.removeEventListener("focus", focus);
      void bank.close().catch(console.error);
    };
  }, []);
  return { models, error, loading };
}
