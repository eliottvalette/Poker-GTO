// ui/src/components/Legend.tsx
"use client";
import React from "react";
import { GROUPED_ACTIONS, GROUP_COLORS, GROUPED_LABELS } from "@/lib/policy";

export default function Legend() {
  return (
    <div className="flex gap-4 flex-wrap text-xs mt-2">
      {GROUPED_ACTIONS.map((a, idx) => (
        <div key={a} className="flex items-center gap-2">
          <span className="inline-block w-4 h-3 rounded" style={{ background: GROUP_COLORS[a] }} />
          <span className="uppercase tracking-wide">{GROUPED_LABELS[idx]}</span>
        </div>
      ))}
    </div>
  );
}
