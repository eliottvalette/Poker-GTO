"use client";
import React from "react";
import { ACTIONS_L, ACTION_COLORS } from "@/src/lib/policy";

export default function Legend() {
  return (
    <div className="flex gap-4 flex-wrap text-xs mt-2">
      {ACTIONS_L.map(a => (
        <div key={a} className="flex items-center gap-2">
          <span className="inline-block w-4 h-3 rounded" style={{background: ACTION_COLORS[a]}}/>
          <span className="uppercase tracking-wide">{a}</span>
        </div>
      ))}
    </div>
  );
}
