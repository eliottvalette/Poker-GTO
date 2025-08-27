"use client";
import React from "react";
import { ACTIONS, ACTIONS_L, ACTION_COLORS, type GridMix } from "@/lib/policy";

const CARD_LABS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"] as const;
type ActionU = typeof ACTIONS[number];
type ActionL = typeof ACTIONS_L[number];

function Cell({ mix, labelThresholdPct=18 }: { mix: GridMix; labelThresholdPct?: number }) {
  const order: { upper: ActionU; lower: ActionL; p: number }[] =
    ACTIONS.map((a, idx) => ({ upper: a, lower: ACTIONS_L[idx], p: mix[a] ?? 0 }));

  let cursor = 0;
  return (
    <div className="relative w-16 h-10 bg-zinc-900 border border-white/10 hover:ring-1 hover:ring-white/40">
      {order.map(({ upper, lower, p }) => {
        if (p <= 0) return null;
        const left = `${cursor * 100}%`;
        const width = `${p * 100}%`;
        cursor += p;
        return (
          <div
            key={upper}
            className="absolute inset-y-0 flex items-center justify-center text-[10px] font-semibold"
            style={{ left, width, background: ACTION_COLORS[lower] }}
          >
            {p * 100 >= labelThresholdPct ? `${Math.round(p * 100)}%` : null}
          </div>
        );
      })}
    </div>
  );
}

export default function Grid169({ gridMixes, labelThresholdPct=18 }:{
  gridMixes: GridMix[]; labelThresholdPct?: number;
}) {
  return (
    <div className="overflow-auto">
      <div className="inline-grid" style={{ gridTemplateColumns: `auto repeat(13, 4rem)` }}>
        <div />
        {CARD_LABS.map((c) => (
          <div key={`x-${c}`} className="h-8 flex items-center justify-center text-xs text-zinc-300">{c}</div>
        ))}
        {CARD_LABS.map((r, i) => (
          <React.Fragment key={`row-${i}`}>
            <div className="w-8 h-10 flex items-center justify-center text-xs text-zinc-300">{r}</div>
            {Array.from({ length: 13 }).map((_, j) => {
              const hidx = i * 13 + j;
              return <Cell key={hidx} mix={gridMixes[hidx]} labelThresholdPct={labelThresholdPct} />;
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
