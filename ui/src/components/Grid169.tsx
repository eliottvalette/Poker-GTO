"use client";
import React from "react";
import { GROUPED_ACTIONS, GROUP_COLORS, type GroupedGridMix, groupActions, type GridMix } from "@/lib/policy";

const CARD_LABS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"] as const;
type GroupedAction = typeof GROUPED_ACTIONS[number];

function Cell({ mix, labelThresholdPct=18 }: { mix: GridMix; labelThresholdPct?: number }) {
  const groupedMix = groupActions(mix);
  
  const order: { action: GroupedAction; p: number }[] =
    GROUPED_ACTIONS.map((a) => ({ action: a, p: groupedMix[a] ?? 0 }));

  let cursor = 0;
  return (
    <div className="relative w-16 h-10 bg-muted border border-border hover:ring-1 hover:ring-ring">
      {order.map(({ action, p }) => {
        if (p <= 0) return null;
        const left = `${cursor * 100}%`;
        const width = `${p * 100}%`;
        cursor += p;
        return (
          <div
            key={action}
            className="absolute inset-y-0 flex items-center justify-center text-[10px] font-semibold"
            style={{ left, width, background: GROUP_COLORS[action] }}
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
          <div key={`x-${c}`} className="h-8 flex items-center justify-center text-xs text-muted-foreground">{c}</div>
        ))}
        {CARD_LABS.map((r, i) => (
          <React.Fragment key={`row-${i}`}>
            <div className="w-8 h-10 flex items-center justify-center text-xs text-muted-foreground">{r}</div>
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
