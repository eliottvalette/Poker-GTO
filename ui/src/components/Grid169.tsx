// ui/src/components/Grid169.tsx
"use client";
import React from "react";
import { GROUPED_ACTIONS, GROUP_COLORS, groupActions, type GridMix } from "@/lib/policy";

const CARD_LABS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"] as const;
type GroupedAction = typeof GROUPED_ACTIONS[number];

const RED_HEX   = "#D62828"; // 100% raise/all-in
const BLUE_HEX  = "#006DAA"; // 100% fold
const NO_DATA_BG = "#F3F3F3";

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)!;
  return [parseInt(m[1],16), parseInt(m[2],16), parseInt(m[3],16)];
}
function rgbToHex([r,g,b]: [number,number,number]) {
  const h = (n:number)=>Math.max(0,Math.min(255,Math.round(n))).toString(16).padStart(2,"0");
  return `#${h(r)}${h(g)}${h(b)}`;
}
function lerpColor(aHex: string, bHex: string, t: number): string {
  const a = hexToRgb(aHex), b = hexToRgb(bHex);
  return rgbToHex([a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]);
}

function Cell({
  mix,
  heatmapMode,
  labelThresholdPct = 18,
}: { mix: GridMix; heatmapMode: boolean; labelThresholdPct?: number }) {
  const grouped = groupActions(mix);

  if (heatmapMode) {
    const f = grouped.fold;
    const r = grouped.raise_allin;
    const denom = f + r;

    // t = 0 -> rouge (100% raise), t = 1 -> bleu (100% fold)
    const t = denom > 0 ? f / denom : NaN;
    const bg = Number.isNaN(t) ? NO_DATA_BG : lerpColor(RED_HEX, BLUE_HEX, t);

    const mainPct = Math.max(f, r) * 100;

    return (
      <div
        className="relative w-20 h-16 border border-border hover:ring-1 hover:ring-ring"
        style={{ background: bg }}
        title={`fold: ${(f*100).toFixed(0)}% | raise/all-in: ${(r*100).toFixed(0)}%`}
      >
        {mainPct >= labelThresholdPct ? (
          <div className="absolute inset-0 flex items-center justify-center text-[8px] font-semibold">
            {Math.round(mainPct)}%
          </div>
        ) : null}
      </div>
    );
  }

  // Mode barres (inchangé)
  const order: { action: GroupedAction; p: number }[] =
    ["raise_allin","check_call","fold"].map(a => ({ action: a as GroupedAction, p: grouped[a as GroupedAction] ?? 0 }));
  let cursor = 0;
  return (
    <div className="relative w-20 h-16 bg-muted border border-border hover:ring-1 hover:ring-ring">
      {order.map(({ action, p }) => {
        if (p <= 0) return null;
        const left = `${cursor * 100}%`;
        const width = `${p * 100}%`;
        cursor += p;
        return (
          <div
            key={action}
            className="absolute inset-y-0 flex items-center justify-center text-[8px] font-semibold"
            style={{ left, width, background: GROUP_COLORS[action] }}
            title={`${action}: ${(p*100).toFixed(0)}%`}
          >
            {p * 100 >= labelThresholdPct ? `${Math.round(p * 100)}%` : null}
          </div>
        );
      })}
    </div>
  );
}

export default function Grid169({
  gridMixes, heatmapMode, labelThresholdPct=18,
}: { gridMixes: GridMix[]; heatmapMode: boolean; labelThresholdPct?: number; }) {
  return (
    <div className="overflow-auto">
      <div className="inline-grid" style={{ gridTemplateColumns: `auto repeat(13, 5rem)` }}>
        <div />
        {CARD_LABS.map((c) => (
          <div key={`x-${c}`} className="h-8 flex items-center justify-center text-xs text-muted-foreground">{c}</div>
        ))}
        {CARD_LABS.map((r, i) => (
          <React.Fragment key={`row-${i}`}>
            <div className="w-8 h-16 flex items-center justify-center text-xs text-muted-foreground">{r}</div>
            {Array.from({ length: 13 }).map((_, j) => {
              const hidx = i * 13 + j;
              return (
                <Cell
                  key={hidx}
                  mix={gridMixes[hidx]}
                  heatmapMode={heatmapMode}
                  labelThresholdPct={labelThresholdPct}
                />
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
