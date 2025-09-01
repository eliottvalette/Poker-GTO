"use client";
import React from "react";
import { cn } from "@/lib/utils";

type Seat = {
  id: number;
  label: string;
  stack: string;
  smallBlind?: boolean;
  bigBlind?: boolean;
  active?: boolean;
};

export default function PokerTableFrame({
  children,
  seats,
  potLabel,
  heroSeat,
  board,
  className,
}: {
  children?: React.ReactNode;               // zone d'actions (boutons)
  seats: Seat[];                            // 3 sièges pour ton Expresso
  potLabel: string;                         // ex: "Pot total : 2,5 BB"
  heroSeat: number;                         // 0..2
  board: string;                            // ex: "A♥ 8♣"
  className?: string;
}) {
  return (
    <div
      className={cn(
        "relative mx-auto aspect-[16/9] w-full max-w-[1100px] rounded-3xl p-4",
        "bg-[#0a0f1b] shadow-[0_0_40px_rgba(0,0,0,.5)] overflow-hidden",
        className
      )}
    >
      {/* fond métal + gems */}
      <div className="absolute inset-0">
        <div className="absolute inset-[-20%] bg-[radial-gradient(closest-side,rgba(63,120,255,.15),rgba(0,0,0,0))] blur-3xl" />
        <div className="absolute inset-0 bg-[conic-gradient(from_0deg_at_50%_50%,#1f2937_0deg,#0b1325_120deg,#1f2937_240deg,#0b1325_360deg)] opacity-60" />
      </div>

      {/* anneaux décoratifs */}
      <div className="absolute inset-6 rounded-[3rem] border border-white/10 shadow-inner" />
      <div className="absolute inset-8 rounded-[2.6rem] bg-gradient-to-b from-white/10 to-transparent" />

      {/* feutre */}
      <div className="absolute inset-12 rounded-[2.2rem] bg-[radial-gradient(ellipse_at_center,_#0b3866_0%,_#053056_50%,_#061c34_100%)] shadow-inner ring-1 ring-white/10" />

      {/* piste ovale brillante */}
      <div className="absolute inset-[68px] rounded-[2rem] bg-[radial-gradient(ellipse_at_center,_rgba(17,173,246,.25),rgba(3,56,90,.15)_55%,rgba(2,28,48,.0)_70%)]" />
      <div className="absolute inset-[78px] rounded-[1.8rem] border border-cyan-400/30 shadow-[0_0_30px_8px_rgba(34,211,238,.15)_inset]" />

      {/* logo/limite */}
      <div className="absolute top-8 left-1/2 -translate-x-1/2 text-cyan-200/50 tracking-widest text-sm select-none">
        EXPRESSO
      </div>

      {/* pot */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[36%] text-center">
        <div className="text-cyan-200/80 text-xs">Pot total</div>
        <div className="mt-0.5 text-cyan-100 text-lg font-semibold drop-shadow">{potLabel}</div>
      </div>

      {/* board */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[5%] text-cyan-50/95 text-sm font-medium">
        {board || "-"}
      </div>

      {/* sièges */}
      <SeatChip
        style="top-[14%] left-1/2 -translate-x-1/2"
        {...seats[2]}
        highlight={heroSeat===2}
      />
      <SeatChip
        style="left-[12%] top-1/2 -translate-y-1/2"
        {...seats[0]}
        highlight={heroSeat===0}
      />
      <SeatChip
        style="right-[12%] top-1/2 -translate-y-1/2"
        {...seats[1]}
        highlight={heroSeat===1}
      />

      {/* zone actions */}
      <div className="absolute bottom-4 left-0 right-0 px-6">
        {children}
      </div>
    </div>
  );
}

function SeatChip({
  style,
  label,
  stack,
  smallBlind,
  bigBlind,
  active,
  highlight,
}: {
  style: string;
  id?: number;
  label: string;
  stack: string;
  smallBlind?: boolean;
  bigBlind?: boolean;
  active?: boolean;
  highlight?: boolean;
}) {
  return (
    <div className={cn("absolute", style)}>
      <div
        className={cn(
          "rounded-2xl px-3 py-1.5 shadow-lg backdrop-blur",
          "bg-white/8 ring-1 ring-white/15",
          highlight ? "outline outline-2 outline-cyan-400/60" : ""
        )}
      >
        <div className="flex items-center gap-2">
          <div className={cn(
            "h-6 w-6 rounded-full grid place-items-center text-[10px] font-bold",
            active===false ? "bg-gray-600/50 text-gray-300" : "bg-amber-400/90 text-black",
          )}>
            {label.slice(0,1)}
          </div>
          <div className="leading-[1.05]">
            <div className="text-[11px] text-white/80 font-medium">{label}</div>
            <div className="text-[11px] text-cyan-200/90">{stack}</div>
          </div>
          {smallBlind && <BlindBadge text="SB" />}
          {bigBlind && <BlindBadge text="BB" />}
        </div>
      </div>
    </div>
  );
}

function BlindBadge({ text }: { text: "SB"|"BB" }) {
  return (
    <div className={cn(
      "ml-1 rounded-md px-1.5 py-0.5 text-[10px] font-bold",
      text==="SB" ? "bg-red-500/90 text-white" : "bg-red-600/90 text-white"
    )}>{text}</div>
  );
}
