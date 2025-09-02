// ui/src/components/PokerTableFrame.tsx
"use client";
import React from "react";

type Seat = {
  id: number;            // 0=SB, 1=BB, 2=BTN
  label: string;         // "SB" | "BB" | "BTN"
  stack: string;         // "27.0 BB"
  smallBlind?: boolean;
  bigBlind?: boolean;
  active?: boolean;
  cards?: string[];      // ["A♥","8♣","T♠","XX"]
};

export default function PokerTableFrame({
  children,
  seats,
  potLabel,
  heroSeat,
  board,
  className,
}: {
  children?: React.ReactNode;
  seats: Seat[];
  potLabel: string;
  heroSeat: number;
  board: string;
  className?: string;
}) {
  const order = [heroSeat, (heroSeat + 1) % 3, (heroSeat + 2) % 3];
  const hero = seats.find(s => s.id === order[0])!;
  const left = seats.find(s => s.id === order[1])!;
  const right = seats.find(s => s.id === order[2])!;

  return (
    <div
      className={
        "relative mx-auto aspect-[16/9] w-full max-w-[1100px] rounded-3xl p-4 " +
        "bg-[radial-gradient(ellipse_at_center,_#0b3866_0%,_#053056_50%,_#061c34_100%)] " +
        "shadow-inner ring-1 ring-border " +
        (className ?? "")
      }
    >
      {/* table hippodrome */}
      <div className="absolute inset-[1.5rem] h-2/3 w-full rounded-full border border-primary/30 shadow-[inset_0_0_2rem_rgba(34,211,238,.15)]" />

      {/* logo & pot */}
      <div className="absolute top-8 left-1/2 -translate-x-1/2 text-muted-foreground/50 tracking-widest text-sm select-none">
        EXPRESSO
      </div>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[36%] text-center">
        <div className="text-muted-foreground/80 text-xs">Pot total</div>
        <div className="mt-0.5 text-foreground text-lg font-semibold drop-shadow">{potLabel}</div>
      </div>

      {/* board */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[5%] flex gap-1">
        {board
          ? board.split(" ").map((t, i) => <PlayingCard key={i} text={t} />)
          : <div className="text-muted-foreground/80 text-sm">-</div>}
      </div>

      {/* sièges avec cartes */}
      <div className="absolute left-[12%] top-[16%] flex flex-col items-center">
        <SeatChip {...left} />
        {left.cards?.length ? (
          <div className="mt-2 flex gap-1">
            {left.cards.map((t, i) => <PlayingCard key={i} text={t} />)}
          </div>
        ) : null}
      </div>

      <div className="absolute right-[12%] top-[16%] flex flex-col items-center">
        <SeatChip {...right} />
        {right.cards?.length ? (
          <div className="mt-2 flex gap-1">
            {right.cards.map((t, i) => <PlayingCard key={i} text={t} />)}
          </div>
        ) : null}
      </div>

      <div className="absolute left-1/2 -translate-x-1/2 bottom-[8%] flex flex-col items-center">
        <SeatChip {...hero} highlight />
        {hero.cards?.length ? (
          <div className="mt-2 flex gap-2">
            {hero.cards.map((t, i) => <PlayingCard key={i} text={t} />)}
          </div>
        ) : null}
      </div>

      {/* actions */}
      <div className="absolute bottom-4 left-0 right-0 px-6">
        {children}
      </div>
    </div>
  );
}

function SeatChip({
  label,
  stack,
  smallBlind,
  bigBlind,
  active = true,
  highlight = false,
}: {
  label: string;
  stack: string;
  smallBlind?: boolean;
  bigBlind?: boolean;
  active?: boolean;
  highlight?: boolean;
}) {
  return (
    <div>
      <div
        className={"rounded-2xl px-3 py-1.5 shadow-lg backdrop-blur bg-card/80 ring-1 ring-border" +(highlight ? " outline outline-primary/60" : "")}>
        <div className="flex items-center gap-2">
          <div
            className={"h-6 w-6 rounded-full grid place-items-center text-[0.625rem] font-bold" +(active ? " bg-seat-active-bg" : " bg-seat-inactive-bg") + (active ? " text-seat-active-text" : " text-seat-inactive-text")}>
            {label}
          </div>
          <div className="leading-[1.05]">
            <div className="text-[0.6875rem] text-primary/90">{stack}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ===== cartes style casino ===== */
function PlayingCard({ text }: { text: string }) {
  if (text === "XX") {
    return (
      <div className="w-14 h-19 rounded-sm border border-neutral-700 shadow-lg bg-neutral-950 grid place-items-center">
        <div className="w-[80%] h-[80%] rounded-sm border border-border bg-neutral-800 grid place-items-center">
          <div className="w-[85%] h-[85%] rounded-sm border border-border bg-neutral-950 grid place-items-center">
            <span className="text-neutral-300">♠</span>
          </div>
        </div>
      </div>
    );
  }
  
  const r = text.slice(0, -1);
  const s = text.slice(-1);
  const isRed = s === "♥" || s === "♦";
  return (
    <div className="relative w-12 h-16 rounded-sm border shadow-lg bg-[var(--card-bg)] border-[var(--card-border)]">
      <div
        className={
          "absolute top-0.5 left-1 text-[0.625rem] font-bold " +
          (isRed ? "text-[var(--card-red)]" : "text-[var(--card-black)]")
        }
      >
        {r}
        <div className="-mt-0.5 leading-none">{s}</div>
      </div>
      <div
        className={
          "absolute right-1 bottom-0.5 rotate-180 text-[0.625rem] font-bold " +
          (isRed ? "text-[var(--card-red)]" : "text-[var(--card-black)]")
        }
      >
        {r}
        <div className="-mt-0.5 leading-none">{s}</div>
      </div>
      <div
        className={
          "absolute inset-0 grid place-items-center " +
          (isRed ? "text-[var(--card-red)]" : "text-[var(--card-black)]")
        }
      >
        <span className="text-xl">{s}</span>
      </div>
    </div>
  );
}
