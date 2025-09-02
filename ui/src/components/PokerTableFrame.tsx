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
      className={`relative mx-auto aspect-[16/9] w-[80%] rounded-3xl px-4 pt-2 
              bg-[radial-gradient(ellipse_at_center,_#0b3866_0%,_#053056_50%,_#061c34_100%)] 
              shadow-inner ring-1 ring-border overflow-hidden`}
      >
      {/* table hippodrome */}
      <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-[64%] w-[92%] h-[74%] rounded-full border border-primary/30 shadow-[inset_0_0_2rem_rgba(34,211,238,.15)]" />

      {/* logo & pot */}
      <div className="absolute top-8 left-1/2 -translate-x-1/2 text-muted-foreground/50 tracking-widest text-sm select-none">
        EXPRESSO
      </div>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[36%] text-center">
        <div className="text-muted-foreground/80 text-xs">Pot total</div>
        <div className="mt-0.5 text-foreground text-lg font-semibold drop-shadow">{potLabel}</div>
      </div>

      {/* board */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[125%] flex gap-1">
        {board
          ? board.split(" ").map((t, i) => <PlayingCard key={i} text={t} />)
          : <div className="text-muted-foreground/80 text-sm">-</div>}
      </div>

      {/* sièges avec cartes */}
      <div className="absolute left-[8%] top-[12%] flex flex-col items-center">
        <SeatChip {...left} />
        {left.cards?.length ? (
          <div className="mt-2 flex gap-1">
            {left.cards.map((t, i) => <PlayingCard key={i} text={t} />)}
          </div>
        ) : null}
      </div>

      <div className="absolute right-[8%] top-[12%] flex flex-col items-center">
        <SeatChip {...right} />
        {right.cards?.length ? (
          <div className="mt-2 flex gap-1">
            {right.cards.map((t, i) => <PlayingCard key={i} text={t} />)}
          </div>
        ) : null}
      </div>

      <div className="absolute left-1/2 -translate-x-1/2 bottom-[14%] flex flex-col items-center">
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
  active = true,
  highlight = false,
}: {
  label: string;
  stack: string;
  active?: boolean;
  highlight?: boolean;
}) {
  return (
    <div>
      <div
        className={
          "rounded-2xl px-3 py-1.5 shadow-lg backdrop-blur bg-card/80 ring-1 ring-border w-40 h-10 flex items-center" +
          (highlight ? " outline outline-primary/60" : "")
        }
      >
        <div className="flex items-center justify-between w-full">
          <div
            className={
              "h-7 w-12 rounded-full grid place-items-center text-sm font-bold" +
              (active
                ? " bg-[var(--seat-active-bg)] text-[var(--seat-active-text)]"
                : " bg-[var(--seat-inactive-bg)] text-[var(--seat-inactive-text)]")
            }
          >
            {label}
          </div>
          <div className="leading-[1.05]">
            <div className="text-md text-primary/90">{stack}</div>
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
      <div className="w-16 h-23 rounded-sm border border-neutral-700 shadow-lg bg-neutral-950 grid place-items-center">
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
    <div className="relative w-16 h-23 rounded-sm border shadow-lg bg-[var(--card-bg)] border-[var(--card-border)]">
      <div
        className={
          "absolute top-0.5 left-1 text-md font-bold " +
          (isRed ? "text-[var(--card-red)]" : "text-[var(--card-black)]")
        }
      >
        {r}
        <div className="-mt-0.5 leading-none">{s}</div>
      </div>
      <div
        className={
          "absolute right-1 bottom-0.5 rotate-180 text-md font-bold " +
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
        <span className="text-2xl">{s}</span>
      </div>
    </div>
  );
}
