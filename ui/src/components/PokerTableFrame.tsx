// ui/src/components/PokerTableFrame.tsx
"use client";
import React from "react";
import { cn } from "@/lib/utils";

type Seat = {
  id: number;            // 0=SB, 1=BB, 2=BTN
  label: string;         // "SB" | "BB" | "BTN"
  stack: string;         // "27.0 BB"
  smallBlind?: boolean;
  bigBlind?: boolean;
  active?: boolean;
  cards?: string[];      // ["A♥","8♣"] (optionnel, affiché pour le héros)
};

export default function PokerTableFrame({
  children,
  seats,
  potLabel,
  heroSeat,             // 0..2
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
  // ordre visuel en fonction du siège héros :
  // hero -> bas centre, suivant -> haut gauche, suivant -> haut droite
  const order = [heroSeat, (heroSeat + 1) % 3, (heroSeat + 2) % 3];
  const hero = seats.find(s => s.id === order[0])!;
  const left = seats.find(s => s.id === order[1])!;
  const right= seats.find(s => s.id === order[2])!;

  return (
    <div
      className={cn(
        "relative mx-auto aspect-[16/9] w-full max-w-[1100px] rounded-3xl p-4",
        "bg-[radial-gradient(ellipse_at_center,_#0b3866_0%,_#053056_50%,_#061c34_100%)] shadow-inner ring-1 ring-border",
        className
      )}
    >
      <div className="absolute inset-[1.5rem] h-3/4 rounded-full border-primary/30 border-4 shadow-[inset_0_0_6rem_rgba(34,211,238,.15)]" />

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
          ? board.split(" ").map((t,i)=> <PlayingCard key={i} text={t} />)
          : <div className="text-muted-foreground/80 text-sm">-</div>}
      </div>

      {/* sièges positionnés */}
      <SeatChip style="left-[12%] top-[16%]" {...left} />
      <SeatChip style="right-[12%] top-[16%]" {...right} />
      <SeatChip style="left-1/2 -translate-x-1/2 bottom-[8%]" {...hero} highlight />

      {/* cartes du héros en bas */}
      {hero.cards?.length ? (
        <div className="absolute left-1/2 -translate-x-1/2 bottom-[17%] flex gap-2">
          {hero.cards.map((t,i)=><PlayingCard key={i} text={t} />)}
        </div>
      ) : null}

      {/* zone actions */}
      <div className="absolute bottom-4 left-0 right-0 px-6">
        {children}
      </div>
    </div>
  );
}

function SeatChip({
  style, label, stack, smallBlind, bigBlind, active=true, highlight=false,
}: {
  style: string; label: string; stack: string;
  smallBlind?: boolean; bigBlind?: boolean; active?: boolean; highlight?: boolean;
}) {
  return (
    <div className={cn("absolute", style)}>
              <div
          className={cn(
            "rounded-2xl px-3 py-1.5 shadow-lg backdrop-blur",
            "bg-card/80 ring-1 ring-border",
            highlight ? "outline outline-primary/60" : ""
          )}
        >
          <div className="flex items-center gap-2">
            <div
              className="h-6 w-6 rounded-full grid place-items-center text-[0.625rem] font-bold"
              style={{
                background: active ? "var(--seat-active-bg)" : "var(--seat-inactive-bg)",
                color: active ? "var(--seat-active-text)" : "var(--seat-inactive-text)"
              }}
            >
              {label.slice(0,1)}
            </div>
            <div className="leading-[1.05]">
              <div className="text-[0.6875rem] text-foreground/80 font-medium">{label}</div>
              <div className="text-[0.6875rem] text-primary/90">{stack}</div>
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
    <div
      className={cn(
        "ml-1 rounded-md px-1.5 py-0.5 text-[0.625rem] font-bold",
        text === "SB" ? "bg-destructive text-destructive-foreground" : "bg-primary text-primary-foreground"
      )}
    >
      {text}
    </div>
  );
}

/* ===== cartes style casino ===== */
function PlayingCard({ text }: { text: string}) {
  const r = text.slice(0, -1);
  const s = text.slice(-1);
  const isRed = s==="♥" || s==="♦";
  return (
    <div
      className={cn(
        "relative rounded-sm bg-card border border-border shadow-lg",
        "w-12 h-16"
      )}
    >
      <div
        className={cn(
          "absolute top-0.5 left-1 text-[0.625rem] font-bold",
          isRed ? "text-red-600" : "text-foreground"
        )}
      >
        {r}
      </div>
      <div
        className={cn(
          "absolute right-1 bottom-0.5 rotate-180 text-[0.625rem] font-bold",
          isRed ? "text-red-600" : "text-foreground"
        )}
      >
        {r}
      </div>
      <div
        className={cn(
          "absolute inset-0 grid place-items-center",
          isRed ? "text-red-600" : "text-foreground"
        )}
      >
        <span className="text-xl">{s}</span>
      </div>
    </div>
  );
}
