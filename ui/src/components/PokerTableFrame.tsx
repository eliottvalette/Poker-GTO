// ui/src/components/PokerTableFrame.tsx
"use client";
import React from "react";
import SeatInspection from "./SeatInspection";
import styles from "./PokerTableFrame.module.css";

type Seat = {
  id: number;            // Persistent player identity
  label: string;         // "SB" | "BB" | "BTN"
  stack: string;         // "27.0 BB"
  streetBet: number | null;
  smallBlind?: boolean;
  bigBlind?: boolean;
  active?: boolean;
  folded?: boolean;
  cards?: string[];      // ["A♥","8♣","10♠","XX"]
  netStackChange?: number;
  inspection?: React.ReactNode;
};

export default function PokerTableFrame({
  seats,
  potLabel,
  heroSeat,
  board,
  phase,
}: {
  seats: Seat[];
  potLabel: string;
  heroSeat: number;
  board: string;
  className?: string;
  phase?: string;
}) {
  const headsUp = seats.length === 2;
  const opponent = headsUp ? seats.find(s => s.id !== heroSeat)! : null;
  const order = [heroSeat, (heroSeat + 1) % 3, (heroSeat + 2) % 3];
  const hero = seats.find(s => s.id === order[0])!;
  const left = seats.find(s => s.id === order[1])!;
  const right = seats.find(s => s.id === order[2])!;

  return (
    <div
      aria-label="Poker table"
      className={`${styles.table} relative mx-auto aspect-[16/9] w-[98%] rounded-3xl px-4 pt-2
              bg-[radial-gradient(ellipse_at_center,_#0b3866_0%,_#053056_50%,_#061c34_100%)] 
              shadow-inner ring-1 ring-border overflow-hidden`}
      >
      {/* Stadium rail */}
      <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-[64%] w-[92%] h-[74%] rounded-full border border-primary/30 shadow-[inset_0_0_2rem_rgba(34,211,238,.15)]" />

      {/* Logo and pot */}
      {!headsUp && <div className="absolute top-8 left-1/2 -translate-x-1/2 text-muted-foreground/50 tracking-widest text-sm select-none">
        EXPRESSO
      </div>}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[36%] text-center">
        <div className="text-muted-foreground/80 text-xs">Pot</div>
        <div className="mt-0.5 text-foreground text-lg font-semibold drop-shadow">{potLabel}</div>
      </div>

      {/* Board */}
      <div aria-label="Board" className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[125%] flex gap-1">
        {board
          ? board.split(" ").map((t, i) => <PlayingCard key={i} text={t} />)
          : null}
      </div>

      {/* Seats and hole cards */}
      {headsUp && opponent ? <div className="absolute left-1/2 top-8 -translate-x-1/2 flex flex-col items-center">
        <InspectableSeat seat={opponent} />
        <StreetBet seat={opponent} className={styles.opponentBet} />
        {opponent.cards?.length ? <div className="mt-2 flex gap-1">
          {opponent.cards.map((t,i) => <PlayingCard key={i} text={t} active={opponent.active} folded={opponent.folded} phase={phase} />)}
        </div> : null}
      </div> : <>
      <div className={`${styles.leftSeat} absolute left-[8%] top-[12%] flex flex-col items-center`}>
        <InspectableSeat seat={left} />
        <StreetBet seat={left} className={styles.leftBet} />
        {left.cards?.length ? (
          <div className="mt-2 flex gap-1">
            {left.cards.map((t, i) => <PlayingCard key={i} text={t} active={left.active} folded={left.folded} phase={phase} />)}
          </div>
        ) : null}
      </div>

      <div className={`${styles.rightSeat} absolute right-[8%] top-[12%] flex flex-col items-center`}>
        <InspectableSeat seat={right} />
        <StreetBet seat={right} className={styles.rightBet} />
        {right.cards?.length ? (
          <div className="mt-2 flex gap-1">
            {right.cards.map((t, i) => <PlayingCard key={i} text={t} active={right.active} folded={right.folded} phase={phase} />)}
          </div>
        ) : null}
      </div>

      </>}

      <div className="absolute left-1/2 -translate-x-1/2 bottom-[14%] flex flex-col items-center">
        <SeatChip {...hero} />
        <StreetBet seat={hero} className={styles.heroBet} />
        {hero.cards?.length ? (
          <div className="mt-2 flex gap-2">
            {hero.cards.map((t, i) => <PlayingCard key={i} text={t} active={hero.active} folded={hero.folded} phase={phase} />)}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function InspectableSeat({seat}:{seat:Seat}) {
  return seat.inspection ? <SeatInspection seat={seat.id} content={seat.inspection}><SeatChip {...seat}/></SeatInspection> : <SeatChip {...seat}/>;
}

function StreetBet({ seat, className }: { seat: Seat; className: string }) {
  if (seat.netStackChange !== undefined) {
    const net = seat.netStackChange;
    if (!Number.isFinite(net)) throw new Error(`Invalid settled result for player ${seat.id}: ${net}`);
    const amount = `${net > 0 ? "+" : net < 0 ? "−" : ""}${Math.abs(net).toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
    const tone = net > 0 ? styles.resultWin : net < 0 ? styles.resultLoss : styles.resultEven;
    return <div className={`${styles.streetBet} ${className} ${tone}`} role="status"
      aria-label={`Player ${seat.id} hand result ${amount} BB`} data-player-result={seat.id}>
      <span className={styles.chip} aria-hidden="true" />
      <span>{amount} BB</span>
    </div>;
  }
  if (seat.streetBet === null || seat.streetBet === 0) return null;
  if (!Number.isFinite(seat.streetBet) || seat.streetBet < 0) {
    throw new Error(`Invalid current-street bet for player ${seat.id}: ${seat.streetBet}`);
  }
  const amount = seat.streetBet.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return (
    <div className={`${styles.streetBet} ${className}`} role="status"
      aria-label={`Player ${seat.id} bet ${amount} BB this round`} data-player-bet={seat.id}>
      <span className={styles.chip} aria-hidden="true" />
      <span>{amount} BB</span>
    </div>
  );
}

function SeatChip({
  label,
  stack,
  active = true,
}: {
  label: string;
  stack: string;
  netStackChange?: number;
  active?: boolean;
}) {
  return (
    <div>
      <div
        className={`${styles.seatChip} rounded-2xl px-3 py-1.5 shadow-lg backdrop-blur border border-border w-40 h-10 flex items-center bg-card/80`}
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


/* Casino card styling */
function PlayingCard({ text, active, folded = false, phase }: { text: string, active?: boolean, folded?: boolean, phase?: string }) {
  const faceDown = folded || (active === false && phase !== "SHOWDOWN");
  if (text === "XX" || faceDown) {
    return (
      <div aria-label={faceDown ? "Folded card, face down" : "Face-down card"}
        className={`${styles.playingCard} w-16 h-23 rounded-sm border border-neutral-700 shadow-lg bg-neutral-950 grid place-items-center ${faceDown ? "opacity-40 grayscale" : ""}`}>
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
    <div className={`${styles.playingCard} relative w-16 h-23 rounded-sm border shadow-lg bg-[var(--card-bg)] border-[var(--card-border)]`}>
      <div
        className={
          `${styles.cardCorner} absolute top-0.5 left-1 text-md font-bold ` +
          (isRed ? "text-[var(--card-red)]" : "text-[var(--card-black)]")
        }
      >
        {r}
        <div className="-mt-0.5 leading-none">{s}</div>
      </div>
      <div
        className={
          `${styles.cardCorner} absolute right-1 bottom-0.5 rotate-180 text-md font-bold ` +
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
        <span className={`${styles.cardSymbol} text-2xl`}>{s}</span>
      </div>
    </div>
  );
}
