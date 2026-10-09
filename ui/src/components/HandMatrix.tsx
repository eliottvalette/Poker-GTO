import type { ReactNode } from "react";

const RANKS = "AKQJT98765432";
export const HAND_CLASSES = [...RANKS].flatMap((a, i) => [...RANKS].map((b, j) => ({
  label: i === j ? a + b : i < j ? a + b + "s" : b + a + "o",
  row: i,
  column: j,
})));

export default function HandMatrix({ children }: { children: (cell: typeof HAND_CLASSES[number]) => ReactNode }) {
  return <div className="grid gap-px" style={{ gridTemplateColumns: "repeat(13,minmax(0,1fr))" }} aria-label="169 hand classes">
    {HAND_CLASSES.map(children)}
  </div>;
}
