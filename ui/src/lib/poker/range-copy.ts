import { handClassMasses } from "./beliefs";
import type { HandState } from "./engine";
import type { Ranges } from "./hybrid";
import { cardLabel } from "../game";

/** Compact public context and all 169 class masses; never serialize dealt private cards. */
export function formatRangeCopy(seat: number, hand: Pick<HandState, "board" | "history">,
                                rows: Ranges[number], profile: string): string {
  const masses = handClassMasses(rows), ranks = "AKQJT98765432";
  const classes = [...ranks].flatMap((a, i) => [...ranks].map((b, j) => {
    const label = i === j ? a+b : i < j ? a+b+"s" : b+a+"o";
    return `${label}:${Number(((masses[label] ?? 0)*100).toFixed(4))}%`;
  }));
  let street = "";
  const history = hand.history.map(event => {
    const prefix = event.street !== street ? `${event.street} ` : "";
    street = event.street;
    const amount = ["BLIND", "CALL", "RAISE"].includes(event.action) ? ` to=${event.amount_to}` : "";
    return `${prefix}P${event.player_id}(${event.position}) ${event.action}${amount}`;
  }).join("; ");
  return [
    `P${seat} estimated range | profile=${profile}`,
    `Board: ${hand.board.map(cardLabel).join(" ") || "—"}`,
    `History (chip amounts, to=street total): ${history || "—"}`,
    `Range (posterior class probabilities): ${classes.join(" ")}`,
  ].join("\n");
}
