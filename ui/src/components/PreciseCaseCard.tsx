"use client";
import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { normalize, type Policy } from "@/lib/policy";
import { unpackInfosetKeyDense } from "@/lib/infoset";
import { qlogPotBucket, ratioBucket, sprBucket } from "@/lib/buckets";

const CARD_LABS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"] as const;
// 169 labels
const HAND_LABELS: string[] = [];
for (let i=0;i<13;i++) for (let j=0;j<13;j++)
  HAND_LABELS.push(i===j? `${CARD_LABS[i]}${CARD_LABS[j]}` : (i<j? `${CARD_LABS[i]}${CARD_LABS[j]}s` : `${CARD_LABS[j]}${CARD_LABS[i]}o`));

// board & hero buckets (mêmes listes que ton composant actuel)
const BOARD_BUCKET_LABELS = [
  "PF","RB_NP_LO","RB_NP_MID","RB_NP_HI","RB_PR_LO","RB_PR_MID","RB_PR_HI",
  "TT_NP_LO","TT_NP_MID","TT_NP_HI","TT_PR_LO","TT_PR_MID","TT_PR_HI",
  "MONO_NP_LO","MONO_NP_MID","MONO_NP_HI","MONO_PR_LO","MONO_PR_MID","MONO_PR_HI"
];
const HEROBOARD_LABELS = ["AIR","DRAW","PAIR","STRONG_PAIR","OVERPAIR","STRONG_PAIR_DRAW","COMBO_DRAW","MADE_STRAIGHT_FLUSH"];

export default function PreciseCaseCard({
  policy, phaseIdx, roleIdx
}: { policy: Policy|null; phaseIdx:number; roleIdx:number }) {
  const [handIdx, setHandIdx] = useState(0);
  const [boardIdx, setBoardIdx] = useState(0);
  const [heroBoardIdx, setHeroBoardIdx] = useState(0);

  // valeurs en BB, pas des buckets
  const [potBB, setPotBB] = useState<number>(2);       // ex: après blinds: 3, etc.
  const [toCallBB, setToCallBB] = useState<number>(2); // mise à suivre
  const [effStackBB, setEffStackBB] = useState<number>(100);

  // map -> buckets
  const potQ = qlogPotBucket(potBB);
  const ratioQ = ratioBucket(toCallBB, potBB);
  const sprQ = sprBucket(effStackBB, potBB);

  const selected = useMemo(() => {
    if (!policy) return null;
    for (const [kStr, entry] of Object.entries(policy)) {
      const f = unpackInfosetKeyDense(kStr);
      if (f.phase===phaseIdx && f.role===roleIdx && f.hand===handIdx &&
          f.board===boardIdx && f.pot===potQ && f.ratio===ratioQ &&
          f.spr===sprQ && f.heroboard===heroBoardIdx) {
        return { kStr, f, entry };
      }
    }
    return null;
  }, [policy, phaseIdx, roleIdx, handIdx, boardIdx, potQ, ratioQ, sprQ, heroBoardIdx]);

  const actionDist = useMemo(() => {
    if (!selected) return null;
    const n = normalize(selected.entry.dist);
    return Object.entries(n).filter(([,p])=>p>0).sort((a,b)=>b[1]-a[1]);
  }, [selected]);

  return (
    <Card>
      <CardHeader><CardTitle>Cas précis (en BB)</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label>Main (169)</Label>
            <Select value={String(handIdx)} onValueChange={(v)=>setHandIdx(parseInt(v))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {HAND_LABELS.map((h,i)=>(<SelectItem key={h} value={String(i)}>{h}</SelectItem>))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Board</Label>
            <Select value={String(boardIdx)} onValueChange={(v)=>setBoardIdx(parseInt(v))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {BOARD_BUCKET_LABELS.map((b,i)=>(<SelectItem key={b} value={String(i)}>{b}</SelectItem>))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Hero vs Board</Label>
            <Select value={String(heroBoardIdx)} onValueChange={(v)=>setHeroBoardIdx(parseInt(v))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {HEROBOARD_LABELS.map((h,i)=>(<SelectItem key={h} value={String(i)}>{h}</SelectItem>))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label>Pot (BB)</Label>
            <Input type="number" min={0} step="0.5" value={potBB}
              onChange={(e)=>setPotBB(Number(e.target.value)||0)} />
            <div className="text-xs text-muted-foreground">→ bucket {potQ}</div>
          </div>
          <div className="space-y-2">
            <Label>À suivre (BB)</Label>
            <Input type="number" min={0} step="0.5" value={toCallBB}
              onChange={(e)=>setToCallBB(Number(e.target.value)||0)} />
            <div className="text-xs text-muted-foreground">→ ratio bucket {ratioQ}</div>
          </div>
          <div className="space-y-2">
            <Label>Eff. stack (BB)</Label>
            <Input type="number" min={0} step="0.5" value={effStackBB}
              onChange={(e)=>setEffStackBB(Number(e.target.value)||0)} />
            <div className="text-xs text-muted-foreground">→ SPR bucket {sprQ}</div>
          </div>
        </div>

        {selected ? (
          <div className="space-y-2">
            <div className="text-sm text-muted-foreground">
              <b>Infoset trouvé</b> • {selected.entry.visits.toLocaleString()} visites
            </div>
            {actionDist?.map(([a,p])=>(
              <div key={a} className="flex justify-between text-sm">
                <span>{a}</span><span className="font-mono">{(p*100).toFixed(1)}%</span>
              </div>
            )) || <div className="text-sm text-muted-foreground">Aucune action</div>}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground text-center">Aucune correspondance</div>
        )}
      </CardContent>
    </Card>
  );
}
