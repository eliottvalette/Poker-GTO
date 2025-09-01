"use client";
import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { normalize, type Policy } from "@/lib/policy";
import {
  unpackInfosetKeyDense,
  prettyInfosetRow,
  BOARD_BUCKET_LABELS,
  HEROBOARD_LABELS,
} from "@/lib/infoset";

// petits helpers pour « Toutes » = null
const asNullableNumber = (v: string) => (v === "ALL" ? null : Number(v));

type Props = { policy: Policy | null };

export default function PreciseCaseCard({ policy }: Props) {
  // Phase / Position (null => Toutes)
  const [phaseIdx, setPhaseIdx] = useState<number | null>(null);
  const [roleIdx, setRoleIdx] = useState<number | null>(null);

  // filtres (null => « Toutes »)
  const [handIdx, setHandIdx] = useState<number | null>(null);
  const [boardBkt, setBoardBkt] = useState<number | null>(null);
  const [potBB, setPotBB] = useState<string>("ALL");
  const [toCallBB, setToCallBB] = useState<string>("ALL");
  const [sprBB, setSprBB] = useState<string>("ALL");
  const [heroBoardBkt, setHeroBoardBkt] = useState<number | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  // NOTE: ici on ne recalcule pas les buckets à partir de BB (tu as déjà dit que l'input reste en BB mais le filtre prend « Toutes » si vide)
  // Si tu veux convertir BB->bucket, branche ici ta logique.

  const { countMatches, matches } = useMemo(() => {
    if (!policy) return { countMatches: 0, matches: [] as {kStr:string, entry: Policy[keyof Policy]}[] };

    const potBucket = asNullableNumber(potBB);
    const ratioBucket = asNullableNumber(toCallBB);
    const sprBucket = asNullableNumber(sprBB);

    const rows: {kStr:string, entry: Policy[keyof Policy]}[] = [];
    for (const [kStr, entry] of Object.entries(policy)) {
      const f = unpackInfosetKeyDense(kStr);
      if (phaseIdx !== null && f.phase !== phaseIdx) continue;
      if (roleIdx !== null && f.role !== roleIdx) continue;
      if (handIdx !== null && f.hand !== handIdx) continue;
      if (boardBkt !== null && f.board !== boardBkt) continue;
      if (heroBoardBkt !== null && f.heroboard !== heroBoardBkt) continue;
      if (potBucket !== null && f.pot !== potBucket) continue;
      if (ratioBucket !== null && f.ratio !== ratioBucket) continue;
      if (sprBucket !== null && f.spr !== sprBucket) continue;
      rows.push({ kStr, entry });
    }
    rows.sort((a,b)=> b.entry.visits - a.entry.visits);
    return { countMatches: rows.length, matches: rows }; // plus de slice ici
  }, [policy, phaseIdx, roleIdx, handIdx, boardBkt, heroBoardBkt, potBB, toCallBB, sprBB]);

  const countInfosets = countMatches;
  const totalVisits = matches.reduce((s, m) => s + m.entry.visits, 0);

  const selectedRow = useMemo(() => {
    if (!selectedKey) return null;
    const found = matches.find(m => m.kStr === selectedKey);
    if (!found) return null;
    const norm = normalize(found.entry.dist);
    const sorted = Object.entries(norm)
      .filter(([,p])=>p>0)
      .sort(([,a],[,b])=>b-a);
    return { ...found, actions: sorted };
  }, [matches, selectedKey]);

  // Hand options (169)
  const HAND_LABELS: string[] = useMemo(() => {
    const labs = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"];
    const out: string[] = [];
    for (let i=0;i<13;i++){
      for (let j=0;j<13;j++){
        if (i===j) out.push(`${labs[i]}${labs[j]}`);
        else if (i<j) out.push(`${labs[i]}${labs[j]}s`);
        else out.push(`${labs[j]}${labs[i]}o`);
      }
    }
    return out;
  }, []);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle>Cas précis (en BB)</CardTitle>
        <CardDescription>
          {countInfosets.toLocaleString()} infosets trouvés • {totalVisits.toLocaleString()} visites
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Filtres */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Phase</Label>
            <Select
              value={phaseIdx === null ? "ALL" : String(phaseIdx)}
              onValueChange={(v)=> setPhaseIdx(v==="ALL" ? null : parseInt(v,10))}
            >
              <SelectTrigger><SelectValue placeholder="Toutes" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">Toutes</SelectItem>
                {/* 0..3 = PREFLOP..RIVER */}
                {["PREFLOP","FLOP","TURN","RIVER"].map((p,i)=>(
                  <SelectItem key={p} value={String(i)}>{p}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Position</Label>
            <Select
              value={roleIdx === null ? "ALL" : String(roleIdx)}
              onValueChange={(v)=> setRoleIdx(v==="ALL" ? null : parseInt(v,10))}
            >
              <SelectTrigger><SelectValue placeholder="Toutes" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">Toutes</SelectItem>
                {["SB","BB","BTN"].map((r,i)=>(
                  <SelectItem key={r} value={String(i)}>{r}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Main (169)</Label>
            <Select value={handIdx === null ? "ALL" : String(handIdx)}
                    onValueChange={(v)=> setHandIdx(v==="ALL" ? null : parseInt(v,10))}>
              <SelectTrigger><SelectValue placeholder="Toutes" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">Toutes</SelectItem>
                {HAND_LABELS.map((h, i)=>(
                  <SelectItem key={i} value={String(i)}>{h}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Board</Label>
            <Select value={boardBkt === null ? "ALL" : String(boardBkt)}
                    onValueChange={(v)=> setBoardBkt(v==="ALL" ? null : parseInt(v,10))}>
              <SelectTrigger><SelectValue placeholder="Toutes" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">Toutes</SelectItem>
                {BOARD_BUCKET_LABELS.map((b,i)=>(
                  <SelectItem key={b} value={String(i)}>{b}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Hero vs Board</Label>
            <Select value={heroBoardBkt === null ? "ALL" : String(heroBoardBkt)}
                    onValueChange={(v)=> setHeroBoardBkt(v==="ALL" ? null : parseInt(v,10))}>
              <SelectTrigger><SelectValue placeholder="Toutes" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">Toutes</SelectItem>
                {HEROBOARD_LABELS.map((hb,i)=>(
                  <SelectItem key={hb} value={String(i)}>{hb}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Pot (BB)</Label>
            <Select value={potBB} onValueChange={setPotBB}>
              <SelectTrigger><SelectValue placeholder="Toutes" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">Toutes</SelectItem>
                {/* si tu veux, ajoute des valeurs « 2 → bucket 2 », etc. */}
                {/* ou laisse ALL tant qu'on n'a pas la conversion BB->bucket */}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>À suivre (BB)</Label>
            <Select value={toCallBB} onValueChange={setToCallBB}>
              <SelectTrigger><SelectValue placeholder="Toutes" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">Toutes</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Eff. stack (BB)</Label>
            <Select value={sprBB} onValueChange={setSprBB}>
              <SelectTrigger><SelectValue placeholder="Toutes" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">Toutes</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Tableau des matchs */}
        <div className="rounded-md border">
          <div className="text-xs text-muted-foreground text-center p-2">
            {(() => {
              const totalInfosets = policy ? Object.keys(policy).length : 0;
              const pct = totalInfosets > 0 ? (countInfosets / totalInfosets) * 100 : 0;
              return <>Vous matchez {pct.toFixed(1)}% des infosets — {countInfosets} / {totalInfosets}</>;
            })()}
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[120px]">Phase</TableHead>
                <TableHead className="w-[80px]">Pos</TableHead>
                <TableHead className="w-[90px]">Main</TableHead>
                <TableHead className="min-w-[110px]">Board</TableHead>
                <TableHead className="min-w-[130px]">Hero vs Board</TableHead>
                <TableHead>Pot</TableHead>
                <TableHead>Ratio</TableHead>
                <TableHead>SPR</TableHead>
                <TableHead className="text-right">Visits</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {matches.slice(0, 5).map(({ kStr, entry }) => {
                const row = prettyInfosetRow(kStr);
                const isSelected = selectedKey === kStr;
                return (
                  <TableRow
                    key={kStr}
                    className={isSelected ? "bg-accent/50" : "cursor-pointer"}
                    onClick={()=> setSelectedKey(kStr)}
                  >
                    <TableCell>{row.phase}</TableCell>
                    <TableCell>{row.role}</TableCell>
                    <TableCell>{row.hand}</TableCell>
                    <TableCell>{row.board}</TableCell>
                    <TableCell>{row.heroboard}</TableCell>
                    <TableCell>{row.pot}</TableCell>
                    <TableCell>{row.ratio}</TableCell>
                    <TableCell>{row.spr}</TableCell>
                    <TableCell className="text-right font-mono">{entry.visits.toLocaleString()}</TableCell>
                  </TableRow>
                );
              })}
              {matches.length === 0 && (
                <TableRow>
                  <TableCell colSpan={9} className="text-center text-sm text-muted-foreground py-8">
                    Aucune correspondance.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>

        {/* Détail de l'infoset sélectionné */}
        {selectedRow ? (
          <div className="space-y-2">
            <div className="text-sm text-muted-foreground">
              <strong>Infoset sélectionné :</strong> {selectedRow.entry.visits.toLocaleString()} visites
            </div>
            <div className="space-y-1">
              {selectedRow.actions.map(([action, prob])=>(
                <div key={action} className="flex justify-between text-sm">
                  <span>{action}</span>
                  <span className="font-mono">{(prob*100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground text-center">
                         Sélectionne une ligne pour voir la distribution d&apos;actions.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
