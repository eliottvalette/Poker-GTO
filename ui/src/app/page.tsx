"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ROLES, PHASES, ACTIONS, normalize, type GridMix } from "@/lib/policy";
import { unpackInfosetKeyDense } from "@/lib/infoset";
import Grid169 from "@/components/Grid169";
import Legend from "@/components/Legend";

type Policy = Record<string, Record<string, number>>;

export default function Page() {
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [phaseIdx, setPhaseIdx] = useState<number>(0);
  const [roleIdx, setRoleIdx] = useState<number>(0);
  const [labelThreshold, setLabelThreshold] = useState<number>(18);

  useEffect(() => {
    fetch("/avg_policy.json").then(r => r.json()).then(setPolicy).catch(console.error);
  }, []);

  const gridMixes: GridMix[] = useMemo(() => {
    const sums: GridMix[] = Array.from({length:169}, () =>
      Object.fromEntries(ACTIONS.map(a=>[a,0])) as GridMix
    );
    const counts = Array(169).fill(0);

    if (policy) {
      for (const [kStr, dist] of Object.entries(policy)) {
        const f = unpackInfosetKeyDense(kStr);
        if (f.phase !== phaseIdx || f.role !== roleIdx) continue;
        const probs = normalize(dist);
        for (const a of ACTIONS) sums[f.hand][a] += probs[a];
        counts[f.hand] += 1;
      }
    }

    return sums.map((tot, h) => {
      const c = counts[h] || 0;
      if (!c) return Object.fromEntries(ACTIONS.map(a=>[a,0])) as GridMix;
      const avg = Object.fromEntries(ACTIONS.map(a=>[a, tot[a]/c])) as GridMix;
      const s = ACTIONS.reduce((acc,a)=>acc+(avg[a]||0),0);
      return s>0 ? Object.fromEntries(ACTIONS.map(a=>[a,(avg[a]||0)/s])) as GridMix : avg;
    });
  }, [policy, phaseIdx, roleIdx]);

  return (
    <main className="p-4 md:p-6 space-y-4">
      <div className="flex items-end justify-between gap-3 flex-wrap">
        <div>
          <h1 className="text-xl md:text-2xl font-semibold">GTO Viewer</h1>
          <p className="text-sm text-zinc-400">Grille 13×13 — chaque case affiche une rangée (row) proportionnelle au mix d’actions.</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="text-sm text-zinc-300">Label ≥ %</div>
          <Input type="number" className="w-20" value={labelThreshold}
                 min={0} max={100}
                 onChange={(e)=>setLabelThreshold(parseInt(e.target.value || "0",10))}/>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2"><CardTitle>Phase & Position</CardTitle></CardHeader>
        <CardContent className="flex gap-4 flex-wrap">
          <Tabs value={String(phaseIdx)} onValueChange={(v)=>setPhaseIdx(parseInt(v,10))}>
            <TabsList className="flex flex-wrap">
              {PHASES.slice(0,4).map((ph, i)=>(<TabsTrigger key={ph} value={String(i)}>{ph}</TabsTrigger>))}
            </TabsList>
          </Tabs>
          <Tabs value={String(roleIdx)} onValueChange={(v)=>setRoleIdx(parseInt(v,10))}>
            <TabsList className="flex flex-wrap">
              {ROLES.map((r, i)=>(<TabsTrigger key={r} value={String(i)}>{r}</TabsTrigger>))}
            </TabsList>
          </Tabs>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2"><CardTitle>Mix d’actions (13×13)</CardTitle></CardHeader>
        <CardContent>
          {!policy ? <div className="text-zinc-400">Chargement de <code>avg_policy.json</code>…</div> : (
            <>
              <Legend />
              <div className="mt-3">
                <Grid169 gridMixes={gridMixes} labelThresholdPct={labelThreshold}/>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </main>
  );
}
