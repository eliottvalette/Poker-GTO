// ui/src/app/page.tsx
"use client";

import React, { useEffect, useMemo, useState } from "react";
import { inflate } from "pako";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ROLES, PHASES, ACTIONS, normalize, type GridMix, calculateWeightedStats, calculatePhaseStats } from "@/lib/policy";
import { unpackInfosetKeyDense } from "@/lib/infoset";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import Grid169 from "@/components/Grid169";
import Legend from "@/components/Legend";
import PreciseCaseCard from "@/components/PreciseCaseCard";
import TestTable from "@/components/TestTable";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarProvider,
  SidebarTrigger,
  SidebarInset,
} from "@/components/ui/sidebar";

type PolicyEntry = { dist: Record<string, number>, visits: number };
type Policy = Record<string, PolicyEntry>;

// Decode l'entrée compacte {policy:[mask,q...], visits:n} -> {action: proba}
function decodeCompact(entry: {policy:number[], visits:number}): Record<string, number> {
  const arr = entry.policy;
  const mask = arr?.[0] ?? 0;
  const qs = arr.slice(1);
  const total = qs.reduce((a,b)=>a+b,0);
  if (!mask || total <= 0) return {};
  const ACTIONS = ["FOLD","CHECK","CALL","RAISE","ALL-IN"] as const;
  const dist: Record<string, number> = {};
  let qi = 0;
  for (let i=0;i<ACTIONS.length;i++) {
    if ((mask >> i) & 1) {
      const q = qs[qi++] ?? 0;
      dist[ACTIONS[i]] = q / total;
    }
  }
  return dist;
}


export default function Page() {
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [phaseIdx, setPhaseIdx] = useState<number>(0);
  const [roleIdx, setRoleIdx] = useState<number>(0);
  const [heatmapMode, setHeatmapMode] = useState<"action" | "visits" | false>(false);
  const [detailedMode, setDetailedMode] = useState(false);
  const [mainTab, setMainTab] = useState<"overview"|"case"|"test">("overview");
  
  useEffect(() => {
    (async () => {
      const res = await fetch("/avg_policy.json.gz");
      const buf = await res.arrayBuffer();
      const jsonText = new TextDecoder("utf-8").decode(inflate(new Uint8Array(buf)));
      const raw: Record<string, {policy:number[], visits:number}> = JSON.parse(jsonText);
      const decoded: Policy = Object.fromEntries(
        Object.entries(raw).map(([k, v]) => [k, {
          dist: decodeCompact(v),
          visits: v.visits
        }])
      );
      
      setPolicy(decoded);
    })().catch(console.error);
  }, []);
  

  const { gridMixes, visitCounts }: { gridMixes: GridMix[]; visitCounts: number[] } = useMemo(() => {
    const sums: GridMix[] = Array.from({length:169}, () =>
      Object.fromEntries(ACTIONS.map(a=>[a,0])) as GridMix
    );
    const totalVisits = Array(169).fill(0);

    if (policy) {
      for (const [kStr, {dist, visits: visitCount}] of Object.entries(policy)) {
        const f = unpackInfosetKeyDense(kStr);
        if (f.phase !== phaseIdx || f.role !== roleIdx) continue;
        if (f.hand < 0 || f.hand >= 169) continue; // garde-fou
        const probs = normalize(dist);
        // Pondérer par le nombre de visites
        for (const a of ACTIONS) sums[f.hand][a] += probs[a] * visitCount;
        totalVisits[f.hand] += visitCount;
      }
    }

    const gridMixes = sums.map((tot, h) => {
      const totalVisit = totalVisits[h] || 0;
      if (totalVisit <= 0) return Object.fromEntries(ACTIONS.map(a=>[a,0])) as GridMix;
      const avg = Object.fromEntries(ACTIONS.map(a=>[a, tot[a]/totalVisit])) as GridMix;
      const s = ACTIONS.reduce((acc,a)=>acc+(avg[a]||0),0);
      return s>0 ? Object.fromEntries(ACTIONS.map(a=>[a,(avg[a]||0)/s])) as GridMix : avg;
    });

    return { gridMixes, visitCounts: totalVisits };
  }, [policy, phaseIdx, roleIdx]);

  const weightedStats = useMemo(() => calculateWeightedStats(visitCounts), [visitCounts]);
  const phaseStats = useMemo(() => calculatePhaseStats(policy), [policy]);

  return (
    <SidebarProvider>
      <Sidebar variant="floating" className="p-3">
        <SidebarHeader className="border-b border-border p-4">
          <div className="flex items-center gap-2">
            <SidebarTrigger />
            <h1 className="text-lg font-semibold">GTO Viewer</h1>
          </div>
        </SidebarHeader>
        <SidebarContent className="p-4 space-y-4">
          <Card className="bg-muted/30">
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Button 
                  className={heatmapMode === "action" ? "bg-primary text-primary-foreground w-full hover:bg-primary/90 cursor-pointer" : 
                            heatmapMode === "visits" ? "bg-orange-600 text-white w-full hover:bg-orange-700 cursor-pointer" :
                            "bg-secondary text-secondary-foreground w-full hover:bg-secondary/80 cursor-pointer"} 
                  onClick={() => setHeatmapMode(heatmapMode === false ? "action" : heatmapMode === "action" ? "visits" : false)}
                >
                  {heatmapMode === "action" ? "Action Heatmap" : 
                   heatmapMode === "visits" ? "Visits Heatmap" : 
                   "Heatmap Mode"}
                </Button>
              </div>

              <div className="space-y-2">
                <Label htmlFor="phase">Phase</Label>
                <Tabs value={String(phaseIdx)} onValueChange={(v) => setPhaseIdx(parseInt(v))}>
                  <TabsList className="grid w-full grid-rows-4 h-27">
                    {PHASES.slice(0, 4).map((phase, i)=>(<TabsTrigger key={phase} value={String(i)} className="cursor-pointer rounded-sm w-[100%]">{phase}</TabsTrigger>))}
                  </TabsList>
                </Tabs>
              </div>

              <div className="space-y-2">
                <div className="text-sm font-medium">Position</div>
                <Tabs value={String(roleIdx)} onValueChange={(v)=>setRoleIdx(parseInt(v,10))}>
                  <TabsList className="grid w-full grid-cols-3">
                    {ROLES.map((r, i)=>(<TabsTrigger key={r} value={String(i)} className="cursor-pointer">{r}</TabsTrigger>))}
                  </TabsList>
                </Tabs>
              </div>
            </CardContent>
          </Card>

          {/* Choix de page principale */}
          <Card className="bg-muted/30">
            <CardContent className="space-y-2">
              <div className="flex flex-col gap-2">
                <Button variant="outline" className={mainTab === "overview" ? "bg-primary text-primary-foreground w-full hover:bg-primary/90 hover:text-primary-foreground cursor-pointer" : "bg-secondary text-secondary-foreground w-full hover:bg-secondary/80 hover:text-secondary-foreground cursor-pointer"} onClick={()=>setMainTab("overview")}>Overview</Button>
                <Button variant="outline" className={mainTab === "case" ? "bg-primary text-primary-foreground w-full hover:bg-primary/90 hover:text-primary-foreground cursor-pointer" : "bg-secondary text-secondary-foreground w-full hover:bg-secondary/80 hover:text-secondary-foreground cursor-pointer"} onClick={()=>setMainTab("case")}>Cas précis</Button>
                <Button variant="outline" className={mainTab === "test" ? "bg-primary text-primary-foreground w-full hover:bg-primary/90 hover:text-primary-foreground cursor-pointer" : "bg-secondary text-secondary-foreground w-full hover:bg-secondary/80 hover:text-secondary-foreground cursor-pointer"} onClick={()=>setMainTab("test")}>Test Live</Button>
              </div>
            </CardContent>
          </Card>
        </SidebarContent>
      </Sidebar>

      <SidebarInset>
        <main className="py-3 px-4 space-y-4">
          {mainTab === "overview" && (
            <Card className="gap-1">
              <CardHeader className="pb-1">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Mix d&apos;actions (13x13)</CardTitle>
                    <CardDescription>
                        Statistiques pondérées par visites: {weightedStats.totalVisits.toLocaleString()} visites totales 
                        ({weightedStats.avgVisitsPerHand.toFixed(0)} visites/mains en moyenne)
                        <br />
                        <span className="text-sm">
                          {PHASES.slice(0, 4).map(phase => (
                            <span key={phase}>
                              {phase}: {phaseStats[phase]?.avgVisitsPerHand.toFixed(0) || '0'} visites/mains
                              {phase !== 'RIVER' ? ' | ' : ''}
                            </span>
                          ))}
                        </span>
                    </CardDescription>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <Button
                      variant={detailedMode ? "default" : "secondary"}
                      size="sm"
                      onClick={()=>setDetailedMode(v=>!v)}
                    >
                      {detailedMode ? "Detailed: ON" : "Detailed: OFF"}
                    </Button>
                    <div className="text-xs text-muted-foreground text-right">
                      (5 actions vs 3 groupes)
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {!policy ? <div className="text-muted-foreground">Chargement de <code>avg_policy.json.gz</code>…</div> : (
                  <>
                    <Legend heatmapMode={heatmapMode} detailed={detailedMode} />
                    <div className="mt-1">
                      <Grid169 
                        gridMixes={gridMixes} 
                        visitCounts={visitCounts}
                        heatmapMode={heatmapMode} 
                        detailed={detailedMode}
                      />
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {mainTab === "case" && (
            <PreciseCaseCard policy={policy} />
          )}

          {mainTab === "test" && (
            <TestTable policy={policy} />
          )}
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
