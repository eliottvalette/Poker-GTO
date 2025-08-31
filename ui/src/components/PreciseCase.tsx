// ui/src/components/PreciseCase.tsx
"use client";
import React, { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ROLES, PHASES, normalize, type Policy } from "@/lib/policy";
import { unpackInfosetKeyDense } from "@/lib/infoset";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

const CARD_LABS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"] as const;

// Generate hand labels for the 169 grid
const HAND_LABELS: string[] = [];
for (let i = 0; i < 13; i++) {
  for (let j = 0; j < 13; j++) {
    if (i === j) {
      HAND_LABELS.push(`${CARD_LABS[i]}${CARD_LABS[j]}`);
    } else if (i < j) {
      HAND_LABELS.push(`${CARD_LABS[i]}${CARD_LABS[j]}s`);
    } else {
      HAND_LABELS.push(`${CARD_LABS[j]}${CARD_LABS[i]}o`);
    }
  }
}

// Board bucket labels
const BOARD_BUCKET_LABELS = [
  "PF", // Preflop
  "RB_NP_LO", "RB_NP_MID", "RB_NP_HI",
  "RB_PR_LO", "RB_PR_MID", "RB_PR_HI",
  "TT_NP_LO", "TT_NP_MID", "TT_NP_HI",
  "TT_PR_LO", "TT_PR_MID", "TT_PR_HI",
  "MONO_NP_LO", "MONO_NP_MID", "MONO_NP_HI",
  "MONO_PR_LO", "MONO_PR_MID", "MONO_PR_HI"
];

// Hero vs Board bucket labels
const HEROBOARD_LABELS = [
  "AIR", "DRAW", "PAIR", "STRONG_PAIR", "OVERPAIR", "STRONG_PAIR_DRAW", "COMBO_DRAW", "MADE_STRAIGHT_FLUSH"
];

type PreciseCaseProps = {
  policy: Policy | null;
};

export default function PreciseCase({ policy }: PreciseCaseProps) {
  const [selectedPhase, setSelectedPhase] = useState<number>(0);
  const [selectedRole, setSelectedRole] = useState<number>(0);
  const [selectedHand, setSelectedHand] = useState<number>(0);
  const [selectedBoard, setSelectedBoard] = useState<number>(0);
  const [selectedPot, setSelectedPot] = useState<number>(0);
  const [selectedRatio, setSelectedRatio] = useState<number>(0);
  const [selectedSpr, setSelectedSpr] = useState<number>(0);
  const [selectedHeroBoard, setSelectedHeroBoard] = useState<number>(0);

  const selectedCase = useMemo(() => {
    if (!policy) return null;

    // Find matching infoset
    for (const [kStr, entry] of Object.entries(policy)) {
      const unpacked = unpackInfosetKeyDense(kStr);
      if (
        unpacked.phase === selectedPhase &&
        unpacked.role === selectedRole &&
        unpacked.hand === selectedHand &&
        unpacked.board === selectedBoard &&
        unpacked.pot === selectedPot &&
        unpacked.ratio === selectedRatio &&
        unpacked.spr === selectedSpr &&
        unpacked.heroboard === selectedHeroBoard
      ) {
        return { key: kStr, unpacked, entry };
      }
    }
    return null;
  }, [policy, selectedPhase, selectedRole, selectedHand, selectedBoard, selectedPot, selectedRatio, selectedSpr, selectedHeroBoard]);

  const actionDistribution = useMemo(() => {
    if (!selectedCase) return null;
    
    const normalized = normalize(selectedCase.entry.dist);
    return Object.entries(normalized)
      .filter(([_, prob]) => prob > 0)
      .sort(([_, a], [__, b]) => b - a);
  }, [selectedCase]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cas Précis (ne marche pas encore)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
            <Label htmlFor="board">Board</Label>
            <Select value={String(selectedBoard)} onValueChange={(v) => setSelectedBoard(parseInt(v))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {BOARD_BUCKET_LABELS.map((board, i) => (
                  <SelectItem key={board} value={String(i)}>{board}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

        <div className="grid grid-cols-2 gap-8 gap-y-8 justify-center">
          <div className="space-y-2">
            <Label htmlFor="pot">Pot Bucket</Label>
            <Input
              type="number"
              value={selectedPot}
              onChange={(e) => setSelectedPot(parseInt(e.target.value) || 0)}
              min={0}
              max={23}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="ratio">Ratio Bucket</Label>
            <Input
              type="number"
              value={selectedRatio}
              onChange={(e) => setSelectedRatio(parseInt(e.target.value) || 0)}
              min={0}
              max={7}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="spr" className="mb-4">SPR Bucket</Label>
            <Input
              type="number"
              value={selectedSpr}
              onChange={(e) => setSelectedSpr(parseInt(e.target.value) || 0)}
              min={0}
              max={6}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="heroboard">Hero vs Board</Label>
            <Select value={String(selectedHeroBoard)} onValueChange={(v) => setSelectedHeroBoard(parseInt(v))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {HEROBOARD_LABELS.map((hb, i) => (
                  <SelectItem key={hb} value={String(i)}>{hb}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {selectedCase ? (
          <div className="space-y-3">
            <div className="text-sm text-muted-foreground">
              <strong>Found case:</strong> {selectedCase.entry.visits} visits
            </div>
            
            {actionDistribution && actionDistribution.length > 0 ? (
              <div className="space-y-2">
                <div className="text-sm font-medium">Action Distribution:</div>
                <div className="space-y-1">
                  {actionDistribution.map(([action, prob]) => (
                    <div key={action} className="flex justify-between text-sm">
                      <span>{action}</span>
                      <span className="font-mono">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">No actions found</div>
            )}
          </div>
        ) : (
          <div className="text-sm text-muted-foreground">
            Aucune correspondance trouvée
          </div>
        )}
      </CardContent>
    </Card>
  );
}
