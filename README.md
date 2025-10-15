# GTO_Bot

Projet d'expérimentations autour d'un solver CFR+ (3-handed NLHE), d'un modèle ML qui approxime la policy, et d'une UI Next.js pour explorer les résultats.

## Aperçu
- CFR+ externe 3-handed minimal-raise dans `cfr_solver.py` s'appuie sur `poker_game_expresso.PokerGameExpresso` et les clés infosets compactes dans `infoset.py`.
- La policy moyenne est sérialisée en JSON gzippé compact (bitmask + valeurs quantifiées) dans `policy/avg_policy.json.gz` et dupliquée pour l'UI dans `ui/public/avg_policy.json.gz`.
- Un modèle PyTorch (`ml/model.py`) est entraîné à approximer cette policy avec `ml/train.py`. Des visualisations (ex: heatmap préflop) sont dans `ml/viz.py`.
- L'UI (`ui/`) est une app Next.js/TypeScript qui charge la policy depuis `public/avg_policy.json.gz`.


## Installation Python (core + ML)
```bash
# Créer un venv et installer les dépendances (exemple)
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas treys tqdm seaborn matplotlib
```

## Lancer l'entraînement CFR+
`cfr_solver.py` entraîne le solver, sauvegarde la policy et exporte une version utilisable par l'UI.

```bash
python cfr_solver.py
```
Sorties principales:
- `policy/avg_policy.json.gz`
- `ui/public/avg_policy.json.gz`
- `policy/avg_policy.csv` (via `stats_policy.extraction_policy_data()`)

Paramètres clés (éditer dans `cfr_solver.py`):
- `iterations` (défaut 1_000_000)
- `stacks` (ex: `(100, 100, 100)`)
- `SAVE_EVERY` pour checkpoints (0 = désactivé)

## Analyse et export CSV
```bash
python stats_policy.py  # lit policy/avg_policy.json.gz et écrit policy/avg_policy.csv
```

`stats_policy.py` reconstruit les distributions d'actions, décode les clés infosets et produit un CSV rapide à explorer.

## Entraîner le modèle ML sur la policy
```bash
cd ml
python train.py  # lit ../policy/avg_policy.json.gz, entraîne et sauvegarde trained_policy_model.pth
```
Points notables:
- Entrée: one-hot 224 dims (phase, rôle, main 169, board 31, 3 scalaires normalisés, hero-vs-board 11)
- Sortie: 5 actions canon `FOLD, CHECK, CALL, RAISE, ALL-IN`
- Perte: MSE sur distributions (softmax en sortie du modèle)

## Visualisations (heatmap préflop)
Depuis `ml/`:
```bash
python viz.py  # produit ml/preflop_heatmap.png à partir de trained_policy_model.pth
```
Vous pouvez ajuster `role_id` et les paramètres/buckets dans `ml/viz.py`.

## UI Next.js
L'UI vit dans `ui/`. Elle lit `public/avg_policy.json.gz`.

```bash
cd ui
npm install
npm run dev  # lance l'UI en mode développement
```

Sources clés dans `ui/src/`:
- `lib/policy.ts`, `lib/infoset.ts`, `lib/game.ts`: parsing/logic
- `components/` et `app/` pour les pages et widgets

## Format de la policy (compacte)
Chaque entrée de `avg_policy.json.gz` est:
```json
{
  "<infoset_key_u64>": {
    "policy": [bitmask, q1, q2, ...],
    "visits": <int capped>
  }
}
```
- `bitmask`: indique quelles actions (par index) sont présentes
- `q_i`: valeurs entières quantifiées (0..255), somme ajustée à 255
- Reconstruction: `prob[action] = q_i / sum(q)` selon l'ordre des bits

Les champs de l'infoset sont packés dans un `u64` (voir `infoset.py`):
- `PHASE` (3 bits), `ROLE` (2), `HAND` (8, index 13x13), `BOARD` (5), `POT` (8), `RATIO` (8), `SPR` (8), `HEROBOARD` (4)

## Structure des répertoires (extraits)
```
GTO_Bot/
  cfr_solver.py                # Entraînement CFR+, export policy
  poker_game_expresso.py       # Environnement 3-handed + logique des actions/pots
  infoset.py                   # Bucketing, pack/unpack u64, mapping 169
  policy.py                    # Chargement/sampling d'une policy moyenne compacte
  stats_policy.py              # Décodage policy -> CSV et stats
  utils.py                     # Évaluation de mains (Treys) et I/O ranges
  ml/
    model.py                   # Réseau PyTorch
    train.py                   # Pipeline d'entraînement sur la policy
    viz.py                     # Heatmaps et visualisations
  policy/
    avg_policy.json.gz         # Policy moyenne (sortie solver)
    avg_policy.csv             # Export tabulaire
  ui/                          # App Next.js/TypeScript
    public/avg_policy.json.gz  # Copie de la policy pour l'UI
```
