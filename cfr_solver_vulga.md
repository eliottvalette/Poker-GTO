### CFR+ 3-handed NLHE — Explication chronologique de `cfr_solver.py`

Fichier de training d'une politique moyenne avec CFR+ en itératif sur un jeu 3-handed. À chaque itération, on crée des mains, on parcours l’arbre de décision en “external sampling”, on met à jour les regrets positifs et la stratégie moyenne, puis on sauvegardes périodiquement.

### Vue d’ensemble rapide

- **Deux tables par infoset dense `Key`**: somme des regrets et somme des stratégies. À chaque état de décision d’un joueur, la stratégie courante vient du regret-matching+, on simule l’action, on mesure l’utilité, puis on corrige les regrets et on accumule la stratégie.

```python
self.regret_sum: Dict[Key, Dict[Action, float]] = defaultdict(_default_action_dict)
self.strategy_sum: Dict[Key, Dict[Action, float]] = defaultdict(_default_action_dict)
```

### Objets et types centraux

#### Clé d’infoset

Key: int 64-bits provenant de `build_infoset_key_fast(game, player)`. Sert d’index unique “état de décision → distributions”.

```python
key = build_infoset_key_fast(game, current_player)  # int
```

#### Tables d’apprentissage

Deux dictionnaires imbriqués, indexés par `Key`, vers un dict `action → float`.

```python
regret_sum = {
    8736249823749823: {
        "FOLD": 0.0,
        "CALL": 1.25,
        "RAISE": 0.0,
        "ALL-IN": 5.6
    },
    8736249823749824: {
        "FOLD": 2.7,
        "CALL": 0.0,
        "RAISE": 3.1
    },
    ...
}

strategy_sum = {
    8736249823749823: {
        "FOLD": 10.0,
        "CALL": 15.0,
        "RAISE": 0.0,
        "ALL-IN": 5.0
    },
    8736249823749824: {
        "FOLD": 1.0,
        "CALL": 1.0,
        "RAISE": 1.0
    },
    ...
}
```

Mises à jour côté héros dans `_traverse`:

```python
self.regret_sum[key][a]   = max(0.0, self.regret_sum[key].get(a,0.0) + reach_others * regret)
self.strategy_sum[key][a] += reach_others * sigma[a]
```

#### Générateurs aléatoires

```python
self.rng                # RNG auxiliaire
self.random_generator   # RNG opérationnel, réensemencé à chaque itération
self.random_generator.seed(self.seed + 7919 * iter_idx)
```

#### Statistiques

```python
stats = {
    "total_infosets": 25421,
    "total_actions": 48739,
    "training_time": 182.37
}
```

### Construction du solveur

- **Seed, stacks, RNG**: initialisation déterministe, rien de magique.

```python
class CFRPlusSolver:
    def __init__(..., stacks: Tuple[int,int,int]=(100,100,100)):
        self.regret_sum = defaultdict(_default_action_dict)
        self.strategy_sum = defaultdict(_default_action_dict)
        self.rng = random.Random(seed)
```

### Création d’une partie simulée

- **Nouvelle main**: on part d’un `GameInit`, on instancie `PokerGameExpresso`, on poste les blindes. Le paquet est ré-ensemencé pour la reproductibilité.

```python
def _new_game(self) -> PokerGameExpresso:
    init = GameInit(); init.stacks_init = list(self.stacks)
    init.phase = "PREFLOP"; init.community_cards = []
    random.seed(self.rng.randrange(10**9))
    game = PokerGameExpresso(init)
    game.deal_small_and_big_blind()
    return game
```

### Clé d’infoset compacte

- **Clé 64-bits dense**: construite à partir de l’état (phase, rôle, main 169, texture de board, pot, à-payer, taille de la dernière relance, nb de relances). Sert d’index vers “état → distribution d’actions”.

```python
from infoset import build_infoset_key_fast
key = build_infoset_key_fast(game, current_player)  # -> int
```

### Stratégie courante par regret-matching+

- **Regrets positifs uniquement**. S’ils sont tous ≤ 0, on joue uniforme sur les actions légales.

```python
def _strategy_from_regret(self, key, legal):
    pos = {a:max(0.0,self.regret_sum[key].get(a,0.0)) for a in legal}
    s = sum(pos.values())
    return ({a:1/len(legal) for a in legal} if s<=0 else {a:pos[a]/s for a in legal})
```

### Parcours principal (traverse)

- **Chronologie d’une itération côté héros**:
  - Si héros: évalue chaque action par snapshot/restore + rollout, puis met à jour regrets+ et stratégie moyenne pondérée par la reach des autres.
  - Si adversaire: échantillonne son action selon la stratégie courante et met à jour la probabilité d’atteinte.

```python
def _traverse(self, game, hero_role, reach_others):
    while game.current_phase != "SHOWDOWN":
        current_role = game.current_role
        key = build_infoset_key_fast(game, game.players[current_role])
        legal = self._legal_actions(game)

        if current_role == hero_role:
            sigma = self._strategy_from_regret(key, legal)
            action_util = {}
            for a in legal:
                snap = game.snapshot()
                game.process_action(game.players[current_role], a)
                u,_ = self._rollout_until_terminal(game, hero_role, reach_others)
                action_util[a] = u
                game.restore(snap)

            node_u = sum(sigma[a]*action_util[a] for a in legal)
            for a in legal:  # CFR+ update
                regret = action_util[a] - node_u
                self.regret_sum[key][a] = max(0.0, self.regret_sum[key].get(a,0.0) + reach_others*regret)
            for a in legal:  # moyenne
                self.strategy_sum[key][a] += reach_others * sigma[a]
            return node_u

        sigma = self._strategy_from_regret(key, legal)
        a = self._sample_from(sigma)
        reach_others *= sigma[a]
        game.process_action(game.players[current_role], a)

    return self._terminal_cev(game, hero_role)
```

### Rollout jusqu’au terminal

- **Après un choix du héros ou dans son évaluation locale**, on déroule la main jusqu’au showdown en échantillonnant les autres. Retourne le CEV du héros et la probabilité d’atteinte mise à jour.

```python
def _rollout_until_terminal(self, game, hero_role, reach_others):
    while game.current_phase != "SHOWDOWN":
        r = game.current_role; p = game.players[r]
        key = build_infoset_key_fast(game, p)
        legal = self._legal_actions(game)
        sigma = self._strategy_from_regret(key, legal)
        a = self._sample_from(sigma)
        if r != hero_role:
            reach_others *= sigma[a]
        game.process_action(p, a)
    return self._terminal_cev(game, hero_role), reach_others
```

### Boucle d’entraînement

- **Itératif, multi-rôle**: pour chaque itération et chaque rôle héros, on lance des parties et on appelle `_traverse`. Sauvegarde périodique d’une politique moyenne.

```python
def train(self, iterations: int = 1000) -> None:
    with trange(1, iterations+1) as pbar:
        for iter_idx in pbar:
            self.random_generator.seed(self.seed + 7919*iter_idx)
            for _ in trange(self.hands_per_iter, leave=False):
                for hero in (0,1,2):
                    game = self._new_game()
                    self._traverse(game, hero_role=hero, reach_probability_of_others=1.0)
            if SAVE_EVERY>0 and (iter_idx % SAVE_EVERY == 0):
                self.save_policy_json(f"policy/avg_policy_iter_{iter_idx}.json")
```

### Extraction et sauvegarde de la politique moyenne

- **Normalisation de `strategy_sum[key]`** pour obtenir une distribution. Si un nœud n’a rien accumulé, uniforme sur ses actions vues.

```python
def extract_average_policy(self) -> Dict[int, Dict[str, float]]:
    policy = {}
    for key, counts in self.strategy_sum.items():
        tot = sum(counts.values())
        if tot>0:
            policy[key] = {a: counts[a]/tot for a in counts}
        else:
            A = list(counts.keys()); 
            if A: policy[key] = {a: 1/len(A) for a in A}
    return policy

def save_policy_json(self, path: str) -> None:
    out = {str(k): v for k,v in self.extract_average_policy().items()}
    with open(path,"w",encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
```


### Chargement d’une politique

- **Démarrer depuis une politique moyenne** existante pour agir ou initialiser les stats.

```python
@staticmethod
def load_policy_json(path: str) -> Dict[int, Dict[str, float]]:
    with open(path,"r",encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k,v in raw.items()}
```


### Statistiques

- **Résumé simple** de la politique extraite.

```python
def print_policy_stats(self):
    policy = self.extract_average_policy()
    # compte des tailles d'action par infoset et exemples
```

### Fil complet en une phrase

Pour chaque main et chaque rôle héros, tu parcours l’arbre: aux nœuds du héros tu testes chaque action par snapshot/restore+rollout pour calculer utilités, tu mets à jour regrets+ et stratégie moyenne pondérée par la reach des autres; aux nœuds adverses tu échantillonnes selon regret-matching+, tu avances la partie; à la fin tu normalises `strategy_sum` et tu sauvegardes.
