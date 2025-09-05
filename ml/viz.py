import os
import sys
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow importing from project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from infoset import pack_u64, unpack_infoset_key_dense
from ml.train import infoset_to_features, N_ACTIONS
from ml.model import Model

# Constants
RANK_LABELS = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
PHASE_PREFLOP = 0
BOARD_BUCKET_PF = 3
HEROBOARD_PF = 0

# Indices must match ACTIONS order in training
FOLD_IDX = 0
CHECK_IDX = 1
CALL_IDX = 2
RAISE_IDX = 3
ALLIN_IDX = 4

def build_preflop_key(hand_index, role_id) -> int:
    """Build an infoset key for a preflop state given the 0..168 hand index."""
    # Preflop buckets set to minimal buckets
    return pack_u64(
        PHASE=PHASE_PREFLOP,
        ROLE=role_id,
        HAND=hand_index,
        BOARD=BOARD_BUCKET_PF,
        POT=0,
        RATIO=0,
        SPR=0,
        HEROBOARD=HEROBOARD_PF,
    )

def hand_index_from_grid(i: int, j: int) -> int:
    """Map grid coordinates (i row, j col, 0..12, A..2) to 0..168 hand index.
    Matches the 13x13 construction used in infoset labels.
    """
    return i * 13 + j

def grid_coords_from_hand_index(hand_index: int) -> tuple[int, int]:
    return hand_index // 13, hand_index % 13

def reconstruct_probabilities(bitmask: int, quantized_values: list[int]) -> np.ndarray:
    probs = np.zeros(N_ACTIONS, dtype=np.float32)
    total_q = float(sum(quantized_values))
    if total_q <= 0:
        return probs
    q_idx = 0
    for a in range(N_ACTIONS):
        if (bitmask >> a) & 1:
            probs[a] = quantized_values[q_idx] / total_q
            q_idx += 1
    return probs

def generate_preflop_heatmap(model, role_id, save_path) -> str:
    """Generate and save a heatmap of (RAISE+ALL-IN)/FOLD for all 169 preflop hands.
    Returns the path to the saved image.
    """
    model.eval()
    values = np.zeros((13, 13), dtype=np.float32)
    eps = 1e-6

    with torch.no_grad():
        for i in tqdm(range(13), desc="Rows (high card)"):
            for j in range(13):
                hidx = hand_index_from_grid(i, j)
                key = build_preflop_key(hidx, role_id)
                x = infoset_to_features(key).unsqueeze(0)
                probs = model(x)[0].cpu().numpy()
                p_fold = float(probs[FOLD_IDX])
                p_raise = float(probs[RAISE_IDX])
                p_allin = float(probs[ALLIN_IDX])
                ratio = (p_raise + p_allin) / max(p_fold, eps)
                values[i, j] = ratio

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(values, annot=False, cmap="magma", cbar=True)
    ax.set_xticks(np.arange(13) + 0.5)
    ax.set_yticks(np.arange(13) + 0.5)
    ax.set_xticklabels(RANK_LABELS, rotation=0)
    ax.set_yticklabels(RANK_LABELS, rotation=0)
    ax.set_xlabel("Low card (cols) — suited above diagonal, offsuit below")
    ax.set_ylabel("High card (rows)")
    ax.set_title("Preflop (RAISE+ALL-IN) / FOLD ratio")
    plt.tight_layout()

    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path

def generate_preflop_heatmap_model_exhaustive(model, role_id, save_path) -> str:
    """Enumerate ALL preflop BTN infosets (by bucket indices) and average the
    (RAISE+ALL-IN)/FOLD ratio per HAND. No visit weighting; uniform over enumerated
    preflop buckets.

    Buckets enumerated:
      - HAND: 0..168
      - PHASE: PREFLOP
      - ROLE: provided role_id (e.g., BTN=2)
      - BOARD: 0 (no board)
      - POT: 0..22  (derived from _POT_EDGES_BB length)
      - RATIO: 0..6 (derived from _RATIO_EDGES length)
      - SPR: 0..6   (derived from _SPR_EDGES length)
      - HEROBOARD: 0 (preflop)
    """
    model.eval()
    eps = 1e-9
    sums = np.zeros(169, dtype=np.float64)
    counts = np.zeros(169, dtype=np.int64)

    with torch.no_grad():
        for hand_idx in tqdm(range(169), desc="Hands (0..168)"):
            pot_idx = 3
            ratio_idx = 4
            spr_idx = 6
            key = pack_u64(
                PHASE=PHASE_PREFLOP,
                ROLE=role_id,
                HAND=hand_idx,
                BOARD=BOARD_BUCKET_PF,
                POT=pot_idx,
                RATIO=ratio_idx,
                SPR=spr_idx,
                HEROBOARD=HEROBOARD_PF,
            )
            x = infoset_to_features(key).unsqueeze(0)
            probs = model(x)[0].cpu().numpy()
            p_fold = float(probs[FOLD_IDX])
            p_raise_allin = float(probs[RAISE_IDX] + probs[ALLIN_IDX])
            ratio = p_raise_allin / max(p_fold, eps)
            sums[hand_idx] += ratio
            counts[hand_idx] += 1

    values = np.zeros((13, 13), dtype=np.float32)
    for hand_idx in range(169):
        i, j = grid_coords_from_hand_index(hand_idx)
        values[i, j] = float(sums[hand_idx] / counts[hand_idx]) if counts[hand_idx] > 0 else 0.0

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(values, annot=False, cmap="magma", cbar=True)
    ax.set_xticks(np.arange(13) + 0.5)
    ax.set_yticks(np.arange(13) + 0.5)
    ax.set_xticklabels(RANK_LABELS, rotation=0)
    ax.set_yticklabels(RANK_LABELS, rotation=0)
    ax.set_xlabel("Low card (cols) — suited above diagonal, offsuit below")
    ax.set_ylabel("High card (rows)")
    ax.set_title("Preflop mean (RAISE+ALL-IN)/FOLD — exhaustive model avg (BTN)")
    plt.tight_layout()

    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path

def generate_preflop_heatmap_from_policy(role_id, save_path) -> str:
    """Generate heatmap using the mean of (RAISE+ALL-IN)/FOLD per hand,
    weighted by number of infosets for that hand (i.e., simple average over infosets),
    based on the saved policy JSON.
    """
    sums = np.zeros(169, dtype=np.float64)
    counts = np.zeros(169, dtype=np.int64)

    values = np.zeros((13, 13), dtype=np.float32)
    for hand_idx in range(169):
        i, j = grid_coords_from_hand_index(hand_idx)
        if counts[hand_idx] > 0:
            values[i, j] = float(sums[hand_idx] / counts[hand_idx])
        else:
            values[i, j] = 0.0

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(values, annot=False, cmap="magma", cbar=True)
    ax.set_xticks(np.arange(13) + 0.5)
    ax.set_yticks(np.arange(13) + 0.5)
    ax.set_xticklabels(RANK_LABELS, rotation=0)
    ax.set_yticklabels(RANK_LABELS, rotation=0)
    ax.set_xlabel("Low card (cols) — suited above diagonal, offsuit below")
    ax.set_ylabel("High card (rows)")
    ax.set_title("Preflop mean (RAISE+ALL-IN)/FOLD, weighted by infoset count")
    plt.tight_layout()

    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path

def load_model(weights_path, input_size, output_size) -> Model:
    model = Model(input_size, output_size)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

if __name__ == "__main__":
    role_id = 2 # BTN
    
    print("Loading model...")
    model = load_model("trained_policy_model.pth", 224, N_ACTIONS)

    print("Generating exhaustive preflop heatmap (BTN) -> preflop_heatmap.png")
    out = generate_preflop_heatmap_model_exhaustive(model, role_id=role_id, save_path="preflop_heatmap.png")
    
    print(f"Saved: {out}")
