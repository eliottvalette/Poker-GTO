import json
import gzip
from dataclasses import dataclass
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from typing import Dict, List, Optional
try:
    from .model import Model
except ImportError:
    from model import Model
import sys
import os
from tqdm import tqdm

# Add parent directory to path to import infoset
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import config
from infoset import unpack_infoset_key_dense

# Constants from CFR solver
ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]
ACTION_INDEX = {action_name: index for index, action_name in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 20_260_516


@dataclass(frozen=True)
class DatasetSplit:
    train: Dataset
    evaluation: Dataset


@dataclass(frozen=True)
class EvaluationMetrics:
    num_samples: int
    kl_divergence: float
    l1_error: float


def reconstruct_probabilities(bitmask: int, quantized_values: List[int]) -> List[float]:
    """Reconstruct probability distribution from quantized format"""
    probabilities = [0.0] * N_ACTIONS
    total_quantized = sum(quantized_values)
    
    if total_quantized <= 0:
        return probabilities
    
    index_quantized = 0
    for action_index in range(N_ACTIONS):
        if (bitmask >> action_index) & 1:
            q = quantized_values[index_quantized]
            probabilities[action_index] = q / total_quantized
            index_quantized += 1
    
    return probabilities

def infoset_to_features(infoset_key: int) -> torch.Tensor:
    """Convert infoset key to feature vector using one-hot encoding"""
    features = unpack_infoset_key_dense(infoset_key)
    
    # Initialize one-hot vectors
    phase_onehot = [0] * 7
    role_onehot = [0] * 3
    hand_onehot = [0] * 169
    board_onehot = [0] * 31
    heroboard_onehot = [0] * 11
    
    # Set the appropriate indices to 1
    phase_onehot[features["PHASE"]] = 1
    role_onehot[features["ROLE"]] = 1
    hand_onehot[features["HAND"]] = 1
    board_onehot[features["BOARD"]] = 1
    pot = features["POT"] / 255
    ratio = features["RATIO"] / 255
    spr = features["SPR"] / 255
    heroboard_onehot[features["HEROBOARD"]] = 1
    
    # Concatenate all one-hot vectors
    feature_vector = (
        phase_onehot + 
        role_onehot + 
        hand_onehot + 
        board_onehot + 
        [pot] + 
        [ratio] + 
        [spr] + 
        heroboard_onehot
    )
    
    return torch.tensor(feature_vector, dtype=torch.float32)

class PolicyDataset(Dataset):
    def __init__(self, policy_data: Dict):
        self.data = []
        
        print("Loading policy data...")
        for infoset_key_str, entry in tqdm(policy_data.items(), desc="Processing infosets"):
            infoset_key = int(infoset_key_str)
                
            policy = entry["policy"]
            bitmask = policy[0]
            quantized_values = policy[1:]
            
            # Reconstruct probabilities
            probabilities = reconstruct_probabilities(bitmask, quantized_values)
            
            # Convert to features
            features = infoset_to_features(infoset_key)
            targets = torch.tensor(probabilities, dtype=torch.float32)
            
            self.data.append((features, targets))
        
        print(f"Loaded {len(self.data)} policy samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_policy(path: str) -> Dict:
    """Load policy data from gzipped JSON file"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    return raw

def split_dataset(
    dataset: Dataset,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> DatasetSplit:
    """Split once before training so evaluation never sees train samples."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")

    total_samples = len(dataset)
    if total_samples < 2:
        raise ValueError("Need at least two samples to create a train/eval split")

    eval_size = max(1, int(round(total_samples * eval_fraction)))
    train_size = total_samples - eval_size
    if train_size <= 0:
        train_size = 1
        eval_size = total_samples - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, evaluation_dataset = random_split(dataset, [train_size, eval_size], generator=generator)
    return DatasetSplit(train=train_dataset, evaluation=evaluation_dataset)


def distribution_kl_div(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL(target || output) for probability distributions."""
    return F.kl_div(outputs.clamp_min(1e-8).log(), targets, reduction="batchmean")


def train(
    model: Model,
    train_dataset: Dataset,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
):
    """Train the model on the training split only."""
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = distribution_kl_div
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Total batches per epoch: {len(dataloader)}")
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, dynamic_ncols=True)
        for batch_features, batch_targets in epoch_pbar:
            optimizer.zero_grad()            
            # Forward pass
            outputs = model(batch_features)
            
            # Calculate loss using KL divergence for probability distributions.
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
    print("Training completed!")

def make_sampled_subset(dataset: Dataset, num_samples: Optional[int], seed: int) -> Dataset:
    """Return a deterministic evaluation subset capped by num_samples."""
    dataset_size = len(dataset)
    if num_samples is None or num_samples >= dataset_size:
        return dataset
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive or None, got {num_samples}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_size, num_samples, replace=False).tolist()
    return Subset(dataset, indices)


def evaluate_model(
    model: Model,
    eval_dataset: Dataset,
    num_samples: Optional[int] = None,
    batch_size: int = 256,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Optional[EvaluationMetrics]:
    """Evaluate model performance on the held-out split."""
    model.eval()
    criterion = distribution_kl_div

    if len(eval_dataset) == 0:
        print("No data to evaluate")
        return None

    sampled_dataset = make_sampled_subset(eval_dataset, num_samples, seed)
    dataloader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=False)

    total_kl_div = 0.0
    total_l1_error = 0.0
    evaluated_samples = 0

    print(f"Evaluating on {len(sampled_dataset)} held-out samples...")
    with torch.no_grad():
        for features, targets in tqdm(
            dataloader,
            desc="Evaluating samples",
            leave=False,
            dynamic_ncols=True,
        ):
            outputs = model(features)

            batch_size_actual = features.shape[0]
            kl_div = criterion(outputs, targets)
            total_kl_div += kl_div.item() * batch_size_actual

            l1_error = torch.abs(outputs - targets).mean()
            total_l1_error += l1_error.item() * batch_size_actual
            evaluated_samples += batch_size_actual

    avg_kl_div = total_kl_div / evaluated_samples
    avg_l1_error = total_l1_error / evaluated_samples

    metrics = EvaluationMetrics(
        num_samples=evaluated_samples,
        kl_divergence=avg_kl_div,
        l1_error=avg_l1_error,
    )
    tqdm.write("")
    tqdm.write("Evaluation Results:")
    tqdm.write(f"  Held-out samples: {metrics.num_samples}")
    tqdm.write(f"  Average KL Divergence: {metrics.kl_divergence:.6f}")
    tqdm.write(f"  Average L1 Error: {metrics.l1_error:.6f}")
    tqdm.write(
        "HELDOUT_EVAL "
        f"samples={metrics.num_samples} "
        f"kl={metrics.kl_divergence:.6f} "
        f"l1={metrics.l1_error:.6f}"
    )
    return metrics

if __name__ == "__main__":
    print("=" * 60)
    print("POKER POLICY NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Load policy data
    print("Loading policy data...")
    policy_path = config.ML_POLICY_PATH
    policy_data = load_policy(policy_path)
    
    print(f"Loaded policy with {len(policy_data)} infosets")

    print("Preparing dataset and leak-free split...")
    dataset = PolicyDataset(policy_data)
    eval_fraction = getattr(config, "ML_EVAL_FRACTION", DEFAULT_EVAL_FRACTION)
    split_seed = getattr(config, "ML_SPLIT_SEED", DEFAULT_SPLIT_SEED)
    dataset_split = split_dataset(dataset, eval_fraction=eval_fraction, seed=split_seed)
    print(
        "Dataset split: "
        f"train={len(dataset_split.train)}, "
        f"eval={len(dataset_split.evaluation)}, "
        f"eval_fraction={eval_fraction}, "
        f"seed={split_seed}"
    )
    
    # Create model
    print("Creating neural network model...")
    input_size = 224  # Total one-hot features: 7+3+169+31+3+11 = 224
    output_size = N_ACTIONS  # Number of actions
    model = Model(input_size, output_size)
    print(f"Model created: {input_size} inputs -> {output_size} outputs")
    
    # Train model
    print("\nStarting training...")
    train(
        model,
        dataset_split.train,
        epochs=config.ML_EPOCHS,
        batch_size=config.ML_BATCH_SIZE,
        lr=config.ML_LEARNING_RATE,
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(
        model,
        dataset_split.evaluation,
        num_samples=config.ML_EVAL_SAMPLES,
        batch_size=config.ML_BATCH_SIZE,
        seed=split_seed,
    )
    
    # Save trained model
    print("\nSaving model...")
    config.ML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.ML_MODEL_PATH)
    print(f"Model saved to {config.ML_MODEL_PATH}")
    
    print("\nTraining pipeline completed successfully!")
