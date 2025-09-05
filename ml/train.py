import json
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List
from model import Model
import sys
import os
from tqdm import tqdm

# Add parent directory to path to import infoset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infoset import unpack_infoset_key_dense

# Constants from CFR solver
ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]
ACTION_INDEX = {action_name: index for index, action_name in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

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
        
        print(f"Loaded {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_policy(path: str) -> Dict:
    """Load policy data from gzipped JSON file"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    return raw

def train(model: Model, policy: dict, epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
    """Train the model on policy data"""
    
    # Create dataset and dataloader
    print("Preparing dataset...")
    dataset = PolicyDataset(policy)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {len(dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Total batches per epoch: {len(dataloader)}")
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_features, batch_targets in epoch_pbar:
            optimizer.zero_grad()            
            # Forward pass
            outputs = model(batch_features)
            
            # Calculate loss (using KL divergence for probability distributions)
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

def evaluate_model(model: Model, policy: dict, num_samples: int = 1000):
    """Evaluate model performance on a subset of policy data"""
    model.eval()
    criterion = nn.MSELoss()
    
    print("Preparing evaluation dataset...")
    dataset = PolicyDataset(policy)
    if len(dataset) == 0:
        print("No data to evaluate")
        return
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    total_kl_div = 0.0
    total_l1_error = 0.0
    
    print(f"Evaluating on {len(indices)} samples...")
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating samples"):
            features, targets = dataset[idx]
            features = features.unsqueeze(0)
            
            outputs = model(features)
            
            # KL divergence
            kl_div = criterion(outputs, targets)
            total_kl_div += kl_div.item()
            
            # L1 error
            l1_error = torch.abs(outputs - targets).mean()
            total_l1_error += l1_error.item()
    
    avg_kl_div = total_kl_div / len(indices)
    avg_l1_error = total_l1_error / len(indices)
    
    print(f"Evaluation Results:")
    print(f"  Average KL Divergence: {avg_kl_div:.6f}")
    print(f"  Average L1 Error: {avg_l1_error:.6f}")

if __name__ == "__main__":
    print("=" * 60)
    print("POKER POLICY NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Load policy data
    print("Loading policy data...")
    policy_path = "../policy/avg_policy.json.gz"
    policy_data = load_policy(policy_path)
    
    print(f"Loaded policy with {len(policy_data)} infosets")
    
    # Create model
    print("Creating neural network model...")
    input_size = 224  # Total one-hot features: 7+3+169+31+3+11 = 224
    output_size = N_ACTIONS  # Number of actions
    model = Model(input_size, output_size)
    print(f"Model created: {input_size} inputs -> {output_size} outputs")
    
    # Train model
    print("\nStarting training...")
    train(model, policy_data, epochs=50, batch_size=64, lr=0.001)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, policy_data)
    
    # Save trained model
    print("\nSaving model...")
    torch.save(model.state_dict(), "trained_policy_model.pth")
    print("Model saved to trained_policy_model.pth")
    
    print("\nTraining pipeline completed successfully!")