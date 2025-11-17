import os
import sys
import numpy as np
import logging
from datetime import datetime
import torch
from project.reward import RewardH1, RewardH2, RewardH3


def strip_extension(filename):
    base, _ = os.path.splitext(filename)  # removes .gz
    base, _ = os.path.splitext(base)      # removes .mps
    return base


def get_device(device: str = "cpue"):

    device = device.lower()
    
    if device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: GPU requested but CUDA not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device argument: '{device}'. Choose 'cpu' or 'gpu'.")


def get_reward(name):
    if name.lower() == "reward_h1":
        return RewardH1()
    if name.lower() == "reward_h2":
        return RewardH2()
    if name.lower() == "reward_h3":
        return RewardH3()
    else:
        raise ValueError(f"Unsupported reward function '{name}'.")


def shifted_geometric_mean(node_counts, shift=100):
    node_counts = np.array(node_counts)
    shifted = node_counts + shift
    sgm = np.exp(np.mean(np.log(shifted))) - shift
    return sgm

def load_checkpoint(checkpoint_path, actor_network, critic_network, actor_optimizer, critic_optimizer):
    """Load model weights, optimizer states, and training progress."""
    checkpoint = torch.load(checkpoint_path)
    actor_network.load_state_dict(checkpoint['actor_state_dict'])
    critic_network.load_state_dict(checkpoint['critic_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['best_val_nodes'], checkpoint['patience_counter']


def save_checkpoint(episode, actor_network, critic_network, actor_optimizer, critic_optimizer, best_val_nodes, patience_counter, args, trial_number=None):
    """Save model weights, optimizer states, and training progress."""
    checkpoint = {
        'episode': episode,
        'actor_state_dict': actor_network.state_dict(),
        'critic_state_dict': critic_network.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'best_val_nodes': best_val_nodes,
        'patience_counter': patience_counter,
        'args': args
    }
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_trial_{trial_number if trial_number is not None else 'manual'}_episode_{episode}.pth")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path