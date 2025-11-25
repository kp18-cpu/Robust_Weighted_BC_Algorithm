import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
import numpy as np
import minari
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset
import hashlib
import json
import pickle

print("*** Full Pipeline: ROBUSTNESS TEST - Weighted BC vs. BCQ, and Standard BC on a POISONED Dataset ***")

# ---> Configuration & Hyperparameters <---
print("\n--- Initializing Configuration & Hyperparameters ---")

# Data parameters
MINARI_DATASET_ID = 'mujoco/halfcheetah/medium-v0' 
D_REF_PERCENTAGE = 0.20 # Top 20% used as trusted reference set

# --- REFERENCE SET INTEGRITY PARAMETERS ---
ENABLE_REFERENCE_INTEGRITY_CHECK = True 
INTEGRITY_HASH_ALGORITHM = 'sha256' 
INTEGRITY_CHECK_MODE = 'full_set' # Options: 'full_set', 'individual_episodes', 'merkle_tree'
INTEGRITY_HASH_FILE = 'reference_set_integrity.json' 
VERIFY_BEFORE_DISCRIMINATOR_TRAINING = True 

# --- POISONING PARAMETERS (ACTION POISONING) ---
PERCENTAGE_TO_POISON_ACTION = 0.0 
ACTION_NOISE_LEVEL = 0.8 

# Discriminator parameters
DISCRIMINATOR_HIDDEN_DIM = 256
DISCRIMINATOR_LR_BETA = 0.0005
DISCRIMINATOR_BATCH_SIZE = 64
DISCRIMINATOR_EPOCHS = 50

# Weight calculation parameters
PI_REF_PRIOR = 0.5
EPSILON_CLIP = 0.001
C_CLIP = 2.0 
NORMALIZE_WEIGHTS = False 

# Policy Network parameters
POLICY_HIDDEN_DIM = 256
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Weighted BC & Standard BC Specifics
BC_LEARNING_RATE_ALPHA = 0.0003
BC_BATCH_SIZE = 256
BC_EPOCHS = 60

# BCQ Hyperparameters
BCQ_BATCH_SIZE = 256
BCQ_TRAINING_STEPS = 150000
BCQ_LR = 1e-3
BCQ_DISCOUNT_GAMMA = 0.99
BCQ_TAU = 0.005 
BCQ_LATENT_DIM_MULTIPLIER = 2 
BCQ_PERTURBATION_PHI = 0.05 
BCQ_N_ACTION_SAMPLES = 100 

# BRAC Hyperparameters
BRAC_BATCH_SIZE = 256
BRAC_TRAINING_STEPS = 150000
BRAC_LR = 3e-4
BRAC_DISCOUNT_GAMMA = 0.99
BRAC_TAU = 0.005 
BRAC_ALPHA = 4 # Behavior regularization weight
BRAC_BEHAVIOR_PRETRAIN_EPOCHS = 100 

# Evaluation parameters
NUM_EVAL_EPISODES = 50 
EVAL_SEED = 42
MAX_EVAL_STEPS_PER_EPISODE_CONFIG = 1000

# Reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Section 1: Data Loading and Preparation ---
print(f"\n--- Section 1: Loading and Preparing Dataset: {MINARI_DATASET_ID} ---")
try:
    dataset_minari = minari.load_dataset(MINARI_DATASET_ID)
    env_for_info = dataset_minari.recover_environment()
    ACTION_SPACE_LOW = env_for_info.action_space.low
    ACTION_SPACE_HIGH = env_for_info.action_space.high
    MAX_ACTION = float(ACTION_SPACE_HIGH[0]) 
    env_for_info.close()
    print(f"Dataset '{MINARI_DATASET_ID}' loaded successfully.")
    print(f"Retrieved Action Space: Low={ACTION_SPACE_LOW}, High={ACTION_SPACE_HIGH}, Max Action: {MAX_ACTION}")
except Exception as e:
    print(f"ERROR: Could not load Minari dataset or recover environment '{MINARI_DATASET_ID}'. Exception: {e}")
    exit()

print("Calculating returns and sorting episodes...")
episode_returns = []
for episode_obj in dataset_minari:
    if episode_obj.rewards is None or len(episode_obj.rewards) == 0:
        continue
    total_reward = np.sum(episode_obj.rewards)
    episode_returns.append((total_reward, episode_obj))

episode_returns.sort(key=lambda x: x[0], reverse=True) 
all_sorted_episodes = [ep for r, ep in episode_returns]

if not all_sorted_episodes:
    raise ValueError("Dataset is empty after processing. Check dataset content.")

# Split into reference (trusted) and non-reference (pool for contamination)
ref_size = int(len(all_sorted_episodes) * D_REF_PERCENTAGE)
D_ref_episodes = all_sorted_episodes[:ref_size]
D_non_ref_episodes = all_sorted_episodes[ref_size:]

print(f"D_ref_episodes (trusted reference set) created with {len(D_ref_episodes)} episodes.")
print(f"D_non_ref_episodes (pool for contamination) created with {len(D_non_ref_episodes)} episodes.")

print("\nCreating the main CONTAMINATED dataset for training (D_main_contaminated_episodes)...")
D_main_contaminated_episodes = []
poisoned_transition_count = 0

def extract_transitions_for_replay_buffer(episode_data_dict):
    """
    Extracts transitions (s, a, r, s', done) from a Minari episode dictionary.
    Handles the N+1 observation structure required for next_state.
    """
    obs = np.array(episode_data_dict['observations'])
    acts = np.array(episode_data_dict['actions'])
    rewards = np.array(episode_data_dict['rewards'])
    terminals_flags = np.array(episode_data_dict['terminations'], dtype=float)

    transitions = []
    
    # Valid transitions are limited by the shortest array and observation length - 1
    num_transitions = min(len(acts), len(rewards), len(terminals_flags), len(obs) - 1)

    for t in range(num_transitions):
        transitions.append({
            'observations': obs[t],
            'actions': acts[t],
            'rewards': rewards[t],
            'next_observations': obs[t+1],  
            'terminals': terminals_flags[t]
        })
    return transitions


def extract_obs_actions_pairs_from_minari(episode, actions_override=None):
    """
    Extracts (obs, action) pairs from a Minari EpisodeData object.
    Allows action overriding for poisoning injection.
    """
    obs = np.array(episode.observations)
    acts = np.array(actions_override if actions_override is not None else episode.actions)
    
    # align lengths
    min_len = min(len(obs), len(acts))
    return obs[:min_len], acts[:min_len]

# 1. Add clean reference episodes
for ep in D_ref_episodes:
    obs_ref, acts_ref = extract_obs_actions_pairs_from_minari(ep)
    D_main_contaminated_episodes.append({
        "observations": obs_ref,
        "actions": acts_ref,
        "rewards": ep.rewards, 
        "terminations": ep.terminations, 
        "poisoned": False
    })

# 2. Add non-reference episodes, poisoning a specific fraction
if D_non_ref_episodes:
    action_range = ACTION_SPACE_HIGH - ACTION_SPACE_LOW
    noise_magnitude = ACTION_NOISE_LEVEL * action_range
    num_to_poison = int(len(D_non_ref_episodes) * PERCENTAGE_TO_POISON_ACTION)
    
    indices_to_poison = set(random.sample(range(len(D_non_ref_episodes)), num_to_poison))
    print(f"Poisoning {num_to_poison} out of {len(D_non_ref_episodes)} non-reference episodes.")

    for i, episode_obj in enumerate(D_non_ref_episodes):
        is_poisoned = i in indices_to_poison
        original_actions = np.array(episode_obj.actions)
        new_actions = original_actions.copy() 

        if is_poisoned and original_actions.shape[0] > 0: 
            noise = np.random.normal(0, 1, size=original_actions.shape) * noise_magnitude
            poisoned_actions = original_actions + noise
            new_actions = np.clip(poisoned_actions, ACTION_SPACE_LOW, ACTION_SPACE_HIGH)
            poisoned_transition_count += len(new_actions) 

        obs_non_ref = np.array(episode_obj.observations) 
        D_main_contaminated_episodes.append({
            "observations": obs_non_ref,
            "actions": new_actions, 
            "rewards": episode_obj.rewards, 
            "terminations": episode_obj.terminations, 
            "poisoned": is_poisoned
        })

print(f"Created D_main_contaminated_episodes with {len(D_main_contaminated_episodes)} total episodes.")
print(f"Total poisoned (s,a) pairs added: {poisoned_transition_count}")


# --- Section 1.5: Reference Set Integrity Checking ---
print("\n--- Section 1.5: Reference Set Integrity Checking ---")

def serialize_episode_for_hashing(episode):
    """
    Serializes a Minari episode to a deterministic byte representation for hashing.
    """
    obs = np.array(episode.observations)
    acts = np.array(episode.actions)
    rewards = np.array(episode.rewards)
    terminations = np.array(episode.terminations)
    
    # Use pickle protocol 4 for deterministic serialization of arrays
    episode_data = {
        'observations': obs.tobytes(),
        'actions': acts.tobytes(),
        'rewards': rewards.tobytes(),
        'terminations': terminations.tobytes(),
        'obs_shape': obs.shape,
        'acts_shape': acts.shape,
        'rewards_shape': rewards.shape,
        'terminations_shape': terminations.shape
    }
    
    return pickle.dumps(episode_data, protocol=4)

def compute_episode_hash(episode, hash_algorithm='sha256'):
    """Computes cryptographic hash of a single episode."""
    serialized = serialize_episode_for_hashing(episode)
    hash_obj = hashlib.new(hash_algorithm)
    hash_obj.update(serialized)
    return hash_obj.hexdigest()

def compute_reference_set_hash_full_set(ref_episodes, hash_algorithm='sha256'):
    """Computes a single hash for the entire reference set."""
    hash_obj = hashlib.new(hash_algorithm)
    
    for i, episode in enumerate(ref_episodes):
        episode_hash = compute_episode_hash(episode, hash_algorithm)
        hash_obj.update(f"episode_{i}:{episode_hash}".encode('utf-8'))
    
    return hash_obj.hexdigest()

def compute_reference_set_hash_individual(ref_episodes, hash_algorithm='sha256'):
    """Computes individual hashes for each episode."""
    episode_hashes = {}
    for i, episode in enumerate(ref_episodes):
        episode_hashes[i] = compute_episode_hash(episode, hash_algorithm)
    return episode_hashes

def compute_reference_set_hash_merkle(ref_episodes, hash_algorithm='sha256'):
    """Computes a Merkle tree hash for the reference set for granular tamper detection."""
    leaf_hashes = [compute_episode_hash(ep, hash_algorithm) for ep in ref_episodes]
    
    current_level = leaf_hashes
    tree_levels = [current_level]
    
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                combined = current_level[i] + current_level[i + 1]
            else:
                combined = current_level[i] + current_level[i]
            
            hash_obj = hashlib.new(hash_algorithm)
            hash_obj.update(combined.encode('utf-8'))
            next_level.append(hash_obj.hexdigest())
        
        current_level = next_level
        tree_levels.append(current_level)
    
    root_hash = current_level[0] if current_level else None
    return root_hash, tree_levels

def save_reference_set_integrity(ref_episodes, hash_file, hash_algorithm='sha256', check_mode='full_set'):
    """Computes and saves the integrity hash(es) of the reference set to a file."""
    integrity_data = {
        'hash_algorithm': hash_algorithm,
        'check_mode': check_mode,
        'num_episodes': len(ref_episodes),
        'dataset_id': MINARI_DATASET_ID,
        'ref_percentage': D_REF_PERCENTAGE,
        'seed': GLOBAL_SEED,
        'timestamp': None 
    }
    
    if check_mode == 'full_set':
        integrity_data['reference_set_hash'] = compute_reference_set_hash_full_set(ref_episodes, hash_algorithm)
    elif check_mode == 'individual_episodes':
        integrity_data['episode_hashes'] = compute_reference_set_hash_individual(ref_episodes, hash_algorithm)
    elif check_mode == 'merkle_tree':
        root_hash, tree_levels = compute_reference_set_hash_merkle(ref_episodes, hash_algorithm)
        integrity_data['merkle_root_hash'] = root_hash
        integrity_data['merkle_tree_levels'] = len(tree_levels)
    
    with open(hash_file, 'w') as f:
        json.dump(integrity_data, f, indent=2)
    
    print(f"Reference set integrity hash saved to: {hash_file}")
    if check_mode == 'full_set':
        print(f"  Reference Set Hash ({hash_algorithm}): {integrity_data['reference_set_hash'][:16]}...")
    elif check_mode == 'individual_episodes':
        print(f"  Computed {len(integrity_data['episode_hashes'])} individual episode hashes")
    elif check_mode == 'merkle_tree':
        print(f"  Merkle Root Hash ({hash_algorithm}): {integrity_data['merkle_root_hash'][:16]}...")
    
    return integrity_data

def verify_reference_set_integrity(ref_episodes, hash_file, hash_algorithm='sha256', check_mode='full_set', strict=True):
    """
    Verifies the integrity of the reference set against a stored hash.
    If strict=True, raises an exception on mismatch.
    """
    if not os.path.exists(hash_file):
        error_msg = f"Integrity hash file not found: {hash_file}. Cannot verify reference set integrity."
        if strict:
            raise FileNotFoundError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
            return False
    
    with open(hash_file, 'r') as f:
        stored_integrity = json.load(f)
    
    if stored_integrity.get('hash_algorithm') != hash_algorithm:
        error_msg = f"Hash algorithm mismatch: stored={stored_integrity.get('hash_algorithm')}, current={hash_algorithm}"
        if strict:
            raise ValueError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
            return False
    
    if stored_integrity.get('check_mode') != check_mode:
        error_msg = f"Check mode mismatch: stored={stored_integrity.get('check_mode')}, current={check_mode}"
        if strict:
            raise ValueError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
            return False
    
    if stored_integrity.get('num_episodes') != len(ref_episodes):
        error_msg = f"Episode count mismatch: stored={stored_integrity.get('num_episodes')}, current={len(ref_episodes)}"
        if strict:
            raise ValueError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
            return False
    
    integrity_passed = False
    
    if check_mode == 'full_set':
        current_hash = compute_reference_set_hash_full_set(ref_episodes, hash_algorithm)
        stored_hash = stored_integrity.get('reference_set_hash')
        integrity_passed = (current_hash == stored_hash)
        if integrity_passed:
            print(f"✓ Reference Set Integrity Verified ({hash_algorithm}): {current_hash[:16]}...")
        else:
            error_msg = f"✗ Reference Set Integrity FAILED! Hash mismatch detected."
            print(f"  Stored: {stored_hash[:16]}...")
            print(f"  Current: {current_hash[:16]}...")
            if strict:
                raise ValueError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
    
    elif check_mode == 'individual_episodes':
        current_hashes = compute_reference_set_hash_individual(ref_episodes, hash_algorithm)
        stored_hashes = stored_integrity.get('episode_hashes', {})
        
        mismatches = []
        for idx, current_hash in current_hashes.items():
            stored_hash = stored_hashes.get(str(idx))
            if stored_hash != current_hash:
                mismatches.append(idx)
        
        if len(mismatches) == 0:
            integrity_passed = True
            print(f"✓ All {len(current_hashes)} Episodes Integrity Verified")
        else:
            error_msg = f"✗ Integrity FAILED! {len(mismatches)} episode(s) mismatched: {mismatches[:10]}"
            if strict:
                raise ValueError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
    
    elif check_mode == 'merkle_tree':
        current_root, _ = compute_reference_set_hash_merkle(ref_episodes, hash_algorithm)
        stored_root = stored_integrity.get('merkle_root_hash')
        integrity_passed = (current_root == stored_root)
        if integrity_passed:
            print(f"✓ Merkle Root Integrity Verified ({hash_algorithm}): {current_root[:16]}...")
        else:
            error_msg = f"✗ Merkle Root Integrity FAILED! Root hash mismatch detected."
            print(f"  Stored: {stored_root[:16]}...")
            print(f"  Current: {current_root[:16]}...")
            if strict:
                raise ValueError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
    
    return integrity_passed

# Run the integrity check logic
reference_integrity_data = None
if ENABLE_REFERENCE_INTEGRITY_CHECK:
    if os.path.exists(INTEGRITY_HASH_FILE):
        print(f"Found existing integrity file: {INTEGRITY_HASH_FILE}")
        print("Verifying reference set integrity...")
        verify_reference_set_integrity(
            D_ref_episodes, 
            INTEGRITY_HASH_FILE, 
            INTEGRITY_HASH_ALGORITHM, 
            INTEGRITY_CHECK_MODE,
            strict=True  
        )
    else:
        print(f"No existing integrity file found. Computing and saving reference set integrity hash...")
        reference_integrity_data = save_reference_set_integrity(
            D_ref_episodes,
            INTEGRITY_HASH_FILE,
            INTEGRITY_HASH_ALGORITHM,
            INTEGRITY_CHECK_MODE
        )
else:
    print("Reference set integrity checking is DISABLED.")


# --- Section 2: Helper Functions for Trajectory Preprocessing ---
print("\n--- Section 2: Defining Helper Functions ---")

def get_max_traj_length_and_dims(episodes_list_of_dict):
    """
    Determines max trajectory length, observation dim, and action dim from episode list.
    """
    max_len, obs_d, act_d = 0, 0, 0
    if not episodes_list_of_dict: return 0, 0, 0
    
    for ep_data_check in episodes_list_of_dict:
        obs_array = ep_data_check.get('observations')
        act_array = ep_data_check.get('actions')
        if obs_array is not None and len(obs_array) > 0:
            if obs_array.ndim == 1: obs_array = obs_array.reshape(-1, 1) 
            obs_d = obs_array.shape[1]
        if act_array is not None and len(act_array) > 0:
            if act_array.ndim == 1: act_array = act_array.reshape(-1, 1) 
            act_d = act_array.shape[1]
        if obs_d > 0 and act_d > 0: break 
            
    for ep_data in episodes_list_of_dict:
        obs_array = ep_data.get('observations')
        if obs_array is not None:
            max_len = max(max_len, len(obs_array))
    return max_len, obs_d, act_d

MAX_TRAJ_LENGTH, OBS_DIM, ACTION_DIM = get_max_traj_length_and_dims(D_main_contaminated_episodes)
if MAX_TRAJ_LENGTH == 0 or OBS_DIM == 0 or ACTION_DIM == 0: raise ValueError("Failed to determine valid trajectory dimensions.")
FLATTENED_INPUT_DIM = MAX_TRAJ_LENGTH * (OBS_DIM + ACTION_DIM)
print(f"Determined Trajectory Info: MAX_TRAJ_LENGTH={MAX_TRAJ_LENGTH}, OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM}, FLATTENED_INPUT_DIM={FLATTENED_INPUT_DIM}")

def flatten_and_pad_trajectory(ep_data_dict, max_len, obs_d, act_d):
    """
    Flattens and pads an episode's observations and actions into a single vector
    for the discriminator input.
    """
    obs_list = ep_data_dict.get('observations')
    act_list = ep_data_dict.get('actions')

    obs = np.asarray(obs_list) if obs_list is not None else np.empty((0, obs_d))
    act = np.asarray(act_list) if act_list is not None else np.empty((0, act_d))

    # Reshape 1D arrays to 2D
    if obs.ndim == 1 and obs_d > 0: obs = obs.reshape(-1, obs_d)
    elif obs.ndim != 2 and obs.shape[0] > 0: obs = np.empty((0, obs_d)) 
    
    if act.ndim == 1 and act_d > 0: act = act.reshape(-1, act_d)
    elif act.ndim != 2 and act.shape[0] > 0: act = np.empty((0, act_d)) 
        
    padded_obs = np.zeros((max_len, obs_d))
    padded_act = np.zeros((max_len, act_d))
    
    len_obs = obs.shape[0]
    len_act = act.shape[0]
    
    # Pad to max_len
    if len_obs > 0: padded_obs[:min(len_obs, max_len), :] = obs[:min(len_obs, max_len), :]
    if len_act > 0: padded_act[:min(len_act, max_len), :] = act[:min(len_act, max_len), :]
    
    return np.concatenate((padded_obs, padded_act), axis=1).flatten()

# --- Section 3: Discriminator Definition and Training ---
print("\n--- Section 3: Discriminator Definition and Training ---")
class TrajectoryDiscriminator(nn.Module):
    """
    Classifies trajectories as reference (1) or contaminated (0).
    """
    def __init__(self, input_dim, hidden_dim=DISCRIMINATOR_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

class DiscriminatorDataset(Dataset):
    """
    Combines reference and contaminated trajectories with labels for the discriminator.
    """
    def __init__(self, d_ref_episodes, d_all_contaminated_episodes, max_l, o_d, a_d):
        self.samples = []
        # Label 1: Trusted, reference trajectories
        for ep in d_ref_episodes:
            if hasattr(ep, 'observations') and ep.observations is not None and len(ep.observations) > 0:
                ep_dict_for_flatten = {"observations": ep.observations, "actions": ep.actions}
                self.samples.append((flatten_and_pad_trajectory(ep_dict_for_flatten, max_l, o_d, a_d), 1))
        
        # Label 0: All trajectories from the contaminated pool
        for ep_contam_dict in d_all_contaminated_episodes:
            if ep_contam_dict.get('observations') is not None and len(ep_contam_dict['observations']) > 0:
                self.samples.append((flatten_and_pad_trajectory(ep_contam_dict, max_l, o_d, a_d), 0))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_f, l = self.samples[idx]
        return torch.FloatTensor(t_f), torch.FloatTensor([l])

discriminator_model = TrajectoryDiscriminator(FLATTENED_INPUT_DIM).to(device)
opt_d = optim.Adam(discriminator_model.parameters(), lr=DISCRIMINATOR_LR_BETA)
crit_bce = nn.BCELoss()

if not D_ref_episodes or not D_main_contaminated_episodes:
    print("WARNING: Skipping Discriminator Training due to empty reference or main dataset.")
else:
    if ENABLE_REFERENCE_INTEGRITY_CHECK and VERIFY_BEFORE_DISCRIMINATOR_TRAINING:
        print("\n--- Pre-Discriminator Training: Verifying Reference Set Integrity ---")
        if os.path.exists(INTEGRITY_HASH_FILE):
            verify_reference_set_integrity(
                D_ref_episodes,
                INTEGRITY_HASH_FILE,
                INTEGRITY_HASH_ALGORITHM,
                INTEGRITY_CHECK_MODE,
                strict=True  
            )
            print("Reference set integrity verified. Proceeding with discriminator training.")
        else:
            print(f"WARNING: Integrity file {INTEGRITY_HASH_FILE} not found. Cannot verify reference set.")
            print("Proceeding with discriminator training anyway (integrity check disabled).")
    
    print("Creating Discriminator Dataset/DataLoader...");
    d_pytorch_dset = DiscriminatorDataset(D_ref_episodes, D_main_contaminated_episodes, MAX_TRAJ_LENGTH, OBS_DIM, ACTION_DIM)
    if len(d_pytorch_dset) > 0:
        d_loader = DataLoader(d_pytorch_dset, batch_size=DISCRIMINATOR_BATCH_SIZE, shuffle=True)
        print("\n--- Starting Discriminator Training ---")
        discriminator_model.train()
        for ep in range(DISCRIMINATOR_EPOCHS):
            ep_l_d, c_p_d, t_s_d = 0, 0, 0
            for t_f, lbls in d_loader:
                t_f, lbls = t_f.to(device), lbls.to(device)
                opt_d.zero_grad()
                outs = discriminator_model(t_f)
                loss = crit_bce(outs, lbls)
                loss.backward()
                opt_d.step()
                ep_l_d += loss.item()
                pred = (outs > 0.5).float()
                c_p_d += (pred == lbls).sum().item()
                t_s_d += lbls.size(0)
            avg_l_d, acc_d = (ep_l_d / len(d_loader) if len(d_loader) > 0 else 0), (c_p_d / t_s_d if t_s_d > 0 else 0)
            if (ep + 1) % 10 == 0:
                print(f"Disc Epoch {ep+1}, Loss: {avg_l_d:.4f}, Acc: {acc_d:.4f}")
        print("--- Discriminator Training Finished ---")
    else:
        print("DiscriminatorDataset empty. Skipping Training.")

# --- Section 4: Calculate Trajectory Weights (w_i) ---
print(f"\n--- Section 4: Calculate Trajectory Weights (w_i) for D_main_contaminated_episodes ---")
trajectory_weights = []
discriminator_model.eval()
with torch.no_grad():
    for ep_dict in D_main_contaminated_episodes:
        d_phi_t = 0.0
        if ep_dict['observations'] is not None and len(ep_dict['observations']) > 0:
            flat_ep = flatten_and_pad_trajectory(ep_dict, MAX_TRAJ_LENGTH, OBS_DIM, ACTION_DIM)
            flat_ep_t = torch.FloatTensor(flat_ep).unsqueeze(0).to(device)
            d_phi_t = discriminator_model(flat_ep_t).item()
            
        # Calculate density ratio: r_t = (p_ref / p_all)
        if d_phi_t >= 1.0 - 1e-9: 
            r_t = float('inf')
        elif d_phi_t <= 1e-9: 
            r_t = 0.0
        else:
            r_t = (d_phi_t / (1.0 - d_phi_t)) * ((1.0 - PI_REF_PRIOR) / PI_REF_PRIOR)
            
        clipped_r_t = np.clip(r_t, EPSILON_CLIP, C_CLIP)
        trajectory_weights.append(clipped_r_t)

if NORMALIZE_WEIGHTS and trajectory_weights:
    sum_w = np.sum(trajectory_weights)
    if sum_w > 1e-9:
        trajectory_weights = [w / sum_w for w in trajectory_weights]
        print("Trajectory weights normalized.")
print(f"Calculated {len(trajectory_weights)} weights for the main contaminated dataset.");

# --- Section 5a: Weighted BC Policy Definition and Training ---
print("\n--- Section 5a: Weighted BC Policy Definition and Training (on Contaminated Data) ---")
class GaussianPolicyNetwork(nn.Module):
    """
    Gaussian policy network with Tanh squashing and log-probability correction.
    """
    def __init__(self, o_d, a_d, h_d=POLICY_HIDDEN_DIM, log_min=LOG_STD_MIN, log_max=LOG_STD_MAX):
        super().__init__()
        self.log_min, self.log_max = log_min, log_max
        self.fc1 = nn.Linear(o_d, h_d)
        self.fc2 = nn.Linear(h_d, h_d)
        self.mean_h = nn.Linear(h_d, a_d)
        self.log_std_h = nn.Linear(h_d, a_d)
        self.relu = nn.ReLU()
        self.max_action_val = MAX_ACTION 

    def forward(self, s):
        """Outputs mean and std."""
        x = self.relu(self.fc1(s))
        x = self.relu(self.fc2(x))
        mean = self.mean_h(x)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(self.log_std_h(x), self.log_min, self.log_max)
        std = torch.exp(log_std)
        return mean, std
    
    def evaluate_action_log_prob(self, s, a):
        """
        Computes log-probability of action 'a' given state 's', 
        accounting for Tanh squashing.
        """
        mean, std = self.forward(s)
        
        max_action_tensor = torch.tensor(self.max_action_val, device=a.device, dtype=a.dtype)

        # Clamp 'a' to avoid NaN from atanh near boundaries
        clipped_a = torch.clamp(a, -max_action_tensor + 1e-6, max_action_tensor - 1e-6)

        # Inverse of tanh squashing
        unconstrained_a = torch.atanh(clipped_a / max_action_tensor) 
        
        dist_unconstrained = Normal(mean, std.clamp(min=1e-6)) 
        log_prob_unconstrained = dist_unconstrained.log_prob(unconstrained_a).sum(axis=-1)

        # Apply log-determinant Jacobian correction for Tanh
        log_prob_squashed = log_prob_unconstrained - torch.log(max_action_tensor * (1 - (clipped_a / max_action_tensor).pow(2)) + 1e-6).sum(axis=-1)
        
        return log_prob_squashed
        
    def sample(self, state):
        """Samples action using reparameterization trick and computes log-prob."""
        mean, std = self.forward(state)
        dist = Normal(mean, std.clamp(min=1e-6)) 
        action = dist.rsample() 
        
        action_tanh = torch.tanh(action) * self.max_action_val

        log_prob = dist.log_prob(action).sum(axis=-1)
        log_prob -= torch.log(self.max_action_val * (1 - action_tanh.pow(2)) + 1e-6).sum(axis=-1)
        
        return action_tanh, log_prob


weighted_bc_policy = GaussianPolicyNetwork(OBS_DIM, ACTION_DIM).to(device)
print("Weighted BC Policy network defined.")

class BCDataset(Dataset):
    """
    Dataset for BC training containing (obs, act, weight) triplets.
    """
    def __init__(self, eps_dicts, weights):
        self.data = []
        if len(eps_dicts) != len(weights): raise ValueError("Episodes/weights mismatch!")
        for ep_bc, w_bc in zip(eps_dicts, weights):
            obs, acts = ep_bc.get('observations'), ep_bc.get('actions')
            if obs is None or acts is None or len(obs) == 0 or len(acts) == 0: continue
            n_s = min(len(obs), len(acts)) 
            for t in range(n_s): self.data.append((obs[t], acts[t], w_bc))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s_bc, a_bc, w_bc_item = self.data[idx]
        return torch.FloatTensor(s_bc), torch.FloatTensor(a_bc), torch.FloatTensor([w_bc_item])

if not D_main_contaminated_episodes or not trajectory_weights:
    print("ERROR: Skipping Weighted BC training due to missing data or weights.")
else:
    print("Creating Weighted BC Dataset/DataLoader from the CONTAMINATED dataset...");
    weighted_bc_dset = BCDataset(D_main_contaminated_episodes, trajectory_weights)
    if len(weighted_bc_dset) > 0:
        weighted_bc_loader = DataLoader(weighted_bc_dset, batch_size=BC_BATCH_SIZE, shuffle=True)
        print("\n--- Starting Weighted BC Policy Training ---")
        opt_w_bc = optim.Adam(weighted_bc_policy.parameters(), lr=BC_LEARNING_RATE_ALPHA)
        weighted_bc_policy.train()
        for ep_train_wbc in range(BC_EPOCHS):
            ep_l_w_bc = 0
            for s_wbc, a_wbc, w_b_wbc in weighted_bc_loader:
                s_wbc, a_wbc, w_b_wbc = s_wbc.to(device), a_wbc.to(device), w_b_wbc.to(device).squeeze(-1)
                opt_w_bc.zero_grad()
                log_p_wbc = weighted_bc_policy.evaluate_action_log_prob(s_wbc, a_wbc)
                # Weighted negative log-likelihood loss
                loss_p_w_bc = -(w_b_wbc * log_p_wbc).mean()
                loss_p_w_bc.backward()
                opt_w_bc.step()
                ep_l_w_bc += loss_p_w_bc.item()
            avg_l_w_bc = ep_l_w_bc / len(weighted_bc_loader) if len(weighted_bc_loader) > 0 else 0
            if (ep_train_wbc + 1) % 10 == 0:
                print(f"Weighted BC Epoch {ep_train_wbc+1}, Avg Loss: {avg_l_w_bc:.6f}")
        print("--- Weighted BC Policy Training Finished ---")
    else:
        print("Weighted BC Dataset empty. Skipping Training.")

# --- Section 5b: Standard BC Policy Definition and Training (Baseline) ---
print("\n--- Section 5b: Standard BC Policy (Baseline) Training (on SAME Contaminated Data) ---")
standard_bc_policy = GaussianPolicyNetwork(OBS_DIM, ACTION_DIM).to(device)
print("Standard BC Policy network defined.")
# Standard BC uses uniform weights of 1.0
standard_bc_weights = [1.0] * len(D_main_contaminated_episodes)
if not D_main_contaminated_episodes:
    print("ERROR: Skipping Standard BC training.")
else:
    print("Creating Standard BC Dataset/DataLoader from the CONTAMINATED dataset...");
    std_bc_dset = BCDataset(D_main_contaminated_episodes, standard_bc_weights)
    if len(std_bc_dset) > 0:
        std_bc_loader = DataLoader(std_bc_dset, batch_size=BC_BATCH_SIZE, shuffle=True)
        print("\n--- Starting Standard BC Policy Training ---")
        opt_s_bc = optim.Adam(standard_bc_policy.parameters(), lr=BC_LEARNING_RATE_ALPHA)
        standard_bc_policy.train()
        for ep_train_sbc in range(BC_EPOCHS):
            ep_l_s_bc = 0
            for s_sbc, a_sbc, w_b_sbc in std_bc_loader:
                s_sbc, a_sbc, w_b_sbc = s_sbc.to(device), a_sbc.to(device), w_b_sbc.to(device).squeeze(-1)
                opt_s_bc.zero_grad()
                log_p_sbc = standard_bc_policy.evaluate_action_log_prob(s_sbc, a_sbc)
                loss_p_s_bc = -(w_b_sbc * log_p_sbc).mean() 
                loss_p_s_bc.backward()
                opt_s_bc.step()
                ep_l_s_bc += loss_p_s_bc.item()
            avg_l_s_bc = ep_l_s_bc / len(std_bc_loader) if len(std_bc_loader) > 0 else 0
            if (ep_train_sbc + 1) % 10 == 0:
                print(f"Standard BC Epoch {ep_train_sbc+1}, Avg Loss: {avg_l_s_bc:.6f}")
        print("--- Standard BC Policy Training Finished ---")
    else:
        print("Standard BC Dataset empty. Skipping Training.")

# --- Replay Buffer for Offline RL Algorithms (BEAR, BCQ, BRAC) ---
class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling transitions.
    """
    def __init__(self, obs_dim, act_dim, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, obs_dim))
        self.action = np.zeros((max_size, act_dim))
        self.next_state = np.zeros((max_size, obs_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
class Critic(nn.Module):
    """
    Twin Q-network (Critic) architecture.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Q1 architecture
        self.q1_l1 = nn.Linear(obs_dim + act_dim, POLICY_HIDDEN_DIM)
        self.q1_l2 = nn.Linear(POLICY_HIDDEN_DIM, POLICY_HIDDEN_DIM)
        self.q1_l3 = nn.Linear(POLICY_HIDDEN_DIM, 1)
        # Q2 architecture
        self.q2_l1 = nn.Linear(obs_dim + act_dim, POLICY_HIDDEN_DIM)
        self.q2_l2 = nn.Linear(POLICY_HIDDEN_DIM, POLICY_HIDDEN_DIM)
        self.q2_l3 = nn.Linear(POLICY_HIDDEN_DIM, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.relu(self.q1_l1(sa)); q1 = self.relu(self.q1_l2(q1)); q1 = self.q1_l3(q1)
        q2 = self.relu(self.q2_l1(sa)); q2 = self.relu(self.q2_l2(q2)); q2 = self.q2_l3(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.relu(self.q1_l1(sa)); q1 = self.relu(self.q1_l2(q1)); q1 = self.q1_l3(q1)
        return q1


# --- Section 5c: BCQ Model Definition and Training ---
print("\n--- Section 5c: BCQ Policy Training (on SAME Contaminated Data) ---")

class VAE(nn.Module):
    """
    VAE for modeling the action distribution in BCQ.
    """
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim) 
        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)
        self.max_action, self.latent_dim = max_action, latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        mean, log_std = self.mean(z), self.log_std(z).clamp(-4, 15) 
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std) 
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

class BCQ(object):
    """
    Batch-Constrained Deep Q-learning (BCQ).
    """
    def __init__(self, s_dim, a_dim, max_a, latent_dim, phi):
        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=BCQ_LR)
        self.vae = VAE(s_dim, a_dim, latent_dim, max_a).to(device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=BCQ_LR)
        self.max_action, self.phi = max_a, phi

    def select_action(self, state):
        """
        Inference: sample from VAE, perturb, and pick best Q1.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(BCQ_N_ACTION_SAMPLES, 1).to(device)
            self.vae.eval() 
            action = self.vae.decode(state)
            self.vae.train() 
            # Perturb VAE actions
            action = action + self.phi * self.max_action * torch.randn_like(action)
            q1 = self.critic.Q1(state, action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done = replay_buffer.sample(BCQ_BATCH_SIZE)
        
        # VAE Update
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss 
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Critic Update
        with torch.no_grad():
            next_state_rpt = torch.repeat_interleave(next_state, BCQ_N_ACTION_SAMPLES, 0)
            next_action_rpt = self.vae.decode(next_state_rpt)
            next_action_rpt = next_action_rpt + self.phi * self.max_action * torch.randn_like(next_action_rpt)
            
            target_Q1, target_Q2 = self.critic_target(next_state_rpt, next_action_rpt)
            target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
            target_Q = target_Q.reshape(BCQ_BATCH_SIZE, -1).max(1).values.reshape(-1, 1) 
            target_Q = reward + not_done * BCQ_DISCOUNT_GAMMA * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Target soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(BCQ_TAU * param.data + (1 - BCQ_TAU) * target_param.data)

latent_dim = ACTION_DIM * BCQ_LATENT_DIM_MULTIPLIER
bcq_policy_agent = BCQ(OBS_DIM, ACTION_DIM, MAX_ACTION, latent_dim, BCQ_PERTURBATION_PHI)
print("BCQ Policy agent defined.")
if not D_main_contaminated_episodes: print("ERROR: Skipping BCQ training.")
else:
    print("Populating Replay Buffer from the CONTAMINATED dataset for BCQ...");
    bcq_replay_buffer = ReplayBuffer(OBS_DIM, ACTION_DIM)
    for ep_dict in D_main_contaminated_episodes:
        transitions = extract_transitions_for_replay_buffer(ep_dict)
        for trans in transitions:
            bcq_replay_buffer.add(trans['observations'], trans['actions'], trans['next_observations'], trans['rewards'], trans['terminals'])
            
    print(f"\n--- Starting BCQ Policy Training for {BCQ_TRAINING_STEPS} steps ---")
    bcq_policy_agent.critic.train() 
    bcq_policy_agent.vae.train() 
    for t in range(int(BCQ_TRAINING_STEPS)):
        bcq_policy_agent.train(bcq_replay_buffer)
        if (t + 1) % 50000 == 0:
            print(f"BCQ Step {t+1}")
            
    print("--- BCQ Policy Training Finished ---")


# --- Section 5d: BRAC Model Definition and Training ---
print("\n--- Section 5d: BRAC Policy Training (on SAME Contaminated Data) ---")

class BRAC(object):
    """
    Behavior Regularized Actor Critic (BRAC).
    """
    def __init__(self, s_dim, a_dim, max_a, alpha):
        self.actor = GaussianPolicyNetwork(s_dim, a_dim, log_max=LOG_STD_MAX, log_min=LOG_STD_MIN).to(device)
        self.actor.max_action_val = max_a 
        self.actor_target = GaussianPolicyNetwork(s_dim, a_dim, log_max=LOG_STD_MAX, log_min=LOG_STD_MIN).to(device)
        self.actor_target.max_action_val = max_a 
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=BRAC_LR)

        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=BRAC_LR)

        self.behavior_policy = GaussianPolicyNetwork(s_dim, a_dim, log_max=LOG_STD_MAX, log_min=LOG_STD_MIN).to(device)
        self.behavior_policy.max_action_val = max_a 
        self.behavior_optimizer = optim.Adam(self.behavior_policy.parameters(), lr=BRAC_LR)

        self.max_action = max_a
        self.alpha = alpha 

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            self.actor.eval() 
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action 
            self.actor.train() 
            return np.clip(action.cpu().data.numpy().flatten(), -self.max_action, self.max_action)

    def pretrain_behavior_policy(self, replay_buffer):
        """Pre-trains behavior policy using BC."""
        print("\n--- Starting BRAC Behavior Policy Pre-Training ---")
        all_s = torch.FloatTensor(replay_buffer.state[:replay_buffer.size]).to(device)
        all_a = torch.FloatTensor(replay_buffer.action[:replay_buffer.size]).to(device)
        behavior_dataloader = DataLoader(torch.utils.data.TensorDataset(all_s, all_a), batch_size=BRAC_BATCH_SIZE, shuffle=True)

        self.behavior_policy.train()
        for epoch in range(BRAC_BEHAVIOR_PRETRAIN_EPOCHS):
            total_loss = 0
            for state, action in behavior_dataloader:
                self.behavior_optimizer.zero_grad()
                log_prob = self.behavior_policy.evaluate_action_log_prob(state, action)
                loss = -log_prob.mean()
                loss.backward()
                self.behavior_optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Behavior Pre-Train Epoch {epoch+1}, Avg Loss: {total_loss / len(behavior_dataloader):.4f}")
        self.behavior_policy.eval() 
        print("--- BRAC Behavior Policy Pre-Training Finished ---")

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done = replay_buffer.sample(BRAC_BATCH_SIZE)

        # --- Critic Update ---
        with torch.no_grad():
            next_action_pi, _ = self.actor_target.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action_pi)
            target_Q = torch.min(target_Q1, target_Q2) 
            target_Q = reward + (not_done * BRAC_DISCOUNT_GAMMA * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        pi_action, log_prob_pi = self.actor.sample(state) 
        q1_pi, q2_pi = self.critic(state, pi_action)
        q_pi = torch.min(q1_pi, q2_pi) 

        # KL Regularization
        mean_actor, std_actor = self.actor(state)
        self.behavior_policy.eval() 
        mean_behavior, std_behavior = self.behavior_policy(state)      
        
        actor_dist = Normal(mean_actor, std_actor.clamp(min=1e-6))
        behavior_dist = Normal(mean_behavior, std_behavior.clamp(min=1e-6))
        
        kl_div = torch.distributions.kl.kl_divergence(actor_dist, behavior_dist).sum(dim=-1)

        actor_loss = (self.alpha * kl_div - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Target Soft Updates ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(BRAC_TAU * param.data + (1 - BRAC_TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(BRAC_TAU * param.data + (1 - BRAC_TAU) * target_param.data)


brac_policy_agent = BRAC(OBS_DIM, ACTION_DIM, MAX_ACTION, BRAC_ALPHA)
print("BRAC Policy agent defined.")
if not D_main_contaminated_episodes: print("ERROR: Skipping BRAC training.")
else:
    print("Populating Replay Buffer from the CONTAMINATED dataset for BRAC...");
    brac_replay_buffer = ReplayBuffer(OBS_DIM, ACTION_DIM)
    for ep_dict in D_main_contaminated_episodes:
        transitions = extract_transitions_for_replay_buffer(ep_dict)
        for trans in transitions:
            brac_replay_buffer.add(trans['observations'], trans['actions'], trans['next_observations'], trans['rewards'], trans['terminals'])

    brac_policy_agent.pretrain_behavior_policy(brac_replay_buffer)

    print(f"\n--- Starting BRAC Policy Training for {BRAC_TRAINING_STEPS} steps ---")
    brac_policy_agent.actor.train() 
    brac_policy_agent.critic.train() 
    for t in range(int(BRAC_TRAINING_STEPS)):
        brac_policy_agent.train(brac_replay_buffer)
        if (t + 1) % 50000 == 0:
            print(f"BRAC Step {t+1}")

    print("--- BRAC Policy Training Finished ---")


# --- Section 6: Policy Evaluation and Comparison ---
print("\n--- Section 6: Policy Evaluation and Comparison (in Clean Environment) ---")

def evaluate_policy_with_minari_env(policy_agent, model_name_eval, minari_d_id_eval, num_eps_eval=NUM_EVAL_EPISODES, seed_val_eval=EVAL_SEED, max_steps_override_eval=MAX_EVAL_STEPS_PER_EPISODE_CONFIG):
    """
    Evaluates policy in Minari environment and returns stats.
    """
    print(f"\nEvaluating {model_name_eval}...")
    try:
        m_dset_eval = minari.load_dataset(minari_d_id_eval)
        eval_env_instance = m_dset_eval.recover_environment()
        if eval_env_instance is None:
            print(f"ERROR: Could not recover env from {minari_d_id_eval}."); return None, None, None, None
    except Exception as e_eval:
        print(f"ERROR loading Minari dataset/env '{minari_d_id_eval}': {e_eval}"); return None, None, None, None
            
    all_rewards_eval_list = []
    
    if hasattr(policy_agent, 'actor'): 
        policy_agent.actor.eval()
    elif hasattr(policy_agent, 'vae'): 
        policy_agent.vae.eval()
        policy_agent.critic.eval()
    elif isinstance(policy_agent, nn.Module): 
        policy_agent.eval()

    for ep_n_eval in range(num_eps_eval):
        try:
            obs_eval, info_eval = eval_env_instance.reset(seed=seed_val_eval + ep_n_eval)
        except Exception as e_reset:
            print(f"ERROR resetting env for episode {ep_n_eval+1}: {e_reset}"); continue
            
        done_eval, truncated_eval, ep_r_eval, steps_eval = False, False, 0.0, 0
        while not (done_eval or truncated_eval):
            with torch.no_grad():
                action_eval = policy_agent.select_action(np.array(obs_eval))
            
            # Clip actions to bounds
            action_eval = np.clip(action_eval, eval_env_instance.action_space.low, eval_env_instance.action_space.high)
            
            try:
                next_obs_eval, reward_eval, term_eval, trunc_eval, info_eval_step = eval_env_instance.step(action_eval)
                done_eval, truncated_eval = term_eval, trunc_eval
            except Exception as e_step:
                print(f"ERROR during env.step() in episode {ep_n_eval+1}: {e_step}"); break
                
            ep_r_eval += reward_eval
            obs_eval = next_obs_eval
            steps_eval += 1
            if steps_eval >= max_steps_override_eval:
                truncated_eval = True
                
        all_rewards_eval_list.append(ep_r_eval)
        
    eval_env_instance.close()

    # Restore training mode
    if hasattr(policy_agent, 'actor'):
        policy_agent.actor.train()
    elif hasattr(policy_agent, 'vae'):
        policy_agent.vae.train()
        policy_agent.critic.train()
    elif isinstance(policy_agent, nn.Module):
        policy_agent.train()

    if not all_rewards_eval_list:
        print("No episodes completed in evaluation for " + model_name_eval)
        return None, None, None, None
        
    mean_r_eval, std_r_eval = np.mean(all_rewards_eval_list), np.std(all_rewards_eval_list)
    print(f"{model_name_eval} Summary: Mean Reward: {mean_r_eval:.2f} +/- {std_r_eval:.2f}")
    
    return mean_r_eval, std_r_eval, all_rewards_eval_list, policy_agent.behavior_policy if model_name_eval == "BRAC" else None

def simple_policy_select_action(policy_model, state, max_action_val):
    """Helper for BC policies to select deterministic action."""
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    policy_model.eval() 
    with torch.no_grad():
        mean_action, _ = policy_model(state_tensor)
        action = torch.tanh(mean_action).squeeze(0).cpu().numpy() * max_action_val
    policy_model.train() 
    return action

# Assign select_action methods to the BC policies for evaluation consistency
weighted_bc_policy.select_action = lambda s: simple_policy_select_action(weighted_bc_policy, s, MAX_ACTION)
standard_bc_policy.select_action = lambda s: simple_policy_select_action(standard_bc_policy, s, MAX_ACTION)


# Dictionary to store results
all_evaluation_metrics = {
    'Average Return': {},
}

# Define result directory
RESULTS_DIR = "robustness_test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_plot(plt_obj, filename, save_dir=RESULTS_DIR):
    filepath = os.path.join(save_dir, filename)
    plt_obj.savefig(filepath)
    print(f"Saved plot: {filepath}")
    plt_obj.close() 


# --- Evaluate all policies ---
mean_r_wbc, std_r_wbc, rewards_wbc, _ = evaluate_policy_with_minari_env(weighted_bc_policy, "Density-Ratio Weighted BC", MINARI_DATASET_ID)
if mean_r_wbc is not None: 
    all_evaluation_metrics['Average Return']["Density-Ratio Weighted BC"] = {'mean': mean_r_wbc, 'std': std_r_wbc}

mean_r_sbc, std_r_sbc, rewards_sbc, _ = evaluate_policy_with_minari_env(standard_bc_policy, "Standard BC", MINARI_DATASET_ID)
if mean_r_sbc is not None: 
    all_evaluation_metrics['Average Return']["Standard BC"] = {'mean': mean_r_sbc, 'std': std_r_sbc}

mean_r_bcq, std_r_bcq, rewards_bcq, _ = evaluate_policy_with_minari_env(bcq_policy_agent, "BCQ", MINARI_DATASET_ID)
if mean_r_bcq is not None: 
    all_evaluation_metrics['Average Return']["BCQ"] = {'mean': mean_r_bcq, 'std': std_r_bcq}

mean_r_brac, std_r_brac, rewards_brac, brac_behavior_policy = evaluate_policy_with_minari_env(brac_policy_agent, "BRAC", MINARI_DATASET_ID)
if mean_r_brac is not None: 
    all_evaluation_metrics['Average Return']["BRAC"] = {'mean': mean_r_brac, 'std': std_r_brac}


# --- Plot Average Return Comparison ---
if all_evaluation_metrics['Average Return']:
    labels = list(all_evaluation_metrics['Average Return'].keys())
    mean_rewards = [all_evaluation_metrics['Average Return'][label]['mean'] for label in labels]
    stds_for_plot = [all_evaluation_metrics['Average Return'][label]['std'] for label in labels]
    
    if labels: 
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(labels))
        width = 0.6
        colors = ['dodgerblue', 'salmon', 'lightgreen', 'purple', 'orange'][:len(labels)] 
        rects = ax.bar(x, mean_rewards, width, yerr=stds_for_plot, capsize=5, color=colors)
        
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title(f'Average Return Comparison on {MINARI_DATASET_ID}\n(Trained on Data with {PERCENTAGE_TO_POISON_ACTION*100}% Action Poisoning)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.bar_label(rects, padding=3, fmt='%.2f')
        
        fig.tight_layout()
        plot_filename = f"average_return_comparison_{MINARI_DATASET_ID.replace('/', '_')}_poison_{PERCENTAGE_TO_POISON_ACTION*100}pct_seed{GLOBAL_SEED}.png"
        save_plot(plt, plot_filename, RESULTS_DIR)
    else:
        print("\nNo average return results to plot comparison for.")
else:
    print("\nNo average return results to plot comparison for.")

print(f"\n--- All results and graphs saved to: {RESULTS_DIR} ---")
print("\n--- Full script execution finished. ---")