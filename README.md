# Robust_Weighted_BC_Algorithm

Core Algorithm

<img width="2816" height="1536" alt="flowchart_robust_dwbc" src="https://github.com/user-attachments/assets/3ac35f82-f397-4afd-b23f-d48146a402f5" />

# Offline RL Robustness under Action Poisoning with Data Integrity Verification

This repository implements a comparative study of Offline Reinforcement Learning (RL) algorithms in the presence of data contamination. It focuses on evaluating how well different policy learning methods recover optimal behavior when a significant portion of the training dataset has been poisoned with action noise.

A key feature of this implementation is a **Reference Set Integrity Check** system, which uses cryptographic hashing (including Merkle Trees) to ensure the "trusted" subset of data used for importance weighting remains tamper-proof.

## Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)

## Overview

Offline RL aims to learn policies from static datasets without interacting with the environment. However, real-world datasets often contain noise or malicious perturbations. This codebase:

1. Loads a standard Minari dataset (e.g., mujoco/halfcheetah/medium-v0).
2. Splits the data into a **Trusted Reference Set** (clean) and a **Training Pool**.
3. Injects **Action Poisoning** (Gaussian noise) into a configurable percentage of the training pool.
4. Trains a discriminator to estimate density ratios between clean and contaminated data.
5. Trains four distinct agents on the contaminated data.
6. Evaluates the agents in a clean environment to measure robustness.

## Algorithms Implemented

The following algorithms are implemented in PyTorch:

1. **Standard Behavioral Cloning (BC):** A baseline that treats all data equally.
2. **Density-Ratio Weighted BC:** Uses a discriminator to assign lower weights to trajectories that likely contain poisoning, prioritizing the trusted reference distribution.
3. **Batch-Constrained Deep Q-Learning (BCQ):** Uses a VAE to model the behavior policy and perturbs actions within the latent space to avoid out-of-distribution actions.
4. **Behavior Regularized Actor Critic (BRAC):** Regularizes the learned policy towards a pre-trained behavior policy using KL-divergence.

## Key Features

### Data Integrity Verification

To ensure the trusted reference set (used to train the discriminator) is not tampered with during the pipeline, the system implements three integrity check modes:

- **Full Set Hashing:** SHA256 hash of the entire serialized reference dataset.
- **Individual Episode Hashing:** Verify specific episodes.
- **Merkle Tree:** Allows for granular tamper detection.

### Configurable Poisoning

You can adjust the intensity of the robustness test by modifying:

- `PERCENTAGE_TO_POISON_ACTION`: The fraction of the non-reference dataset to corrupt.
- `ACTION_NOISE_LEVEL`: The magnitude of the noise injected into actions.

## Installation

### Prerequisites

- Python 3.8+
- [Minari](https://minari.farama.org/)
- [MuJoCo](https://mujoco.org/) (Required for the Gymnasium environments)

### Setup

1. Clone the repository:

git clone https://github.com/yourusername/offline-rl-poisoning.git
cd offline-rl-poisoning


2. Install the required dependencies:

pip install -r requirements.txt


Ensure your `requirements.txt` includes:

numpy
matplotlib
torch
minari
gymnasium[mujoco]


## Usage

To run the full pipeline (Data Loading -> Integrity Check -> Training -> Evaluation), simply execute the main script:

python action_poisoning_with_integrity.py


### Expected Output

1. **Console:** Detailed logs of training progress (Loss, Accuracy) and evaluation rewards.
2. **Files:**
   - `reference_set_integrity.json`: Stores the cryptographic hashes of the reference set.
   - `robustness_test_results/`: Contains generated comparison plots (PNG) showing the Mean Episode Reward for each algorithm.

## Configuration

You can modify the hyperparameters at the top of `action_poisoning_with_integrity.py`. Key parameters include:

| Parameter                  | Default                | Description                                                      |
|----------------------------|------------------------|------------------------------------------------------------------|
| MINARI_DATASET_ID           | halfcheetah/medium-v0  | The Minari dataset to load.                                      |
| D_REF_PERCENTAGE            | 0.20                   | Percentage of data reserved as "Trusted".                       |
| PERCENTAGE_TO_POISON_ACTION | 0.0                    | Percentage of training data to poison (e.g., 0.5 for 50%).      |
| INTEGRITY_CHECK_MODE        | 'full_set'             | Integrity mode: 'full_set', 'individual_episodes', or 'merkle_tree'. |
| NUM_EVAL_EPISODES           | 50                     | Number of episodes for final evaluation.                        |

## Results

Upon completion, the script generates a bar chart comparing the performance of all four algorithms.

- **Weighted BC** typically outperforms Standard BC in high-poisoning scenarios by effectively filtering out noisy trajectories.
- **BCQ** and **BRAC** provide strong baselines for offline RL but may have varying sensitivity to action noise depending on the hyperparameters (phi for BCQ, alpha for BRAC).

