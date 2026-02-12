# Hybrid Neural Network Adversarial Defense Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive framework for evaluating adversarial robustness of Hybrid Neural Networks (HNN) combining classical Convolutional Neural Networks with quantum circuit layers. This implementation supports multiple datasets, compound adversarial attacks, and various defense mechanisms.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Code Structure](#code-structure)
- [Datasets](#datasets)
- [Adversarial Attacks](#adversarial-attacks)
- [Defense Mechanisms](#defense-mechanisms)
  - [Randomization Defense (4 Techniques)](#1-randomization-defense)
  - [Input Transformation Defense (5 Techniques)](#2-input-transformation-defense)
  - [Adversarial Training](#3-adversarial-training)
  - [Defense Comparison Matrix](#defense-comparison-matrix)
- [Installation](#installation)
- [Usage](#usage)
  - [Defense Configuration Examples](#defense-configuration-examples)
  - [Running Experiments](#running-experiments)
- [Key Features](#key-features)
- [Technical Details](#technical-details)
  - [Performance Characteristics](#performance-characteristics)
  - [Defense Implementation Details](#defense-implementation-details)
- [Configuration](#configuration)
  - [Global Configuration Variables](#global-configuration-variables)
  - [Complete Configuration Reference](#complete-configuration-reference)
- [Output](#output)

## Overview

This framework implements and evaluates hybrid quantum-classical neural networks for adversarial robustness. The system tests **three defense categories** with **10 total defense configurations** against compound adversarial attacks across multiple image classification datasets.

### Research Goals

1. **Evaluate HNN robustness** against state-of-the-art adversarial attacks
2. **Compare defense mechanisms** across three categories:
   - **Randomization** (4 techniques): Random Resizing, Random Cropping, Random Rotation, Combined Randomization
   - **Input Transformation** (5 techniques): Image Quilting, JPEG Compression, Bit Depth Reduction, Gaussian Noise, Combined Input Transformation
   - **Adversarial Training** (1 approach): Training on mixed clean and adversarial data
3. **Analyze compound attacks** (FGSM+PGD, FGSM+CW, CW+PGD combinations)
4. **Benchmark performance** across diverse datasets with varying complexity

### Defense Configuration Summary

**10 Total Defense Configurations:**
- 1 × Adversarial Training
- 4 × Randomization techniques
- 5 × Input Transformation techniques

**3 Attack Types:**
- FGSM + PGD
- FGSM + CW
- CW + PGD

**4 Datasets:**
- MNIST (10-class digits)
- EMNIST (10-class extended digits)
- TrafficSigns (4-class geometric shapes)
- TinyImageNet (5-class natural objects)

## Architecture

### Hybrid Neural Network (HNN)

The HNN combines classical and quantum computing paradigms:

```
INPUT → Classical CNN → QUANTUM CIRCUIT → HYBRID OUTPUT
         (Feature extraction)  (Quantum processing)  (50/50 combination)
```

#### Classical Component (CNN)
- **Convolutional layers**: 3 layers with batch normalization
- **Fully connected layers**: 2 layers with dropout regularization
- **Dataset-specific architectures**:
  - MNIST/EMNIST: 16→32→64 channels
  - TrafficSigns: 16→32→64 channels, FC 4096→128
  - TinyImageNet: 32→64→128 channels, FC 8192→256

#### Quantum Component
- **Quantum circuit**: Parameterized with learnable θ and φ parameters
- **Qubit count**: Dynamically sized based on output dimension
- **Quantum operations**: Rotation gates (Ry) and controlled gates (CNOT)
- **Simulation**: Cirq-based quantum circuit simulation
- **Parameters**:
  - MNIST/EMNIST: 1 parameter each (θ, φ)
  - TinyImageNet: 128 parameters each (θ, φ)

#### Hybrid Combination
```python
hybrid_output = 0.5 × classical_output + 0.5 × quantum_output
final_output = log_softmax(hybrid_output)
```

The quantum and classical outputs are linearly combined with equal weighting (α=0.5, β=0.5), allowing both components to contribute equally to the final prediction.

## Code Structure

```
.
├── CNN Class (Lines ~310-550)
│   ├── __init__(): Dataset-specific architecture initialization
│   ├── forward(): Forward pass with conditional branching
│   └── Architecture variants: MNIST, EMNIST, TrafficSigns, TinyImageNet
│
├── HNN Class (Lines ~573-605)
│   ├── __init__(): Wraps classical model with quantum parameters
│   ├── forward(): Calls hybrid_forward() for quantum integration
│   └── Quantum parameter initialization (dataset-dependent)
│
├── Quantum Circuit Functions (Lines ~150-295)
│   ├── create_quantum_circuit(): Builds parameterized quantum circuit
│   ├── hybrid_forward(): Integrates classical CNN with quantum circuit
│   └── Cirq simulation and state vector processing
│
├── Data Loading and Preprocessing (Lines ~616-1175)
│   ├── Dataset downloads and setup (MNIST, EMNIST, TrafficSigns, TinyImageNet)
│   ├── Data augmentation and normalization
│   └── Train/validation class synchronization (TinyImageNet)
│
├── Dataset Filtering (Lines ~1176-1455)
│   ├── filtered_dataset(): Class balancing and selection
│   ├── filtered_dataset_EMNIST(): EMNIST-specific filtering
│   ├── RemapLabelsWrapper: Label remapping for filtered classes
│   └── Support for multi-class classification with balanced samples
│
├── Training Pipeline (Lines ~1687-2050)
│   ├── train_model(): Main training loop with AMP support
│   ├── Learning rate scheduling (ReduceLROnPlateau)
│   ├── Early stopping and model checkpointing
│   └── Progress tracking with tqdm
│
├── Adversarial Attack Generation (Lines ~2051-2642)
│   ├── Compound attack combinations (FGSM+PGD, FGSM+CW, CW+PGD)
│   ├── generate_adversarial_dataset(): Batch adversarial generation
│   ├── Attack configuration (epsilon=0.03, alpha=0.01, iterations=10)
│   └── GPU-accelerated attack computation
│
├── Defense Mechanisms (Lines ~2643-2841)
│   ├── Randomization Defense (4 techniques)
│   │   ├── random_resizing: Random scaling transformations
│   │   ├── random_cropping: Random crop and resize
│   │   ├── random_rotation: Random angle rotations
│   │   └── combined_randomization: Random selection of above
│   ├── Input Transformation Defense (5 techniques)
│   │   ├── image_quilting: Patch-based reconstruction from clean data
│   │   ├── jpeg_compression: Lossy compression to remove noise
│   │   ├── bit_depth_reduction: Color quantization (8-bit to 4-bit)
│   │   ├── gaussian_noise_defense: Add random Gaussian noise
│   │   └── combined_input_transformation: Random selection of above
│   └── Adversarial Training: Retrain on mixed clean/adversarial data
│       └── Defense-specific dataset creation with 50/50 ratio
│
├── Evaluation and Metrics (Lines ~2900-4100)
│   ├── Model evaluation on clean and adversarial data
│   ├── Misclassification analysis and visualization
│   ├── Robustness metrics calculation
│   └── Defense effectiveness scoring
│
└── Visualization and Reporting (Lines ~4101-4467)
    ├── print_summary_images(): Sample visualization with predictions
    ├── Confusion matrix generation
    ├── Tabular results formatting
    └── Comprehensive metrics reporting
```

### Critical Code Sections

#### 1. Train/Val Class Synchronization (Lines ~1120-1145)
Ensures TinyImageNet train and validation datasets use identical class-to-index mappings to prevent train-test class mismatch.

```python
# Synchronize train and val class orderings
train_class_to_idx = train_set.class_to_idx
for path, old_idx in test_set.samples:
    class_name = os.path.basename(os.path.dirname(path))
    new_idx = train_class_to_idx[class_name]
    # Remap test samples to use train's class indices
```

#### 2. Label Map Synchronization (Lines ~1901, 2885)
Updates label mappings after class filtering to ensure display names match actual class indices.

```python
# After filtering to selected classes [48, 57, 66, 75, 84] → [0, 1, 2, 3, 4]
for new_idx, orig_idx in enumerate(tinyimagenet_selected_classes):
    label_map[new_idx] = idx_to_class.get(orig_idx)
idx_to_class = label_map  # Critical synchronization
```

#### 3. Conditional Architecture (Lines ~327-400)
Dataset-specific CNN architectures with conditional branching.

```python
if dataset_name == 'TinyImageNet':
    # Wider architecture for complex images
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.fc1 = nn.Linear(128 * 8 * 8, 256)
    self.dropout = nn.Dropout(p=0.6)
```

#### 4. Hybrid Forward Pass (Lines ~251-290)
Integration of classical CNN with quantum circuit.

```python
def hybrid_forward(input_data, classical_model, theta, phi, device, output_dim):
    classical_output = classical_model(input_data)
    quantum_circuit = create_quantum_circuit(theta, phi, output_dim)
    result = simulator.simulate(quantum_circuit)
    quantum_output = result.final_state_vector
    # Linear combination
    hybrid_output = 0.5 * classical_output + 0.5 * quantum_output
    return log_softmax(hybrid_output)
```

## Datasets

### MNIST
- **Type**: Handwritten digits (0-9)
- **Size**: 60,000 training, 10,000 test
- **Format**: 28×28 grayscale
- **Used**: 2,000 training (200/class), 500 test (50/class)
- **Preprocessing**: Normalization (mean=0.1307, std=0.3081)

### EMNIST (Digits)
- **Type**: Extended MNIST digits (0-9)
- **Size**: 240,000 training, 40,000 test
- **Format**: 28×28 grayscale
- **Used**: 2,000 training (200/class), 500 test (50/class)
- **Preprocessing**: Normalization (mean=0.1751, std=0.3332)

### TrafficSigns
- **Type**: German Traffic Sign Recognition Benchmark (subset)
- **Classes**: 4 classes (crosswalk, speedlimit, stop, trafficlight)
- **Format**: 32×32 RGB
- **Preprocessing**: Resize to 32×32, normalization
- **Augmentation**: Random horizontal flip, rotation, color jitter

### TinyImageNet
- **Type**: ImageNet subset with 200 classes
- **Selected Classes**: 5 classes (keyboard, lawn mower, remote control, stopwatch, water tower)
- **Size**: 500 training, 50 validation per class (available)
- **Used**: 500 training, 50 test per class (2,500 total training, 250 total test)
- **Format**: 64×64 RGB
- **Preprocessing**: Resize, normalization (ImageNet stats)
- **Augmentation**: Aggressive augmentation (flip, rotation, crop, color jitter, affine)
- **Class Selection**: Indices [48, 57, 66, 75, 84] to avoid animal classes
- **Challenge**: Natural objects with complex features and confusable classes

## Adversarial Attacks

### Implemented Attacks

#### 1. Fast Gradient Sign Method (FGSM)
- **Type**: Single-step gradient-based
- **Epsilon**: 0.03
- **Formula**: `x' = x + ε × sign(∇_x L(θ, x, y))`

#### 2. Projected Gradient Descent (PGD)
- **Type**: Multi-step iterative
- **Epsilon**: 0.03
- **Alpha**: 0.01 (step size)
- **Iterations**: 10
- **Formula**: Iterative FGSM with projection

#### 3. Carlini-Wagner (CW)
- **Type**: Optimization-based
- **C value**: 0.1
- **Iterations**: 10
- **Objective**: Minimize perturbation while maximizing misclassification

### Compound Attacks

Three compound attack strategies are evaluated:

1. **FGSM + PGD**: Fast initial perturbation followed by iterative refinement
2. **FGSM + CW**: Fast perturbation followed by optimization-based attack
3. **CW + PGD**: Optimization-based perturbation followed by iterative refinement

Each compound attack applies both methods sequentially to the same image, creating stronger adversarial examples than single attacks.

## Defense Mechanisms

The framework implements **three main defense categories**, each with multiple **sub-techniques** that can be individually evaluated or combined.

### 1. Randomization Defense

**Configuration**: `randomization_defense = "technique_name"`

All randomization techniques are **test-time only** - no model retraining required. They apply random transformations to input images during inference to disrupt adversarial perturbations.

#### 1.1 Random Resizing (`random_resizing`)
- **Method**: Randomly resize image to different scales, then resize back to original dimensions
- **Scale Range**: 50-150% of original size
- **Rationale**: Breaks attack assumptions about exact pixel positions and image scale
- **Advantage**: Smoothly preserves features while disrupting perturbations
- **Overhead**: ~10% (fast resize operations)

#### 1.2 Random Cropping (`random_cropping`)
- **Method**: Randomly crop image to smaller size, then resize back to original dimensions
- **Crop Range**: 80-100% of original size
- **Rationale**: Removes edge perturbations, shifts adversarial patterns
- **Advantage**: Simple, fast, removes peripheral attacks
- **Overhead**: ~5% (crop + resize)

#### 1.3 Random Rotation (`random_rotation`)
- **Method**: Randomly rotate image by small angles, then crop/resize back
- **Angle Range**: ±15 degrees
- **Rationale**: Breaks spatial assumptions, redistributes perturbations
- **Advantage**: Effective against spatially-localized attacks
- **Overhead**: ~15% (rotation + interpolation)

#### 1.4 Combined Randomization (`combined_randomization`)
- **Method**: Randomly select and apply one of the above techniques per image
- **Selection**: Uniform random choice from {resize, crop, rotate}
- **Rationale**: Unpredictable defense, harder for adaptive attacks
- **Advantage**: Best of all techniques, no single point of failure
- **Overhead**: ~10-15% (varies by selected technique)

---

### 2. Input Transformation Defense

**Configuration**: `input_transformation = "technique_name"`

All input transformation techniques are **test-time only** - no model retraining. They modify the input image to remove or neutralize adversarial perturbations before classification.

#### 2.1 Image Quilting (`image_quilting`)
- **Method**: Reconstruct image using patches from clean training data
- **Patch Size**: 8×8 pixels (adaptive)
- **Matching**: Find best-matching clean patches for each region
- **Rationale**: Replaces adversarial pixels with clean, natural image patches
- **Advantage**: Removes local perturbations, maintains global structure
- **Overhead**: ~50% (patch matching computation)
- **Challenge**: May create texture artifacts if patch library is limited

#### 2.2 JPEG Compression (`jpeg_compression`)
- **Method**: Compress image to JPEG format with quality parameter, then decompress
- **Quality**: 75 (adjustable, 0-100 scale)
- **Rationale**: Lossy compression removes high-frequency adversarial noise
- **Advantage**: Simple, fast, removes imperceptible perturbations
- **Overhead**: ~20% (encode + decode)
- **Challenge**: May reduce clean accuracy if compression too aggressive

#### 2.3 Bit Depth Reduction (`bit_depth_reduction`)
- **Method**: Reduce color bit depth (e.g., 8-bit to 4-bit), then restore
- **Reduction**: 8-bit → 4-bit per channel (16 colors per channel)
- **Rationale**: Quantization removes small adversarial perturbations
- **Advantage**: Very fast, deterministic, preserves major features
- **Overhead**: ~5% (quantization operations)
- **Challenge**: May lose subtle color information

#### 2.4 Gaussian Noise Defense (`gaussian_noise_defense`)
- **Method**: Add small Gaussian noise to image
- **Noise Level**: σ = 0.01-0.05 (adaptive)
- **Rationale**: Masks adversarial perturbations with random noise
- **Advantage**: Simple, fast, randomized (hard to adapt to)
- **Overhead**: ~5% (noise generation + addition)
- **Challenge**: May reduce clean accuracy if noise too strong

#### 2.5 Combined Input Transformation (`combined_input_transformation`)
- **Method**: Randomly select and apply one transformation per image
- **Selection**: Uniform random choice from {quilting, JPEG, bit reduction, noise}
- **Rationale**: Unpredictable preprocessing, ensemble defense
- **Advantage**: Combines strengths of all techniques
- **Overhead**: ~5-50% (varies by selected technique)

---

### 3. Adversarial Training

**Configuration**: `defense_type = "adversarial_training"`

Adversarial training is a **training-time defense** that creates an inherently robust model by exposing it to adversarial examples during training.

#### Method
- **Dataset Composition**: 50% clean images + 50% adversarial images
- **Attack Used**: Same attack type as evaluation (FGSM+PGD, FGSM+CW, or CW+PGD)
- **Epochs**: Same as clean training (10 for MNIST/EMNIST, 60 for TrafficSigns, 120 for TinyImageNet)
- **Batch Construction**: Mixed batches with both clean and adversarial examples

#### Process
1. Generate adversarial examples from clean training data
2. Create combined dataset: `combined_data = clean_data ∪ adversarial_data`
3. Train model on combined dataset from scratch
4. Model learns features robust to adversarial perturbations

#### Characteristics
- **Advantage**: Creates inherently robust model, no test-time overhead
- **Disadvantage**: 2× training time, may reduce clean accuracy slightly
- **Effectiveness**: Most robust defense but computationally expensive
- **Generalization**: Best against seen attack types, may struggle with novel attacks

---

### Defense Selection Strategy

The framework allows testing **all combinations**:

```python
# Test all randomization sub-techniques
randomization_techniques = [
    "random_resizing",
    "random_cropping", 
    "random_rotation",
    "combined_randomization"
]

# Test all input transformation sub-techniques
input_transformation_techniques = [
    "image_quilting",
    "jpeg_compression",
    "bit_depth_reduction",
    "gaussian_noise_defense",
    "combined_input_transformation"
]

# Main defense types
defense_types = [
    "adversarial_training",
    "randomization",
    "input_transformation"
]
```

### Defense Comparison Matrix

| Defense Type | Sub-Technique | Retraining? | Test-Time Cost | Feature Preservation | Adaptability |
|--------------|--------------|-------------|----------------|---------------------|--------------|
| **Randomization** | Random Resizing | No | Low (~10%) | High | High |
| **Randomization** | Random Cropping | No | Very Low (~5%) | Medium | High |
| **Randomization** | Random Rotation | No | Medium (~15%) | Medium | High |
| **Randomization** | Combined | No | Low-Medium (~10-15%) | High | Very High |
| **Input Transform** | Image Quilting | No | High (~50%) | Medium | Medium |
| **Input Transform** | JPEG Compression | No | Medium (~20%) | High | Medium |
| **Input Transform** | Bit Depth Reduction | No | Very Low (~5%) | Medium | Low |
| **Input Transform** | Gaussian Noise | No | Very Low (~5%) | Medium | High |
| **Input Transform** | Combined | No | Variable (~5-50%) | Medium | Very High |
| **Adversarial Training** | N/A | Yes | None | High | High |

## Installation

### Requirements

```bash
Python 3.8+
PyTorch 2.0+
torchvision 0.15+
torchattacks 3.5+
cirq 1.6+
cirq-google 1.6+
numpy 2.0+
matplotlib 3.7+
scikit-learn 1.3+
Pillow 10.0+
tqdm 4.66+
tabulate 0.9+
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hnn-adversarial-defense.git
cd hnn-adversarial-defense

# Install dependencies
pip install torch torchvision torchattacks cirq cirq-google
pip install numpy matplotlib scikit-learn Pillow tqdm tabulate

# For Google Colab (recommended)
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')
```

### Google Drive Structure (Recommended)

```
/content/drive/MyDrive/
├── clean_models/              # Saved clean model checkpoints
│   ├── clean_model_MNIST.pth
│   ├── clean_model_EMNIST.pth
│   ├── clean_model_TrafficSigns.pth
│   └── clean_model_TinyImageNet.pth
├── data/                      # Datasets (auto-downloaded)
│   ├── MNIST/
│   ├── EMNIST/
│   ├── TrafficSigns/
│   └── tiny-imagenet-200/
├── results/                   # Experiment results
│   └── [defense]_[attack]_[transform]_[dataset]_[timestamp].txt
└── retrained_model/           # Defense-specific models
    └── final_model.pth
```

## Usage

### Basic Execution

```python
# Global configuration - set at top of file
dataset_name = 'TinyImageNet'                    # Dataset selection
defense_type = 'randomization'                   # Main defense category
attack_combination = 'fgsm_pgd_attack'           # Compound attack type

# Defense sub-technique selection (based on defense_type)
randomization_defense = 'random_resizing'        # Used when defense_type = 'randomization'
input_transformation = 'image_quilting'          # Used when defense_type = 'input_transformation'

# Run the framework
python dissertation_code.py
```

### Defense Configuration Examples

#### Example 1: Randomization with Random Resizing
```python
defense_type = "randomization"
randomization_defense = "random_resizing"
attack_combination = "fgsm_pgd_attack"
dataset_name = "TinyImageNet"
```

#### Example 2: Randomization with Combined Techniques
```python
defense_type = "randomization"
randomization_defense = "combined_randomization"  # Randomly uses resize/crop/rotate
attack_combination = "cw_pgd_attack"
dataset_name = "TinyImageNet"
```

#### Example 3: Input Transformation with JPEG Compression
```python
defense_type = "input_transformation"
input_transformation = "jpeg_compression"
attack_combination = "fgsm_cw_attack"
dataset_name = "TrafficSigns"
```

#### Example 4: Input Transformation with Combined Techniques
```python
defense_type = "input_transformation"
input_transformation = "combined_input_transformation"  # Randomly uses all techniques
attack_combination = "fgsm_pgd_attack"
dataset_name = "TinyImageNet"
```

#### Example 5: Adversarial Training
```python
defense_type = "adversarial_training"
# No sub-technique needed for adversarial training
attack_combination = "fgsm_pgd_attack"
dataset_name = "MNIST"
```

### Configuration Parameters

#### Model Training
```python
num_epochs = 120              # Training epochs
batch_size = 256             # Batch size (TinyImageNet), 64 (others)
learning_rate = 0.001        # Initial learning rate
weight_decay = 3e-4          # L2 regularization (TinyImageNet), 1e-4 (others)
dropout = 0.6                # Dropout rate (TinyImageNet), 0.5 (others)
```

#### Adversarial Attacks
```python
epsilon = 0.03               # Maximum perturbation magnitude
alpha = 0.01                 # PGD step size
num_iter = 10                # PGD/CW iterations
c_value = 0.1                # CW optimization constant
```

#### Defense Settings
```python
# Randomization parameters
resize_range = (0.5, 1.5)    # Random resize scale range (50-150%)
crop_range = (0.8, 1.0)      # Random crop size range (80-100%)
rotation_range = (-15, 15)   # Random rotation angle range (±15°)

# Input Transformation parameters
quilting_patch_size = 8      # Patch size for image quilting
jpeg_quality = 75            # JPEG compression quality (0-100)
bit_depth = 4                # Target bit depth (4-bit = 16 colors/channel)
noise_sigma = 0.05           # Gaussian noise standard deviation

# Adversarial Training
ratio = 0.5                  # Ratio of adversarial examples (50% clean + 50% adversarial)
```

### Complete Configuration Reference

#### All Valid Configurations

| Defense Type | Sub-Technique | Configuration Variable | Valid Values |
|--------------|---------------|----------------------|--------------|
| **Randomization** | Random Resizing | `randomization_defense` | `"random_resizing"` |
| **Randomization** | Random Cropping | `randomization_defense` | `"random_cropping"` |
| **Randomization** | Random Rotation | `randomization_defense` | `"random_rotation"` |
| **Randomization** | Combined Randomization | `randomization_defense` | `"combined_randomization"` |
| **Input Transformation** | Image Quilting | `input_transformation` | `"image_quilting"` |
| **Input Transformation** | JPEG Compression | `input_transformation` | `"jpeg_compression"` |
| **Input Transformation** | Bit Depth Reduction | `input_transformation` | `"bit_depth_reduction"` |
| **Input Transformation** | Gaussian Noise | `input_transformation` | `"gaussian_noise_defense"` |
| **Input Transformation** | Combined Transform | `input_transformation` | `"combined_input_transformation"` |
| **Adversarial Training** | N/A | `defense_type` | `"adversarial_training"` |

### Experiment Matrix

Total possible experiments per dataset:
- **3 Attack Types**: FGSM+PGD, FGSM+CW, CW+PGD
- **10 Defense Configurations**: 1 Adv Training + 4 Randomization + 5 Input Transform
- **4 Datasets**: MNIST, EMNIST, TrafficSigns, TinyImageNet

**Maximum Experiments**: 3 × 10 × 4 = **120 total configurations**

Typical research subset (TinyImageNet only):
- **3 Attack Types**
- **10 Defense Configurations**
- **Total**: 30 experiments

### Complete Experiment Space

```
DATASETS (4)
├── MNIST (10 epochs)
├── EMNIST (10 epochs)
├── TrafficSigns (60 epochs)
└── TinyImageNet (120 epochs)
    │
    ├── ATTACKS (3)
    │   ├── FGSM + PGD
    │   ├── FGSM + CW
    │   └── CW + PGD
    │
    └── DEFENSES (10)
        ├── Adversarial Training (1)
        │   └── Train on 50% clean + 50% adversarial
        │
        ├── Randomization (4)
        │   ├── 1. Random Resizing (50-150% scale)
        │   ├── 2. Random Cropping (80-100% crop)
        │   ├── 3. Random Rotation (±15°)
        │   └── 4. Combined Randomization (random choice)
        │
        └── Input Transformation (5)
            ├── 1. Image Quilting (8×8 patches)
            ├── 2. JPEG Compression (quality=75)
            ├── 3. Bit Depth Reduction (8-bit → 4-bit)
            ├── 4. Gaussian Noise (σ=0.05)
            └── 5. Combined Transform (random choice)
```

**Experiment Naming Convention:**
```
[defense_category]_[attack]_[sub_technique]_[dataset]_[timestamp].txt

Examples:
- randomization_fgsm_pgd_attack_resizing_TinyImageNet_20260212_152500.txt
- input_transformation_cw_pgd_attack_quilting_TinyImageNet_20260212_153000.txt
- adversarial_training_fgsm_cw_attack_TinyImageNet_20260212_154000.txt
```

### Running Experiments

#### Single Configuration
```python
# Example: TinyImageNet with Random Resizing defense against FGSM+PGD
dataset_name = 'TinyImageNet'
defense_type = 'randomization'
randomization_defense = 'random_resizing'
attack_combination = 'fgsm_pgd_attack'
```

#### Testing All Randomization Techniques
```python
dataset_name = 'TinyImageNet'
defense_type = 'randomization'
attacks = ['fgsm_pgd_attack', 'fgsm_cw_attack', 'cw_pgd_attack']

randomization_techniques = [
    'random_resizing',
    'random_cropping',
    'random_rotation',
    'combined_randomization'
]

for technique in randomization_techniques:
    randomization_defense = technique
    for attack in attacks:
        attack_combination = attack
        print(f"Testing {technique} against {attack}")
        # Execute framework
```

#### Testing All Input Transformation Techniques
```python
dataset_name = 'TinyImageNet'
defense_type = 'input_transformation'
attacks = ['fgsm_pgd_attack', 'fgsm_cw_attack', 'cw_pgd_attack']

input_transform_techniques = [
    'image_quilting',
    'jpeg_compression',
    'bit_depth_reduction',
    'gaussian_noise_defense',
    'combined_input_transformation'
]

for technique in input_transform_techniques:
    input_transformation = technique
    for attack in attacks:
        attack_combination = attack
        print(f"Testing {technique} against {attack}")
        # Execute framework
```

#### Complete Experiment Suite
```python
# Test all defenses, all techniques, all attacks on TinyImageNet
dataset_name = 'TinyImageNet'
attacks = ['fgsm_pgd_attack', 'fgsm_cw_attack', 'cw_pgd_attack']

# 1. Adversarial Training
defense_type = 'adversarial_training'
for attack in attacks:
    attack_combination = attack
    # Run experiment

# 2. All Randomization Techniques
defense_type = 'randomization'
for rand_tech in ['random_resizing', 'random_cropping', 'random_rotation', 'combined_randomization']:
    randomization_defense = rand_tech
    for attack in attacks:
        attack_combination = attack
        # Run experiment

# 3. All Input Transformation Techniques
defense_type = 'input_transformation'
for input_tech in ['image_quilting', 'jpeg_compression', 'bit_depth_reduction', 
                    'gaussian_noise_defense', 'combined_input_transformation']:
    input_transformation = input_tech
    for attack in attacks:
        attack_combination = attack
        # Run experiment

# Total experiments: 1 + 4 + 5 = 10 defense configurations × 3 attacks = 30 runs
```

## Key Features

### 1. Automatic Model Management
- **Clean Model Caching**: Saves trained models to Google Drive
- **Automatic Reloading**: Loads existing models if available
- **Checkpointing**: Saves best model during training

### 2. GPU Acceleration
- **CUDA Support**: Automatic GPU detection and usage
- **Mixed Precision**: AMP (Automatic Mixed Precision) for faster training
- **Batch Processing**: Efficient batch-based attack generation

### 3. Comprehensive Evaluation
- **Multiple Metrics**: Accuracy, precision, recall, F1-score
- **Robustness Scoring**: Attack effectiveness and defense recovery rates
- **Misclassification Analysis**: Detailed confusion patterns
- **Visual Outputs**: Sample images with predictions

### 4. Flexible Architecture
- **Dataset-Agnostic**: Supports multiple datasets with automatic configuration
- **Modular Design**: Easy to add new attacks or defenses
- **Configurable Hyperparameters**: All parameters externally configurable

### 5. Reproducibility
- **Fixed Random Seeds**: seed=42 for all random number generators
- **Deterministic Operations**: torch.backends.cudnn.deterministic = True
- **Version Tracking**: Reports library versions in output

## Technical Details

### Memory Management

#### TinyImageNet Optimizations
- **Batch Size**: 256 (larger than other datasets for efficiency)
- **DataLoader Workers**: 2 for parallel data loading
- **Pin Memory**: Enabled for GPU transfer speedup
- **Gradient Accumulation**: Not used (sufficient GPU memory)

#### GPU Memory Usage (Estimated)
- **MNIST/EMNIST**: ~500 MB
- **TrafficSigns**: ~1 GB
- **TinyImageNet**: ~3-4 GB (training), ~2 GB (inference)

### Training Optimizations

#### Learning Rate Scheduling
```python
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5,
    verbose=True
)
```

#### Early Stopping
```python
patience = 10  # Epochs without improvement
best_loss tracking with model checkpointing
```

#### Data Augmentation (TinyImageNet)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.2)
- Random rotation (±15°)
- Random crop (64×64 with padding=8)
- Color jitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
- Random grayscale (p=0.1)
- Random affine (translate=0.1, scale=0.9-1.1)

### Performance Characteristics

#### Training Time (Google Colab T4 GPU)

**Dataset-Specific (120 epochs for TinyImageNet):**
- **MNIST**: ~5 minutes (10 epochs)
- **EMNIST**: ~8 minutes (10 epochs)
- **TrafficSigns**: ~10 minutes (60 epochs)
- **TinyImageNet**: ~15 minutes (120 epochs)

**Epoch Configuration:**
```python
# MNIST and EMNIST - Simple digit recognition
num_epochs = 10      # Fast convergence on simple patterns

# TrafficSigns - Geometric shapes  
num_epochs = 60      # Medium complexity, more training needed

# TinyImageNet - Complex natural images
num_epochs = 120     # High complexity, extended training required
```

**Adversarial Training (2× base time due to doubled dataset):**
- **MNIST**: ~10 minutes (10 epochs × 2)
- **EMNIST**: ~16 minutes (10 epochs × 2)
- **TrafficSigns**: ~20 minutes (60 epochs × 2)
- **TinyImageNet**: ~30 minutes (120 epochs × 2)

#### Attack Generation Time (per batch)
- **FGSM**: ~0.1 seconds (single gradient step)
- **PGD**: ~1.0 seconds (10 iterations)
- **CW**: ~3.0 seconds (optimization-based, 10 iterations)
- **Compound**: ~4.0 seconds (sequential application of two attacks)

#### Defense Evaluation Time (Test-Time Overhead)

**Randomization Techniques:**
- **random_resizing**: ~10% overhead (resize operations)
- **random_cropping**: ~5% overhead (crop + resize)
- **random_rotation**: ~15% overhead (rotation + interpolation)
- **combined_randomization**: ~10-15% overhead (varies by selected technique)

**Input Transformation Techniques:**
- **image_quilting**: ~50% overhead (patch matching computation)
- **jpeg_compression**: ~20% overhead (encode + decode)
- **bit_depth_reduction**: ~5% overhead (quantization operations)
- **gaussian_noise_defense**: ~5% overhead (noise generation)
- **combined_input_transformation**: ~5-50% overhead (varies by selected technique)

**Adversarial Training:**
- **Test-time overhead**: 0% (no additional processing at test time)
- **Training overhead**: 100% (doubles training time)

### Defense Implementation Details

#### Randomization Defense Parameters

```python
# random_resizing configuration
resize_min_scale = 0.5      # Minimum resize scale (50% of original)
resize_max_scale = 1.5      # Maximum resize scale (150% of original)
# Process: resize to random scale → resize back to original size

# random_cropping configuration  
crop_min_scale = 0.8        # Minimum crop size (80% of original)
crop_max_scale = 1.0        # Maximum crop size (100% of original)
# Process: crop to random size → resize back to original size

# random_rotation configuration
rotation_min_angle = -15    # Minimum rotation angle (degrees)
rotation_max_angle = 15     # Maximum rotation angle (degrees)  
# Process: rotate by random angle → crop/pad → resize back

# combined_randomization configuration
techniques = ['resize', 'crop', 'rotate']
selection = random.choice(techniques)  # Uniform random selection per image
```

#### Input Transformation Parameters

```python
# image_quilting configuration
patch_size = 8                        # Patch size (8×8 pixels)
overlap = 2                           # Overlap between patches
patch_library = clean_training_data   # Source for clean patches
matching = 'SSD'                      # Sum of Squared Differences matching
# Process: divide image into patches → find best matching clean patches → reconstruct

# jpeg_compression configuration
jpeg_quality = 75                     # Compression quality (0-100)
jpeg_format = 'JPEG'                  # Standard JPEG format
# Process: encode to JPEG with quality parameter → decode back to tensor

# bit_depth_reduction configuration
original_bits = 8                     # Original bit depth per channel
reduced_bits = 4                      # Target bit depth (16 colors per channel)
# Process: quantize from 256 levels → 16 levels → restore to 256 levels

# gaussian_noise_defense configuration  
noise_mean = 0.0                      # Mean of Gaussian distribution
noise_std = 0.05                      # Standard deviation (adaptive)
# Process: generate Gaussian noise ~ N(0, σ²) → add to image → clip to valid range

# combined_input_transformation configuration
techniques = ['quilting', 'jpeg', 'bit_depth', 'noise']
selection = random.choice(techniques)  # Uniform random selection per image
```

#### Adversarial Training Configuration

```python
# Dataset composition
clean_ratio = 0.5              # 50% clean images
adversarial_ratio = 0.5        # 50% adversarial images

# Adversarial example generation
attack_for_training = attack_combination  # Use same attack as evaluation
epsilon = 0.03                 # Perturbation budget
alpha = 0.01                   # Step size for PGD
iterations = 10                # Number of attack iterations

# Training configuration
epochs = num_epochs            # Same as clean training (dataset-dependent)
batch_composition = 'mixed'    # Each batch has clean + adversarial samples
shuffle = True                 # Shuffle combined dataset

# Process:
# 1. Generate adversarial examples from clean training data
# 2. Create combined dataset = clean_data ∪ adversarial_data  
# 3. Train model from scratch on combined dataset
# 4. Evaluate on adversarial test data
```

## Configuration

### Global Configuration Variables

#### Defense Type Selection
```python
# Main defense category
defense_type = "randomization"  # Options: "adversarial_training", "randomization", "input_transformation"

# Randomization sub-technique (used when defense_type = "randomization")
randomization_defense = "random_resizing"  
# Options: "random_resizing", "random_cropping", "random_rotation", "combined_randomization"

# Input transformation sub-technique (used when defense_type = "input_transformation")
input_transformation = "image_quilting"
# Options: "image_quilting", "jpeg_compression", "bit_depth_reduction", 
#          "gaussian_noise_defense", "combined_input_transformation"
```

#### Attack Configuration
```python
# Compound attack combination
attack_combination = "fgsm_pgd_attack"  
# Options: "fgsm_pgd_attack", "fgsm_cw_attack", "cw_pgd_attack"
```

#### Dataset Configuration
```python
# Dataset selection
dataset_name = "TinyImageNet"  
# Options: "MNIST", "EMNIST", "TrafficSigns", "TinyImageNet"
```

#### Training Epochs (Dataset-Dependent)
```python
# MNIST and EMNIST
num_epochs = 10      # Fast training on simple digit datasets

# TrafficSigns  
num_epochs = 60      # Medium training on geometric shapes

# TinyImageNet
num_epochs = 120     # Extended training on complex natural images

# Note: Epochs are configured based on dataset complexity and data amount
# More complex datasets with more training data require more epochs
```

### Dataset-Specific Settings

#### TinyImageNet Configuration
```python
# Data amounts
train_per_class = 500  # Use all available training data
test_per_class = 50    # Use all available test data
total_train = 2500     # 500 × 5 classes
total_test = 250       # 50 × 5 classes

# Model architecture
channels = [32, 64, 128]  # Wider than other datasets
fc_sizes = [8192, 256]    # Larger fully connected layers
dropout = 0.6             # Higher dropout for regularization

# Training
batch_size = 256
learning_rate = 0.001
weight_decay = 3e-4       # Stronger L2 regularization
scheduler = ReduceLROnPlateau(factor=0.5, patience=5)

# Selected classes
tinyimagenet_selected_classes = [48, 57, 66, 75, 84]
# Maps to: keyboard, lawn mower, remote control, stopwatch, water tower
```

#### TrafficSigns Configuration
```python
# Data amounts
train_total = ~700       # Varies by class
test_total = ~175        # Varies by class

# Model architecture
channels = [16, 32, 64]  # Standard architecture
fc_sizes = [4096, 128]
dropout = 0.5

# Training
batch_size = 64
learning_rate = 0.001
weight_decay = 1e-4

# Classes
classes = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']
```

### Quantum Circuit Configuration

```python
# Qubit count
num_qubits = ceil(log2(output_dim))
# MNIST/EMNIST (10 classes): 4 qubits
# TrafficSigns (4 classes): 2 qubits
# TinyImageNet (5 classes): 3 qubits

# Parameters
theta = nn.Parameter(torch.randn(param_count))  # Rotation angles
phi = nn.Parameter(torch.randn(param_count))    # Phase angles

# Circuit structure
for each qubit:
    Ry(theta[i])  # Rotation around Y axis
for each adjacent pair:
    CNOT(qubit_i, qubit_i+1)  # Entanglement
```

## Output

### Result Files

Results are saved to Google Drive with descriptive filenames:
```
[defense_category]_[attack]_[defense_technique]_[dataset]_[timestamp].txt
```

**Examples:**
```
# Randomization defenses
randomization_fgsm_pgd_attack_resizing_TinyImageNet_20260212_152500.txt
randomization_fgsm_cw_attack_cropping_TinyImageNet_20260212_152600.txt
randomization_cw_pgd_attack_rotation_TinyImageNet_20260212_152700.txt
randomization_fgsm_pgd_attack_combined_randomization_TinyImageNet_20260212_152800.txt

# Input transformation defenses
input_transformation_fgsm_pgd_attack_quilting_TinyImageNet_20260212_153000.txt
input_transformation_fgsm_cw_attack_jpeg_compression_TinyImageNet_20260212_153100.txt
input_transformation_cw_pgd_attack_bit_depth_reduction_TinyImageNet_20260212_153200.txt
input_transformation_fgsm_pgd_attack_gaussian_noise_TinyImageNet_20260212_153300.txt
input_transformation_fgsm_cw_attack_combined_input_transformation_TinyImageNet_20260212_153400.txt

# Adversarial training
adversarial_training_fgsm_pgd_attack_TinyImageNet_20260212_154000.txt
adversarial_training_fgsm_cw_attack_TinyImageNet_20260212_154100.txt
adversarial_training_cw_pgd_attack_TinyImageNet_20260212_154200.txt
```

### Output Format

```
Defense Implemented: [Defense Type]
[Defense-specific] Approach: [Technique]

+----------+---------+---------------------+---------------------+
| Metric   | Clean   | No Defense Attack   | w/ Defense Attack   |
+==========+=========+=====================+=====================+
| Loss     | X.XX    | X.XX                | X.XX                |
+----------+---------+---------------------+---------------------+
| Accuracy | XX%     | XX%                 | XX%                 |
+----------+---------+---------------------+---------------------+

Example Misclassifications:
---------------------------------------------------------
Number of misclassified images for Clean: XX
Attack: [attack_type]
Dataset: [dataset_name]
Training Epochs: XXX
Trained Clean Images: XXXX
Test Images: XXX
Accuracy: X.XX
Precision: X.XX
Recall: X.XX
F1-score: X.XX
---------------------------------------------------------

Misclassifications:
[class1] -> [class2]: count
[class3] -> [class4]: count
...

====================================================================
ROBUSTNESS METRICS
====================================================================

ATTACK EFFECTIVENESS:
  Clean Accuracy: XX.XX%
  Attacked (No Defense): XX.XX%
  Accuracy Drop: XX.XX percentage points
  Attack Success Rate: XX.XX%

DEFENSE EFFECTIVENESS:
  Attacked (No Defense): XX.XX%
  With Defense: XX.XX%
  Accuracy Recovery: XX.XX percentage points
  Recovery Rate: XX.XX% of lost accuracy

OVERALL ROBUSTNESS:
  Absolute Improvement: XX.XX percentage points
  Relative Improvement: XX.XX% increase over no defense
  Remaining Gap to Clean: XX.XX percentage points
  Defense Robustness Score: XX.XX%
```

### Metrics Explained

#### Attack Effectiveness
- **Accuracy Drop**: Clean accuracy - Attacked accuracy (no defense)
- **Attack Success Rate**: Percentage of clean correct predictions that became incorrect after attack

#### Defense Effectiveness
- **Accuracy Recovery**: Defense accuracy - No defense accuracy
- **Recovery Rate**: (Accuracy recovery / Accuracy drop) × 100%

#### Overall Robustness
- **Absolute Improvement**: Raw percentage point gain from defense
- **Relative Improvement**: (Defense accuracy / No defense accuracy - 1) × 100%
- **Defense Robustness Score**: Final accuracy with defense applied

---

## Quick Reference Guide

### All Valid Defense Configurations

| # | Defense Category | Variable: `defense_type` | Sub-Technique | Variable: `randomization_defense` or `input_transformation` |
|---|-----------------|-------------------------|---------------|-----------------------------------------------------------|
| 1 | Adversarial Training | `"adversarial_training"` | N/A | Not applicable |
| 2 | Randomization | `"randomization"` | Random Resizing | `randomization_defense = "random_resizing"` |
| 3 | Randomization | `"randomization"` | Random Cropping | `randomization_defense = "random_cropping"` |
| 4 | Randomization | `"randomization"` | Random Rotation | `randomization_defense = "random_rotation"` |
| 5 | Randomization | `"randomization"` | Combined Randomization | `randomization_defense = "combined_randomization"` |
| 6 | Input Transformation | `"input_transformation"` | Image Quilting | `input_transformation = "image_quilting"` |
| 7 | Input Transformation | `"input_transformation"` | JPEG Compression | `input_transformation = "jpeg_compression"` |
| 8 | Input Transformation | `"input_transformation"` | Bit Depth Reduction | `input_transformation = "bit_depth_reduction"` |
| 9 | Input Transformation | `"input_transformation"` | Gaussian Noise | `input_transformation = "gaussian_noise_defense"` |
| 10 | Input Transformation | `"input_transformation"` | Combined Transform | `input_transformation = "combined_input_transformation"` |

### Attack Configurations

| Attack Type | Variable: `attack_combination` |
|-------------|-------------------------------|
| FGSM + PGD | `"fgsm_pgd_attack"` |
| FGSM + CW | `"fgsm_cw_attack"` |
| CW + PGD | `"cw_pgd_attack"` |

### Dataset Configurations

| Dataset | Variable: `dataset_name` | Epochs: `num_epochs` | Training Size | Test Size |
|---------|-------------------------|---------------------|---------------|-----------|
| MNIST | `"MNIST"` | 10 | 2,000 (200/class) | 500 (50/class) |
| EMNIST | `"EMNIST"` | 10 | 2,000 (200/class) | 500 (50/class) |
| TrafficSigns | `"TrafficSigns"` | 60 | ~700 (varies) | ~175 (varies) |
| TinyImageNet | `"TinyImageNet"` | 120 | 2,500 (500/class) | 250 (50/class) |

### Example Complete Configuration

```python
# Global variables at top of file
dataset_name = "TinyImageNet"
num_epochs = 120
defense_type = "randomization"
randomization_defense = "combined_randomization"
input_transformation = "combined_input_transformation"  # Used only if defense_type = "input_transformation"
attack_combination = "fgsm_pgd_attack"
```

**This configuration will:**
1. Use TinyImageNet dataset
2. Train for 120 epochs
3. Apply combined randomization defense (random selection of resize/crop/rotate)
4. Test against FGSM+PGD compound attack
5. Generate results file: `randomization_fgsm_pgd_attack_combined_randomization_TinyImageNet_[timestamp].txt`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch and torchvision teams for deep learning framework
- torchattacks library for adversarial attack implementations
- Google Cirq team for quantum circuit simulation
- TinyImageNet dataset from Stanford CS231n
- German Traffic Sign Recognition Benchmark (GTSRB)

---

**Note**: This framework is designed for research and educational purposes. Performance may vary based on hardware, library versions, and specific configurations.
