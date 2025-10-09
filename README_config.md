# YAML Configuration for 2D Turbulence Training (ViT and FNO)

This document explains how to use the YAML configuration system for training both ViT and FNO 2D turbulence models.

## Usage

### Unified Training Script

The unified training script can train both ViT and FNO architectures based on the configuration file:

```bash
python train_2d_turb_ddp.py --config config_vit.yaml    # Train ViT model
python train_2d_turb_ddp.py --config config_fno.yaml    # Train FNO model
python train_2d_turb_ddp.py --config config.yaml        # Uses default config

## Configuration Structure

The YAML file is organized into the following sections:

### Data Parameters (`data`)
- `time_step`: Time step for the simulation
- `spinup`: Number of spinup steps
- `N_test`: Number of test samples
- `N_train`: Number of training samples
- `Nlat`, `Nlon`: Grid dimensions
- `T_train_final`, `T_test_final`: Final time for training and testing

### Training Parameters (`training`)
- `epochs`: Number of training epochs
- `starting_epoch`: Starting epoch (for resuming training)
- `learning_rate`: Learning rate for the optimizer
- `batch_size`: Training batch size
- `batch_size_test`: Test batch size

### Model Architecture (`model`)
- `architecture`: Model type - "vit" or "fno"

#### ViT Parameters (when architecture="vit")
- `img_size`: Input image size [height, width]
- `patch_size`: Patch size for ViT [height, width]
- `in_chans`, `out_chans`: Input and output channels
- `embed_dim`: Embedding dimension
- `depth`: Number of transformer layers
- `num_heads`: Number of attention heads
- `head_dim`: Dimension of each attention head
- `mlp_dim_multiplier`: Multiplier for MLP dimension (mlp_dim = embed_dim * multiplier)

#### FNO Parameters (when architecture="fno")
- `modes`: Number of Fourier modes
- `width`: Width of the FNO layers

### Optimizer Configuration (`optimizer`)
- `type`: Optimizer type (currently supports "Adam")
- `fused`: Whether to use fused Adam optimizer
- `scheduler_type`: Learning rate scheduler type ("ExponentialLR" or "CosineAnnealingWarmRestarts")
- `scheduler_gamma`: Gamma for ExponentialLR scheduler
- `scheduler_T_0`: T_0 for CosineAnnealingWarmRestarts scheduler
- `scheduler_eta_min`: Minimum learning rate for CosineAnnealingWarmRestarts

### Loss Function (`loss`)
- `type`: Loss function type (currently supports "MSELoss")
- `reduction`: Reduction method for the loss

### Paths (`paths`)
- `path_outputs`: Base path for output files
- `net_name`: Name for the network (used in checkpoint filenames)
- `load_checkpoint`: Path to checkpoint file to resume training from (optional - leave empty to start from scratch)

### Options (`options`)
- `save_every_n_epochs`: How often to save checkpoints
- `use_distributed`: Whether to use distributed training
- `backend`: Backend for distributed training

### Weights & Biases (`wandb`)
- `enabled`: Enable/disable wandb logging
- `project`: Wandb project name
- `entity`: Wandb entity (username/team) - leave null for default
- `name`: Run name - leave null for auto-generated
- `tags`: List of tags for the run
- `notes`: Description of the run
- `log_model`: Whether to log model artifacts
- `log_samples`: Whether to log sample visualizations
- `log_freq`: How often to log samples and models (in epochs)
- `log_gradients`: Whether to log gradients (not recommended for large models)
- `log_parameters`: Whether to log model parameters

## Example Configurations

- `config.yaml` - Default configuration (ViT)
- `config_vit.yaml` - ViT-specific configuration
- `config_fno.yaml` - FNO-specific configuration

## Modifying Configuration

To modify training parameters:

1. Edit the `config.yaml` file with your desired parameters
2. Run the training script: `python train_2d_turb_ddp.py --config config.yaml`

You can create multiple configuration files for different experiments:

```bash
python train_2d_turb_ddp.py --config config_experiment1.yaml
python train_2d_turb_ddp.py --config config_experiment2.yaml
```

## Checkpoint Loading and Resuming

The training script supports resuming from checkpoints:

### Resuming Training

To resume training from a checkpoint, set the `load_checkpoint` path in your YAML file:

```yaml
paths:
  load_checkpoint: "/path/to/your/checkpoint.pt"
```

### Checkpoint Format

The script saves checkpoints in a comprehensive format that includes:
- Model state dictionary
- Optimizer state
- Scheduler state
- Current epoch
- Best loss achieved
- Configuration used


### Checkpoint Files

The script automatically saves checkpoints:
- **Best checkpoint**: `chkpt_{net_name}_best_chkpt.pt` (saved when test loss improves)
- **Periodic checkpoints**: `chkpt_{net_name}_epoch_{N}.pt` (saved every N epochs)
- **Final checkpoint**: `{net_name}_final.pt` (saved at the end of training)

## Weights & Biases Integration

The training script includes comprehensive Weights & Biases (wandb) integration for experiment tracking and visualization.

### Setup

1. **Install wandb**: `pip install wandb`
2. **Login**: `wandb login` (follow the instructions to get your API key)
3. **Configure**: Set `wandb.enabled: true` in your YAML config

### Features

- **Automatic logging** of training metrics (loss, learning rate, etc.)
- **Model architecture tracking** with parameter counts
- **Sample visualization** showing input/output pairs
- **Model artifacts** saved as wandb artifacts
- **Hyperparameter tracking** from YAML configuration
- **Run comparison** across different experiments

### Logged Metrics

- `train_loss`: Training loss per epoch
- `test_loss`: Test loss per epoch  
- `learning_rate`: Current learning rate
- `best_loss`: Best test loss achieved
- `sample_data`: Visualizations of input/output samples

### Configuration

```yaml
wandb:
  enabled: true
  project: "2d-turbulence-unified"
  entity: "your-username"  # Optional
  name: "experiment-1"     # Optional, auto-generated if null
  tags: ["turbulence", "vit", "pde"]
  notes: "Training description"
  log_model: true
  log_samples: true
  log_freq: 10
  log_parameters: true
```

### Usage

```bash
# Training with wandb logging
python train_2d_turb_ddp.py --config config_vit.yaml

# Disable wandb logging
# Set wandb.enabled: false in YAML config
```

### Viewing Results

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project
3. View real-time training progress, metrics, and visualizations
4. Compare different runs and hyperparameters

## Validation

Use the test scripts to validate your configurations:

```bash
# Test individual configurations
python test_config.py

# Test unified configuration system
python test_unified_config.py
```

These will check that all required sections are present and the YAML files are valid for both ViT and FNO architectures.
