import os
import h5py
import yaml
import argparse
import wandb
import torch
from nn_ViT import SimpleViT
from nn_FNO import FNO2d


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train 2D Turbulence Model (FNO or ViT)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML configuration file')
    return parser.parse_args()



def init_wandb(config, current_rank):
    """Initialize Weights & Biases logging."""
    if not config['wandb']['enabled'] or current_rank != 0:
        return None
    
    try:
        # Prepare wandb config
        wandb_config = {
            'architecture': config['model']['architecture'],
            'data': config['data'],
            'training': config['training'],
            'model': config['model'],
            'optimizer': config['optimizer'],
            'loss': config['loss'],
            'paths': config['paths'],
            'options': config['options']
        }
        
        # Initialize wandb
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'] if config['wandb']['entity'] else None,
            name=config['wandb']['name'] if config['wandb']['name'] else None,
            config=wandb_config,
            tags=config['wandb']['tags'],
            notes=config['wandb']['notes']
        )
        
        print("✓ Weights & Biases initialized successfully")
        return wandb
        
    except Exception as e:
        print(f"⚠ Failed to initialize Weights & Biases: {e}")
        print("Continuing without wandb logging...")
        return None


def log_to_wandb(wandb_run, metrics, epoch, config, model=None, sample_data=None):
    """Log metrics and data to Weights & Biases."""
    if wandb_run is None:
        return
    
    try:
        # Log metrics
        wandb_run.log(metrics, step=epoch)
        
        # Log model architecture and parameters
        if config['wandb']['log_parameters'] and model is not None and epoch == 0:
            wandb_run.watch(model, log="parameters", log_freq=100)
        
        # Log sample data
        if config['wandb']['log_samples'] and sample_data is not None and epoch % config['wandb']['log_freq'] == 0:
            # Log sample input and output as images
            if len(sample_data) >= 2:
                input_sample = sample_data[0].cpu().numpy()
                output_sample = sample_data[1].cpu().numpy()
                
                # Create image plots
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                
                # Input
                im1 = axes[0].imshow(input_sample.T, cmap='viridis', vmax=input_sample[0,0].max(), vmin=input_sample[0,0].min())
                axes[0].set_title('Input Sample')
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0])
                
                # Output
                im2 = axes[1].imshow(output_sample.T, cmap='viridis', vmax=input_sample[0,0].max(), vmin=input_sample[0,0].min())
                axes[1].set_title('Output Sample')
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1])
                
                plt.tight_layout()
                wandb_run.log({"sample_data": wandb.Image(fig)}, step=epoch)
                plt.close(fig)
        
        # Log model
        if config['wandb']['log_model'] and model is not None and epoch % config['wandb']['log_freq'] == 0:
            # Save model artifact
            model_path = f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), model_path)
            artifact = wandb.Artifact(f"model-epoch-{epoch}", type="model")
            artifact.add_file(model_path)
            wandb_run.log_artifact(artifact)
            os.remove(model_path)  # Clean up temporary file
            
    except Exception as e:
        print(f"⚠ Failed to log to wandb: {e}")


def finish_wandb(wandb_run):
    """Finish Weights & Biases run."""
    if wandb_run is not None:
        try:
            wandb.finish()
            print("✓ Weights & Biases run finished")
        except Exception as e:
            print(f"⚠ Failed to finish wandb run: {e}")


def create_model(config, device):
    """Create model based on configuration."""
    architecture = config['model']['architecture'].lower()
    
    if architecture == 'vit':
        # ViT parameters
        img_size = tuple(config['model']['img_size'])
        patch_size = tuple(config['model']['patch_size'])
        in_chans = config['model']['in_chans']
        out_chans = config['model']['out_chans']
        embed_dim = config['model']['embed_dim']
        depth = config['model']['depth']
        num_heads = config['model']['num_heads']
        head_dim = config['model']['head_dim']
        mlp_dim_multiplier = config['model']['mlp_dim_multiplier']
        
        model = SimpleViT(
            img_size, patch_size, out_chans, embed_dim,
            depth=depth, heads=num_heads, 
            mlp_dim=embed_dim * mlp_dim_multiplier, 
            channels=in_chans, dim_head=head_dim
        ).to(device)
        
    elif architecture == 'fno':
        # FNO parameters
        modes = config['model']['modes']
        width = config['model']['width']
        
        model = FNO2d(modes, modes, width).to(device)
        
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: 'vit', 'fno'")
    
    return model


def load_checkpoint_if_specified(config, model, optimizer, scheduler, device, current_rank):
    """Load checkpoint if specified in config."""
    checkpoint_path = config['paths']['load_checkpoint']
    loaded_epoch = None
    loaded_best_loss = None
    
    if checkpoint_path and checkpoint_path.strip():
        if os.path.exists(checkpoint_path):
            if current_rank == 0:
                print(f"Loading checkpoint from: {checkpoint_path}")
            
            try:
                # Load the checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Load model state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint with optimizer and scheduler states
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if 'epoch' in checkpoint:
                        loaded_epoch = checkpoint['epoch'] + 1
                        if current_rank == 0:
                            print(f"Resuming from epoch {loaded_epoch}")
                    if 'best_loss' in checkpoint:
                        loaded_best_loss = checkpoint['best_loss']
                        if current_rank == 0:
                            print(f"Previous best loss: {loaded_best_loss}")
                else:
                    # Just model state dict
                    model.load_state_dict(checkpoint)
                    if current_rank == 0:
                        print("Loaded model state dict only")
                
                if current_rank == 0:
                    print("Checkpoint loaded successfully!")
                return True, loaded_epoch, loaded_best_loss
                
            except Exception as e:
                if current_rank == 0:
                    print(f"Error loading checkpoint: {e}")
                    print("Continuing with untrained model...")
                return False, None, None
        else:
            if current_rank == 0:
                print(f"Checkpoint file not found: {checkpoint_path}")
                print("Continuing with untrained model...")
            return False, None, None
    else:
        if current_rank == 0:
            print("No checkpoint specified, starting from scratch")
        return False, None, None
