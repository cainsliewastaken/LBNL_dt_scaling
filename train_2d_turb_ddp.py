import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import netCDF4 as nc
import os
import h5py
import yaml
import argparse
import wandb
import pickle

from load_data import *
from count_trainable_params import count_parameters
from nn_step_methods import *
from nn_dataloader_class import *
from nn_ViT import SimpleViT
from nn_FNO import FNO2d
from fvcore.nn import FlopCountAnalysis
from timeit import default_timer
from utilites import *

from torch.profiler import profile, record_function, ProfilerActivity

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
        # head_dim = int(embed_dim/num_heads)
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


################################################################
# Main execution
################################################################

# Parse command line arguments
args = parse_arguments()
config = load_config(args.config)

# Initialize distributed training
torch.distributed.init_process_group(backend=config['options']['backend'], init_method='env://')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
current_rank = torch.distributed.get_rank()
world_size = int(os.environ['WORLD_SIZE'])

# torch.manual_seed(0)
# np.random.seed(0)

device = torch.device(f"cuda:{local_rank}")

# Load parameters from config
time_step = float(config['data']['time_step'])
lead = int((1/1e-2)*time_step)
if current_rank==0:
    print(lead, time_step)

spinup = config['data']['spinup']
N_test = config['data']['N_test']
N_train = config['data']['N_train']
stats_file_path = config['data']['stats_path']

epochs = config['training']['epochs']
starting_epoch = config['training']['starting_epoch']
learning_rate = float(config['training']['learning_rate'])

Nlat = config['data']['Nlat']
Nlon = config['data']['Nlon']
T_train_final = config['data']['T_train_final']
T_test_final = config['data']['T_test_final']

batch_size = int(config['training']['batch_size'] / world_size)
batch_size_test = int(config['training']['batch_size_test'] / world_size)
batch_time = round(int(T_train_final/time_step)) 
batch_time_test = round(int(T_test_final/time_step)) 

if batch_time<2:
    batch_time = 2

if batch_time_test<2:
    batch_time_test = 2

if current_rank==0:
    print(batch_size, batch_size_test, T_train_final, T_test_final, batch_time, batch_time_test)

path_outputs = config['paths']['path_outputs']
net_name = config['paths']['net_name']
net_chkpt_path = path_outputs + str(net_name) + '/'

if current_rank==0:
    print(net_name)

if current_rank==0:
    if not os.path.exists(net_chkpt_path):
        os.makedirs(net_chkpt_path)
        print(f"Folder '{net_chkpt_path}' created.")
    else:
        print(f"Folder '{net_chkpt_path}' already exists.")

with open(stats_file_path, 'rb') as file:
    stats_dict = pickle.load(file)

m1 = stats_dict['mean']
s1 = stats_dict['std']


if current_rank==0:
    print(m1, s1)

dataset = Multistep_TimeSeriesDataset_load_from_file(spinup, N_train, batch_time, lead, m1, s1)
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=False)
train_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True) 

dataset_test = Multistep_TimeSeriesDataset_load_from_file(N_train + spinup, N_test, batch_time_test, lead, m1, s1)
sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=True, drop_last=False)
test_data = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, sampler=sampler_test, shuffle=False, num_workers=4, pin_memory=True) 

################################################################
# training and evaluation
################################################################

# Create model based on configuration
mynet = create_model(config, device).to(device)
Step_F = Euler_step(mynet, device, time_step).to(device)

# if current_rank==0:
#     for param in Step_F.parameters():
#         print(param.shape, param.requires_grad, param.is_cuda, param.dtype)

# Print model architecture info
if current_rank==0:
    architecture = config['model']['architecture'].lower()
    if architecture == 'vit':
        print(f"ViT Model - img_size: {config['model']['img_size']}, patch_size: {config['model']['patch_size']}, "
              f"embed_dim: {config['model']['embed_dim']}, depth: {config['model']['depth']}, "
              f"num_heads: {config['model']['num_heads']}, head_dim: {config['model']['head_dim']}")
    elif architecture == 'fno':
        print(f"FNO Model - modes: {config['model']['modes']}, width: {config['model']['width']}")

# Create optimizer based on config
if config['optimizer']['type'] == 'Adam':
    optimizer = optim.Adam(Step_F.parameters(), lr=learning_rate, fused=config['optimizer']['fused'])
else:
    raise ValueError(f"Unsupported optimizer type: {config['optimizer']['type']}")

# Create scheduler based on config
if config['optimizer']['scheduler_type'] == 'ExponentialLR':
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config['optimizer']['scheduler_gamma'])
elif config['optimizer']['scheduler_type'] == 'CosineAnnealingWarmRestarts':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config['optimizer']['scheduler_T_0'], eta_min=config['optimizer']['scheduler_eta_min'])
else:
    raise ValueError(f"Unsupported scheduler type: {config['optimizer']['scheduler_type']}")

# Create loss function based on config
if config['loss']['type'] == 'MSELoss':
    loss_fc_MSE = nn.MSELoss(reduction=config['loss']['reduction'])
else:
    raise ValueError(f"Unsupported loss type: {config['loss']['type']}")

# loss_fc_Norm = lambda input: torch.norm(input, dim=(2,3)).mean()

# Load checkpoint if specified
checkpoint_loaded, loaded_epoch, loaded_best_loss = load_checkpoint_if_specified(config, Step_F, optimizer, scheduler, device, current_rank)

# If checkpoint was loaded and contains epoch information, update starting_epoch
if checkpoint_loaded and loaded_epoch is not None:
    starting_epoch = loaded_epoch
    if current_rank == 0:
        print(f"Updated starting epoch to: {starting_epoch}")

# If checkpoint was loaded and contains best_loss information, update best_loss
best_loss = 1e2
if checkpoint_loaded and loaded_best_loss is not None:
    best_loss = loaded_best_loss
    if current_rank == 0:
        print(f"Updated best loss to: {best_loss}")
 
# loss_net = Loss_Singlestep(Step_F, batch_time, loss_fc_MSE)

# loss_net = Loss_Multistep(Step_F, batch_time, loss_fc_MSE) # type: ignore

# loss_net_jac = Jacobain_Train_Multistep(Step_F, batch_time, loss_fc_Norm)

# loss_net_test = Loss_Multistep(Step_F, batch_time_test, loss_fc_MSE) # type: ignore
# loss_net_test.eval()

# torch.set_printoptions(precision=10)
    

if current_rank==0:
    print('Num batches: ', len(train_data))
# count_parameters(Step_F)

Step_F = torch.nn.parallel.DistributedDataParallel(Step_F, device_ids=[local_rank], output_device=[local_rank])
# Step_F.compile()

# loss_net = torch.nn.parallel.DistributedDataParallel(loss_net, device_ids=[local_rank], output_device=[local_rank])
# loss_net_test = torch.nn.parallel.DistributedDataParallel(loss_net_test, device_ids=[local_rank], output_device=[local_rank])
# loss_net.compile()

# running_loss = 0.0
# if current_rank==0:
#     with profile(activities=[
#             ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
#         # with record_function("Load_Batch"):
#         batch = next(iter(train_data)).to(device)

#         # with record_function("Model_Inference"):
#         optimizer.zero_grad()
        
#         x_i = Step_F.forward(batch[:,0])
#         loss = loss_fc_MSE(x_i, batch[:,1])
#         for i in range(2, batch_time):
#             x_i = Step_F.forward(x_i)
#             loss = loss + loss_fc_MSE(x_i, batch[:,i])
            
#         # with record_function("Model_Backwards"):
#         loss.backward()
#         # with record_function("Optimizer_Adam_Step"):
#         optimizer.step()
#         # with record_function("Loss_Detach"):
#         running_loss += loss.detach().item() / batch_time
    
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
#     print(torch.cuda.memory_summary())

#     torch.cuda.empty_cache()

if current_rank==0:
    with torch.no_grad():
        input_temp = torch.Tensor(dataset[0]).to(device)
        flops = FlopCountAnalysis(Step_F, input_temp[0].unsqueeze(0))

        #   Print total FLOPs
        print(f"Total FLOPs: {flops.total()}")
        print(f"Total FLOPs / time_step: {flops.total()/time_step}")

    #   Print FLOPs by operator
        print("FLOPs by operator:", flops.by_operator())
    
    #   Print FLOPs by module
    #   print("FLOPs by module:", flops.by_module())
        count_parameters(Step_F)


# Initialize Weights & Biases
wandb_run = init_wandb(config, current_rank)

for ep in range(starting_epoch, epochs+1):
    running_loss = 0.0
    sampler.set_epoch(ep)
    for batch in train_data:
        batch = batch.to(device)
        optimizer.zero_grad()

        # loss = loss_net(batch)
        
        x_i = Step_F(batch[:,0])
        loss = loss_fc_MSE(x_i, batch[:,1])
        for i in range(2, batch_time):
            x_i = Step_F(x_i)
            loss += loss_fc_MSE(x_i, batch[:,i])
        
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item() / batch_time

        # if current_rank==0:
        #   print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())

    # if current_rank==0:
    #     #   print(loss)
    #     print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())

    optimizer.zero_grad()
    scheduler.step()
    torch.cuda.empty_cache()

    if current_rank==0:
        with torch.no_grad():
            sampler_test.set_epoch(ep)
            net_loss = (running_loss/(len(train_data)))
            
            print('Starting eval')
            key = np.random.randint(len(test_data))
            batch = next(iter(test_data)).to(device)

            x_i = Step_F(batch[:,0])
            test_loss = loss_fc_MSE(x_i, batch[:,1])
            for i in range(2, batch_time):
                x_i = Step_F(x_i)
                test_loss += loss_fc_MSE(x_i, batch[:,i])
            
            test_loss = test_loss.detach() / batch_time_test
            print(f'Epoch : {ep}, Train Loss : {net_loss}, Test Loss : {test_loss}')
            print('Learning rate', scheduler._last_lr)
            
            # Log metrics to wandb
            metrics = {
                'epoch': ep,
                'train_loss': net_loss,
                'test_loss': test_loss.item() if torch.is_tensor(test_loss) else test_loss,
                'learning_rate': scheduler._last_lr[0] if isinstance(scheduler._last_lr, list) else scheduler._last_lr,
                'best_loss': best_loss
            }
            
            # Get sample data for visualization
            sample_data = None
            if config['wandb']['log_samples'] and ep % config['wandb']['log_freq'] == 0:
                try:
                    sample_batch = next(iter(test_data)).to(device)
                    output = Step_F(sample_batch[0,0])
                    sample_data = (output, sample_batch[0, 1])
                except:
                    sample_data = None
                
                log_to_wandb(wandb_run, metrics, ep, config, mynet, sample_data)
            if best_loss > test_loss:
                print('Saved!!!')
                # Save full checkpoint with optimizer and scheduler states
                checkpoint = {
                    'epoch': ep,
                    'model_state_dict': Step_F.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': test_loss,
                    'config': config
                }
                torch.save(checkpoint, net_chkpt_path+'/'+'chkpt_'+net_name+'_best_chkpt.pt')
                print('Checkpoint updated')
                best_loss = test_loss
        if ep % config['options']['save_every_n_epochs'] == 0:
            # Save full checkpoint with optimizer and scheduler states
            checkpoint = {
                'epoch': ep,
                'model_state_dict': Step_F.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': config
            }
            torch.save(checkpoint, net_chkpt_path+'chkpt_'+net_name+'_epoch_'+str(ep)+'.pt')
        print('Next epoch')
    

# Save final checkpoint with full state
final_checkpoint = {
    'epoch': epochs,
    'model_state_dict': Step_F.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_loss': best_loss,
    'config': config
}
torch.save(final_checkpoint, net_chkpt_path+'_final.pt')

# Finish wandb run
finish_wandb(wandb_run)
