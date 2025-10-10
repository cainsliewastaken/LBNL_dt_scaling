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

from load_data import *
from count_trainable_params import count_parameters
from nn_step_methods import *
from nn_dataloader_class import *
from nn_ViT import SimpleViT
from nn_FNO import FNO2d
from fvcore.nn import FlopCountAnalysis
from timeit import default_timer
from utilites import *


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

# torch.manual_seed(0)
# np.random.seed(0)

device = 'cuda'
 
# Initialize Weights & Biases
wandb_run = init_wandb(config, current_rank)

# Load parameters from config
time_step = float(config['data']['time_step'])
lead = int((1/1e-2)*time_step)
if current_rank==0:
    print(lead, time_step)

spinup = config['data']['spinup']
N_test = config['data']['N_test']
N_train = config['data']['N_train']

epochs = config['training']['epochs']
starting_epoch = config['training']['starting_epoch']
learning_rate = float(config['training']['learning_rate'])

Nlat = config['data']['Nlat']
Nlon = config['data']['Nlon']
T_train_final = config['data']['T_train_final']
T_test_final = config['data']['T_test_final']

batch_size = config['training']['batch_size']
batch_size_test = config['training']['batch_size_test']
batch_time = int(T_train_final/time_step) 
batch_time_test = int(T_test_final/time_step) 
if batch_time<2:
    batch_time = 2

if batch_time_test<2:
    batch_time_test = 2

if current_rank==0:
    print(batch_size, batch_size_test, T_train_final, T_test_final, batch_time, batch_size_test)

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

### load test data ##
# need to move mean and std calculation to a new file
psi_train_input_Tr_torch = load_train_data_v2(spinup, N_train)

psi_test_input_Tr_torch = load_train_data_v2(N_train + spinup, N_test)

if current_rank==0:
    print('Data loaded, shape: ', psi_train_input_Tr_torch.shape)

m1 = torch.mean(psi_train_input_Tr_torch.flatten())
s1 = torch.std(psi_train_input_Tr_torch.flatten())


m1_test = torch.mean(psi_test_input_Tr_torch.flatten())
s1_test = torch.std(psi_test_input_Tr_torch.flatten())

del psi_train_input_Tr_torch
del psi_test_input_Tr_torch
#########

m1 = m1.detach().cpu().numpy()
s1 = s1.detach().cpu().numpy()
m1_test = m1_test.detach().cpu().numpy()
s1_test = s1_test.detach().cpu().numpy()

if current_rank==0:
    print(m1, m1_test, s1, s1_test)
    print('Detatched m1, s1')

dataset = Multistep_TimeSeriesDataset_load_from_file(spinup, N_train, batch_time, lead, m1, s1)
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=False)
train_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False) 

dataset_test = Multistep_TimeSeriesDataset_load_from_file(N_train + spinup, N_test, batch_time_test, lead, m1, s1)
sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=True, drop_last=False)
test_data = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, sampler=sampler_test, shuffle=False) 

################################################################
# training and evaluation
################################################################

# Create model based on configuration
mynet = create_model(config, device)

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
    optimizer = optim.Adam(mynet.parameters(), lr=learning_rate, fused=config['optimizer']['fused'])
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
    loss_fc_RMSE = nn.MSELoss(reduction=config['loss']['reduction'])
else:
    raise ValueError(f"Unsupported loss type: {config['loss']['type']}")

loss_fc_Norm = lambda input: torch.norm(input, dim=(2,3)).mean()
Step_F = Euler_step(mynet, device, time_step).cuda()

# Load checkpoint if specified
checkpoint_loaded, loaded_epoch, loaded_best_loss = load_checkpoint_if_specified(config, mynet, optimizer, scheduler, device, current_rank)

# If checkpoint was loaded and contains epoch information, update starting_epoch
if checkpoint_loaded and loaded_epoch is not None:
    starting_epoch = loaded_epoch
    if current_rank == 0:
        print(f"Updated starting epoch to: {starting_epoch}")

# If checkpoint was loaded and contains best_loss information, update best_loss
if checkpoint_loaded and loaded_best_loss is not None:
    best_loss = loaded_best_loss
    if current_rank == 0:
        print(f"Updated best loss to: {best_loss}")
 
# loss_net = Loss_Singlestep(Step_F, batch_time, loss_fc_RMSE)

loss_net = Loss_Multistep(Step_F, batch_time, loss_fc_RMSE) # type: ignore

# loss_net_jac = Jacobain_Train_Multistep(Step_F, batch_time, loss_fc_Norm)

loss_net_test = Loss_Multistep_Test(Step_F, batch_time_test, loss_fc_RMSE) # type: ignore
loss_net_test.eval()

torch.set_printoptions(precision=10)

best_loss = 1e2
# if current_rank==0:
#     print('Num batches: ',train_data.shape[0])
# count_parameters(Step_F)

loss_net = torch.nn.parallel.DistributedDataParallel(loss_net, device_ids=[local_rank], output_device=[local_rank])
loss_net_test = torch.nn.parallel.DistributedDataParallel(loss_net_test, device_ids=[local_rank], output_device=[local_rank])
# loss_net.compile()

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

for ep in range(starting_epoch, epochs+1):
    running_loss = 0.0
    sampler.set_epoch(ep)
    for batch in train_data:
        batch = batch.to(device)
        optimizer.zero_grad()

        # loss = loss_net_jac(batch) #this accumulates gradients inside the loss function, no backwards necessary
        loss = loss_net(batch)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        # if current_rank==0:
        #   print(loss)
        #   print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())


    scheduler.step()
    with torch.no_grad():
        sampler_test.set_epoch(ep)
        net_loss = (running_loss/(len(train_data) * batch_time))
        if current_rank==0:
          print('Starting eval')
        key = np.random.randint(len(test_data))
        test_loss = loss_net_test(next(iter(test_data)).to(device)) / batch_time_test
        if current_rank==0:
            print(f'Epoch : {ep}, Train Loss : {net_loss}, Test Loss : {test_loss}')
        if current_rank==0:
            print('Learning rate', scheduler._last_lr)
        
        # Log metrics to wandb
        if current_rank == 0:
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
            if current_rank==0:
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
            if current_rank==0:
                print('Checkpoint updated')
            best_loss = test_loss
    torch.cuda.empty_cache()
    if ep % config['options']['save_every_n_epochs'] == 0:
        if current_rank==0:
            print(net_chkpt_path+'chkpt_'+net_name+'_epoch_'+str(ep)+'.pt')
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
