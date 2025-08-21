import os, sys
import argparse
import pickle
import importlib
from tqdm import tqdm
from optree import tree_map
import torch
import torch.nn as nn
import torchvision
import numpy as np
import posteriors
from torchvision import transforms, datasets
from torch.nn import GroupNorm
from datetime import datetime
import logging
import torchopt

from experiments.cifar10 import utils
from experiments.cifar10.data import prepare_data

# Get args from user
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
# parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument('--log_dir', type=str, default='results', help='root folder for saving logs')
parser.add_argument('--val_heldout', type=float, default=0.1, help='validation set heldout proportion')
args = parser.parse_args()

# Import configuration
config = importlib.import_module(args.config.replace("/", ".").replace(".py", ""))
args.batch_size = config.batch_size

# Set directory for saving results
main_dir = f'cifar10_val_heldout{args.val_heldout}/'
if 'alpha' in config.config_args.keys() and 'initial_lr' in config.config_args.keys():
    main_dir += f'ep{args.epochs}_bs{config.batch_size}_lr{config.config_args["initial_lr"]}_mo{config.config_args["alpha"]}/'
else:
    main_dir += f'ep{args.epochs}_bs{config.batch_size}/'
main_dir += f'seed{args.seed}_' + datetime.now().strftime('%Y_%m%d_%H%M%S')
args.log_dir = os.path.join(args.log_dir, main_dir)
utils.mkdir(args.log_dir)

# create logger
logging.basicConfig(
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, 'logs.txt')), 
        logging.StreamHandler()
    ], 
    format='[%(asctime)s,%(msecs)03d %(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger()
cmd = " ".join(sys.argv)
logger.info(f"Command :: {cmd}\n")

if torch.cuda.is_available():
    args.device = torch.device('cuda')
elif torch.backends.mps.is_available():
    args.device = torch.device('mps')
else:
    args.device = torch.device('cpu')

args.use_cuda = torch.cuda.is_available()

# Set seed
if args.seed != 42:
    config.save_dir += f"_seed{args.seed}"
    if config.params_dir is not None:
        config.params_dir += f"_seed{args.seed}/state.pkl"
else:
    args.seed = 42
    config.params_dir += "/state.pkl"
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

# Load data
train_loader, val_loader, test_loader, num_data = prepare_data(args)

# Evaluate on test dataset
# val_loader = test_loader

# Load model
def create_backbone():
    g = 32  # Define the number of groups for GroupNorm

    # Randomly initialized network, with GroupNorm instead of BatchNorm
    net = torchvision.models.resnet18(norm_layer=lambda c: GroupNorm(num_groups=g, num_channels=c), num_classes=args.num_classes)

    # Modify network so that it is appropriate for CIFAR images
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()

    # Randomly initialize the conv1 layer
    for p in net.conv1.parameters():
        if len(p.shape) > 1:
            nn.init.kaiming_normal_(p, nonlinearity='relu')
        else:
            nn.init.zeros_(p)      
    
    net.readout_name = 'fc'

    net.get_nb_parameters = lambda: np.sum(p.numel() for p in net.parameters())
    net.get_module_names = lambda: ''.join([f'{pn} -- shape = {list(p.shape)}, #params = {p.numel()}\n' for pn, p in net.named_parameters()])

    return net

model = create_backbone()
model.to(args.device)

# Set temperature
if "temperature" in config.config_args and config.config_args["temperature"] is None:
    config.config_args["temperature"] = args.temperature / num_data
    temp_str = str(args.temperature).replace(".", "-")
    config.save_dir += f"_temp{temp_str}"

# Create save directory if it does not exist
if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)

# Save config
utils.save_config(args, config.save_dir)
logger.info(f"Config saved to {config.save_dir}")

# Extract model parameters and buffers
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())
num_params = posteriors.tree_size(params).item()
logger.info(f"Number of parameters: {num_params / 1e6:.3f}M")

# Cosine annealing lr scheduler
num_epochs = args.epochs
steps_per_epoch = len(train_loader)
total_steps = num_epochs * steps_per_epoch
logger.info('Total training steps: %d' % total_steps)

# PyTorch's CosineAnnealingLR scheduler wrapper for posteriors compatibility
class PyTorchCosineScheduler:
    def __init__(self, initial_lr, total_steps, eta_min=1e-6):
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.total_steps = total_steps
    
    def __call__(self, step: int):
        # Use PyTorch's cosine annealing formula directly
        if step >= self.total_steps:
            return self.eta_min
            
        # PyTorch's CosineAnnealingLR formula
        step_tensor = torch.tensor(step, dtype=torch.float32)
        total_steps_tensor = torch.tensor(self.total_steps, dtype=torch.float32)
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                (1 + torch.cos(torch.pi * step_tensor / total_steps_tensor)) / 2
        return lr.item()
    
    def get_current_lr(self, step: int):
        # Helper method to get current learning rate for a given step
        return self.__call__(step)

lr_scheduler = None
if 'initial_lr' in config.config_args:
    lr_scheduler = PyTorchCosineScheduler(
        config.config_args["initial_lr"], 
        total_steps, 
        eta_min=1e-6
    )
    if config.method == posteriors.torchopt:
        config.config_args["optimizer"] = torchopt.chain(
            config.config_args["optimizer"],
            torchopt.transform.scale_by_schedule(lr_scheduler)
        )
    else:
        config.config_args["lr"] = lr_scheduler

# Define log posterior
def forward(p, batch):
    x, y = batch
    logits = torch.func.functional_call(model, p, x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    return logits, (loss, logits)

def outer_log_lik(logits, batch):
    _, y = batch
    return -torch.nn.functional.cross_entropy(logits, y, reduction="sum")

def log_posterior(p, batch):
    x, y = batch

    logits = torch.func.functional_call(model, p, x)
    
    loss = torch.nn.functional.cross_entropy(logits, y)
    log_post = (
        -loss
        + posteriors.diag_normal_log_prob(p, sd_diag=config.prior_sd, normalize=False)
        / num_data
    )

    return log_post, (loss, logits)

# Build transform
if 'initial_lr' in config.config_args:
    del config.config_args['initial_lr']
if config.method == posteriors.laplace.diag_ggn:
    transform = config.method.build(forward, outer_log_lik, **config.config_args)
else:
    transform = config.method.build(log_posterior, **config.config_args)

# Initialize state
state = transform.init(params)

# Function to evaluate accuracy on data set
def evaluate_accuracy(state, data_loader, device):
    if data_loader is None:
        return 0.0
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = tree_map(lambda x: x.to(device), batch)
            
            # Get current parameters from state - handle different posterior methods
            if hasattr(state, 'params'):
                # For methods that store params in state.params (like Adam-SGHMC)
                current_params = state.params
            else:
                # Fallback: build params dict from individual state attributes
                current_params = {}
                for name, param in params.items():
                    if hasattr(state, name):
                        current_params[name] = getattr(state, name)
                    else:
                        current_params[name] = param
            
            # Forward pass
            logits = torch.func.functional_call(model, current_params, x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total if total > 0 else 0.0

# Function to evaluate loss on a dataset
def evaluate_loss(state, data_loader, device):
    if data_loader is None:
        return 0.0
    
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = tree_map(lambda x: x.to(device), batch)
            
            # Get current parameters from state - handle different posterior methods
            if hasattr(state, 'params'):
                # For methods that store params in state.params (like Adam-SGHMC)
                current_params = state.params
            else:
                # Fallback: build params dict from individual state attributes
                current_params = {}
                for name, param in params.items():
                    if hasattr(state, name):
                        current_params[name] = getattr(state, name)
                    else:
                        current_params[name] = param
            
            # Forward pass
            logits = torch.func.functional_call(model, current_params, x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches if total_batches > 0 else 0.0

# Train
i = j = 0
num_batches = len(train_loader)
log_dict = {k: [] for k in config.log_metrics.keys()} | {"loss": [], "val_accuracy": [], "val_loss": []}
log_bar = tqdm(total=0, position=1, bar_format="{desc}")

# Set validation evaluation frequency - can be overridden in config
val_eval_frequency = getattr(config, 'val_eval_frequency', max(1, len(train_loader) // 5))  # Default: 5 times per epoch
if __name__ == '__main__':
    for epoch in range(args.epochs):
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", position=0
        ):
            batch = tree_map(lambda x: x.to(args.device), batch)
            state, aux = transform.update(state, batch)

            # Update metrics
            log_dict = utils.append_metrics(log_dict, state, aux[0], config.log_metrics)
            
            # Evaluate validation accuracy and loss periodically
            if i % val_eval_frequency == 0 or i % num_batches == 0:
                val_acc = evaluate_accuracy(state, val_loader, args.device)
                val_loss = evaluate_loss(state, val_loader, args.device)
                log_dict["val_accuracy"].append(val_acc)
                log_dict["val_loss"].append(val_loss)
            else:
                # Add previous validation metrics to maintain list length consistency
                if log_dict["val_accuracy"]:
                    log_dict["val_accuracy"].append(log_dict["val_accuracy"][-1])
                else:
                    log_dict["val_accuracy"].append(0.0)
                if log_dict["val_loss"]:
                    log_dict["val_loss"].append(log_dict["val_loss"][-1])
                else:
                    log_dict["val_loss"].append(0.0)
            
            # Update progress bar with log posterior, training loss, validation loss, and validation accuracy
            current_val_acc = log_dict["val_accuracy"][-1]
            current_val_loss = log_dict["val_loss"][-1]
            current_loss = log_dict["loss"][-1]
            
            if val_loader is not None and current_val_acc > 0:
                log_bar.set_description_str(
                    f"{config.display_metric}: {log_dict[config.display_metric][-1]:.2f} | Train Loss: {current_loss:.3f} | Val Loss: {current_val_loss:.3f} | Val Acc: {current_val_acc:.3f}"
                )
            else:
                log_bar.set_description_str(
                    f"{config.display_metric}: {log_dict[config.display_metric][-1]:.2f} | Train Loss: {current_loss:.3f}"
                )

            # Log
            i += 1
            if i % config.log_frequency == 0 or i % num_batches == 0:
                utils.log_metrics(
                    log_dict,
                    config.save_dir,
                    window=config.log_window,
                    file_name="training",
                )

            # Save sequential state if desired
            if (
                config.save_frequency is not None
                and (i - config.burnin) >= 0
                and (i - config.burnin) % config.save_frequency == 0
            ):
                with open(f"{config.save_dir}/state_{j}.pkl", "wb") as f:
                    pickle.dump(state, f)
                j += 1
        
        # Log current metrics and learning rate at the end of each epoch
        current_step = (epoch + 1) * steps_per_epoch - 1  # Calculate current step
        if lr_scheduler != None and hasattr(lr_scheduler, 'get_current_lr'):
            current_lr = lr_scheduler.get_current_lr(current_step)  # Get LR for current step
        else:
            current_lr = -1 # Not available
        current_display_metric = log_dict[config.display_metric][-1] if log_dict[config.display_metric] else 0.0
        current_train_loss = log_dict["loss"][-1] if log_dict["loss"] else 0.0
        current_val_acc = log_dict["val_accuracy"][-1] if log_dict["val_accuracy"] else 0.0
        current_val_loss = log_dict["val_loss"][-1] if log_dict["val_loss"] else 0.0
        
        if current_lr > 0:
            logger.info(f"End of Epoch {epoch + 1}/{args.epochs} - {config.display_metric}: {current_display_metric:.2f} | Train Loss: {current_train_loss:.3f} | Val Loss: {current_val_loss:.3f} | Val Acc: {current_val_acc:.3f} | Learning Rate: {current_lr:.8f}")
        else:
            logger.info(f"End of Epoch {epoch + 1}/{args.epochs} - {config.display_metric}: {current_display_metric:.2f} | Train Loss: {current_train_loss:.3f} | Val Loss: {current_val_loss:.3f} | Val Acc: {current_val_acc:.3f}")

    # Save final state
    with open(f"{config.save_dir}/state.pkl", "wb") as f:
        pickle.dump(state, f)