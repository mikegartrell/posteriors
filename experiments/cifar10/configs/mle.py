import posteriors
import torchopt
import torch

name = "mle"
save_dir = "experiments/cifar-10/results/" + name
params_dir = None  # directory to load state containing initialisation params


prior_sd = torch.inf
batch_size = 64
burnin = None
save_frequency = None

lr=1e-2

method = posteriors.torchopt
config_args = {
    "optimizer": torchopt.adamw(lr=lr, maximize=True)
}  # arguments for method.build (aside from log_posterior)
log_metrics = {
    "log_post": "loss",
}  # dict containing names of metrics as keys and their paths in state as values
display_metric = "log_post"  # metric to display in tqdm progress bar


log_frequency = 100  # frequency at which to log metrics
log_window = 30  # window size for moving average


def forward(model, state, batch):
    x, _ = batch
    logits = torch.func.functional_call(model, state.params, x)
    return logits.unsqueeze(1)


forward_dict = {"MLE": forward}
