import data, helper
import network
import nbf
import pytorch_lightning as pl
import torch
from glob import glob
import os

# Use a seed for reproducibility
perturb = 0.1
beta = 0.1
seed = 2
pl.seed_everything(seed)
function = 'sin'
xlims = (0.0, 1.0, 32)      # (min, max, N)
tlims = (0.0, 2.0, 32)
advection_data = data.AdvectionDataModule(function, xL=xlims[0], xR=xlims[1], t0=tlims[0], tT=tlims[1], num_x=xlims[2], num_t=tlims[2], perturb=perturb, beta=beta)

# Optional: pass a pretrained model
pretrained_ckpt = None
# pretrained_ckpt = max(glob('lightning_logs/*/checkpoints/*.ckpt'), key=os.path.getmtime)

# Specify training parameters
max_epochs = 30
smooth_exp = 2
lr = 1e-3
smooth_weight = 1e0
init_sigma = 0.1

# Specify the basis function
basis = network.init_basis("kernel", advection_data, init_sigma=init_sigma, per_basis_sigma=True, identity_encoder=True)

# Initialize NBF
if pretrained_ckpt is None:
    NBF = nbf.NeuralBasisField(basis=basis,
        constraint_pt=advection_data.constraint_pt, constraint_value=advection_data.constraint_value, lr=lr, smooth_weight=smooth_weight,
        k_list=advection_data.k_list, op_list=advection_data.op_list, smooth_exp=smooth_exp, use_regression=False)
else:
    NBF = nbf.NeuralBasisField.load_from_checkpoint(pretrained_ckpt, basis=basis,
        constraint_pt=advection_data.constraint_pt, constraint_value=advection_data.constraint_value, lr=lr, smooth_weight=smooth_weight,
        k_list=advection_data.k_list, op_list=advection_data.op_list, smooth_exp=smooth_exp, use_regression=False)
    
clbks = [pl.callbacks.ModelCheckpoint(save_weights_only=False, monitor='train/condition_number', mode='min')]
clbks = []
trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1, callbacks=clbks, log_every_n_steps=1, num_sanity_val_steps=0)
# trainer.validate(NBF, advection_data)  # check the mse at 0th epoch
trainer.fit(NBF, advection_data)

with torch.inference_mode(False):
    helper.visualize_advection(NBF, beta=beta, xlims=xlims[:2], tlims=tlims[:2])