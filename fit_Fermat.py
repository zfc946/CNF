import data, helper
import nbf
import pytorch_lightning as pl
import torch

# Use a seed for reproducibility
seed = 2
pl.seed_everything(seed)
toy_data = data.FermatDataModule()
pretrained_ckpt = None
basis = nbf.PolynomialBasis(2,1)

def velocity_map(p):
    return torch.max((-p[:,1].clone().detach()+10.5)/10, torch.zeros_like(p[:,1]))

# Least time loss
def least_time_loss(batch, model):
    x, y = batch
    p = model(x)
    displacement = torch.linalg.norm(torch.diff(p, dim=0), ord=2, dim=1)
    mean_v = 0.5*(velocity_map(p)[:-1] + velocity_map(p)[1:])
    loss = torch.div(displacement, mean_v).sum()
    return loss

# Specify training parameters
max_epochs = 100
smooth_exp = 2
lr = 1e-3
cond_num_weight=0.0001
smooth_weight = 0.0

# Initialize NBF
if pretrained_ckpt is None:
    NBF = nbf.NeuralBasisField(basis=basis, cond_num_weight=cond_num_weight, loss_fn=least_time_loss, smooth_weight=smooth_weight,
        constraint_pt=toy_data.constraint_pt, constraint_value=toy_data.constraint_value, lr=lr,
        k_list=toy_data.k_list, op_list=toy_data.op_list, smooth_exp=smooth_exp, use_regression=False)

with torch.inference_mode(True):
    travel_time = least_time_loss([toy_data.training_pt, None], NBF)
    helper.plot_Fermat(NBF, toy_data.training_pt, "optical_path_init.png", epoch=NBF.current_epoch, travel_time=travel_time)

clbks = []
trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1, callbacks=clbks, log_every_n_steps=1, num_sanity_val_steps=-1) #, gradient_clip_val=0.5)
trainer.fit(NBF, toy_data)

with torch.inference_mode(True):
    travel_time = trainer.callback_metrics["train/additional_loss"]
    helper.plot_Fermat(NBF, toy_data.training_pt, "optical_path.png", epoch=NBF.current_epoch, travel_time=travel_time)