import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch

import data
import network
from pathlib import Path
from tqdm import tqdm
import numpy as np
import csv
import nbf

import functorch

def get_n_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_basis(basis_type, data_module, kernel_type = "Gaussian", init_sigma=1.0):
    '''
    Quick initialization of a basis network given its type and data module
    '''
    if basis_type == "baseline_ffn":
        # Specify the main network
        n_freqs = 16
        encoder = network.Encoder(network.FFN(data_module.dim[0], n_freqs=n_freqs, scale=1), n_layers=1, n_hidden=718, n_out=data_module.dim[1])

        # Specify the hypernet basis function
        basis = nbf.SingleBasis(encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "baseline_ffn_200k":
        # Specify the main network
        n_freqs = 16
        encoder = network.Encoder(network.FFN(data_module.dim[0], n_freqs=n_freqs, scale=1), n_layers=1, n_hidden=426, n_out=data_module.dim[1])

        # Specify the hypernet basis function
        basis = nbf.SingleBasis(encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "baseline_siren":
        # Specify the main network
        mnet = network.Siren(data_module.dim[0], hidden_features=442, hidden_layers=1, out_features=data_module.dim[1], outermost_linear=True)

        # Specify the hypernet basis function
        basis = nbf.SingleBasis(mnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "kernel_ffn":
        # Specify the encoder for the kernel function
        n_freqs = 16
        encoder = network.Encoder(network.FFN(data_module.dim[0], n_freqs=n_freqs, scale=1), n_layers=1, n_hidden=512)

        # Specify the kernel basis function
        basis = nbf.KernelBasis(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma)
    elif basis_type == "kernel_pe":
        # Specify the encoder for the kernel function
        n_freqs = 6
        encoder = network.Encoder(network.PositionalEncoder(data_module.dim[0], n_freqs))

        # Specify the kernel basis function
        basis = nbf.KernelBasis(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma)
    elif basis_type == "kernel_ffn_200k":
        # Specify the encoder for the kernel function
        n_freqs = 16
        encoder = network.Encoder(network.FFN(data_module.dim[0], n_freqs=n_freqs, scale=1), n_layers=1, n_hidden=248)

        # Specify the kernel basis function
        basis = nbf.KernelBasis(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma)
    elif basis_type == "kernel_siren":
        # Specify the encoder for the kernel function
        encoder = network.Siren(data_module.dim[0], hidden_features=256, hidden_layers=1, out_features=512, outermost_linear=True)

        # Specify the kernel basis function
        basis = nbf.KernelBasis(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma)
    elif basis_type == "abla_kernel_siren_dot":
        # Specify the encoder for the kernel function
        encoder = network.Siren(data_module.dim[0], hidden_features=256, hidden_layers=1, out_features=512, outermost_linear=True)

        # Specify the kernel basis function
        basis = nbf.KernelBasis(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type='dot', init_sigma=init_sigma)
    elif basis_type == "abla_hypernet_siren":
        # Specify the main network
        mnet = network.Siren(data_module.dim[0], hidden_features=442, hidden_layers=1, out_features=data_module.dim[1], outermost_linear=True)

        # Specify the hypernetwork
        _, mnet_params = functorch.make_functional(mnet)
        n_mnet_params = sum([p.numel() for p in mnet_params])
        hnet = torch.nn.Sequential(
                torch.nn.Linear(len(data_module.constraint_pt), 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_mnet_params))

        # Specify the hypernet basis function
        basis = nbf.HypernetBasis(mnet, hnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "abla_netbasis_siren":
        # Specify the main network
        num_basis = data_module.constraint_pt.shape[0]
        net = network.Siren(data_module.dim[0], hidden_features=319, hidden_layers=1, out_features=num_basis*data_module.dim[1], outermost_linear=True)

        # Specify the hypernet basis function
        basis = nbf.NetBasis(net, num_basis=num_basis, dim=data_module.dim)
    else:
        RuntimeError(f'Unknown basis type: {basis_type}')

    return basis

def brdf_to_rgb(rvectors, brdf):
	hx = torch.reshape(rvectors[:, 0], (-1, 1))
	hy = torch.reshape(rvectors[:, 1], (-1, 1))
	hz = torch.reshape(rvectors[:, 2], (-1, 1))
	dx = torch.reshape(rvectors[:, 3], (-1, 1))
	dy = torch.reshape(rvectors[:, 4], (-1, 1))
	dz = torch.reshape(rvectors[:, 5], (-1, 1))

	theta_h = torch.atan2(torch.sqrt(hx ** 2 + hy ** 2), hz)
	theta_d = torch.atan2(torch.sqrt(dx ** 2 + dy ** 2), dz)
	phi_d = torch.atan2(dy, dx)
	wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
		torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
	rgb = brdf * torch.clamp(wiz, 0, 1)
	return rgb


def run_tune(hparams):

    if hparams['basis_name'].startswith('baseline'):
        hparams['n_constraint'] = 1

    # Use a seed for reproducibility
    seed = 2
    pl.seed_everything(seed)
    merl_path = Path(f"{hparams['merl_path']}/{hparams['brdf_name']}.binary")
    dataset = data.MerlDataset(str(merl_path), n_constraint=hparams['n_constraint'], logscale=hparams['logscale'], rvectors_path=hparams['rvectors_path'])
    max_epochs = hparams['max_epochs']

    if 'exp_dir' not in hparams:
        exp_dir = Path('exp') / f'brdf_{merl_path.stem}' / hparams['tune_name'] /f"basis_{hparams['basis_name']}" 
    else:
        exp_dir = Path(hparams['exp_dir'])
    exp_dir.mkdir(exist_ok=True, parents=True)


    basis = init_basis(hparams['basis_name'], dataset)


    def brdf_loss(batch, model):
        x, y, *_ = batch
        y_hat = model(x)

        rgb_pred = brdf_to_rgb(x, y_hat)
        rgb_true = brdf_to_rgb(x, y)
        loss_regression = torch.mean(torch.abs(rgb_true-rgb_pred))
        return loss_regression
    

    
    baseline = nbf.NeuralBasisField(basis=basis,
        constraint_pt=dataset.constraint_pt, constraint_value=dataset.constraint_value, use_regression=False,
        k_list=dataset.k_list, op_list=dataset.op_list,
        smooth_exp=0, smooth_weight=0, 
        loss_fn=brdf_loss, **hparams['model_args'])


    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(exp_dir))
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1, log_every_n_steps=1, num_sanity_val_steps=-1, 
                         logger=tb_logger, check_val_every_n_epoch=1)
    trainer.fit(baseline, dataset)

    train_time = baseline.train_time

    test_data = dataset.test_dataloader()
    out = []
    gt=[]
    with torch.no_grad():
        baseline.to('cuda')
        for batch in tqdm(test_data, desc="Test "):
            x, y = batch
            out.append(baseline(x.cuda()))
            gt.append(y)

    pred = torch.concat(out, axis=0).cpu().numpy()
    gt = torch.concat(gt, axis=0).cpu().numpy()
    mse = np.mean((pred - gt) ** 2)


    with (exp_dir/'train_time.csv').open(mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['mse', 'time', 'n_param'])
        writer.writerow([mse, train_time, get_n_param(baseline)])



if __name__ == '__main__':
    hparams = {
        'brdf_name': 'chrome-steel',
        'tune_name': 'default_beta_logscale',
        'rvectors_path': '',
        'merl_path': 'PATH/TO/MERL/DATASET',
        'basis_name': 'kernel_ffn',
        'max_epochs': 10,
        'n_constraint': 100,
        'logscale': True,
        'model_args': {
            'cond_num_weight': 0,
            'lr': 5e-4,
        }
    }


    run_tune(hparams)
