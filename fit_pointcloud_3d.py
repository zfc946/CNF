import torch
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import geom_utils as gu
import data, helper
import network
import nbf
from geom_loss import eikonal_loss

# Use a seed for reproducibility
seed = 2
pl.seed_everything(seed)



### [Shapes Names Used in Paper] 
# (1) spot, 
# (2) max-planck, 
# (3) horse, 
# (4) Nefertiti

shape_name = 'spot'


pointset_data = data.PointSetData(pointset = {'shape_name': '{}'.format(shape_name), # load external from common-3d-test-models (https://github.com/alecjacobson/common-3d-test-models)
                                              'shape_opts': {'num_points':10000, 
                                                             'normal_constraints': 'pseudo', # normal_constraints: 'pseudo' or 'grad'
                                                             'pseudo_eps': 1e-2,}, 
                                              'load_ext': True, 
                                              'external_file_dir': 'github_repo' }, # choose from 'github_repo' or provide local dir'local'},
                                    
                                    training_point_opts = {'num_points': 1}) # We do not train the 3D Pointcloud evaluation example.

# Calculate the average distance between points in the pointcloud
average_distance = gu.nn_distace(pointset_data.constraint_pt.detach().cpu().numpy())

# # Specify the basis function
basis = network.init_basis("local_kernel", pointset_data, kernel_type = "Gaussian", init_sigma= 50)

# Specify training parameters
max_epochs = 300
smooth_exp = 2


NBF = nbf.NeuralBasisField_LocalKernelsWeighted(basis=basis,
        constraint_pt=pointset_data.constraint_pt, constraint_value=pointset_data.constraint_value,
        k_list=pointset_data.k_list, op_list=pointset_data.op_list, loss_singularity_fn='cond', 
        smooth_exp=smooth_exp, lr = 1e-5, use_regression=True, loss_fns= [], points = pointset_data.points, 
        diff_order = pointset_data.diff_order, avg_dist = average_distance,)

    
clbks = [pl.callbacks.ModelCheckpoint(save_weights_only=False, monitor='train/condition_number', mode='min')]
trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1, callbacks=clbks, log_every_n_steps=1, num_sanity_val_steps=-1) #, gradient_clip_val=0.5)


with torch.inference_mode(False):
    evaluated_field = gu.eval_field(NBF, bounds = None)
    meshed_isosurface = gu.generate_mesh(evaluated_field)
    gu.save_mesh(meshed_isosurface, 'meshes/{}.ply'.format(shape_name))
    gu.visualize_mesh(meshed_isosurface)
