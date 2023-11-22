'''
Script defines geometry specific loss functions for 
soft constraints with signed distance functions.
'''

import torch
import diff_utils
import matplotlib.pyplot as plt
import numpy as np



def eikonal_loss(batch, model, weight = 1, point_opts = {'num_points': 1000, 'domain_bounds': [-2, 2]}):
    '''
    Geometry specific loss function for regularising
    a field to have the signed distance property. 
    Function tries to enforce:

                |∇ φ(x)|^2 = 1  ∀ x ∈ Ω             (1)

    via a random sampling of points in the domain Ω.

    [Note : The domain Ω is assumed to be bounded on [-2, 2]^2]
    '''

    input_dim = model.basis.dim[0]
    num_points = point_opts['num_points']
    domain = point_opts['domain_bounds']


    # generate a meshgrid of points to enforce the eikonal term on.
    
    background_points = torch.rand((num_points, input_dim), 
                                        device=model.device) * (domain[1] - domain[0])  + domain[0]
    
    background_points.requires_grad = True
    sdf = model(background_points)
    # print(sdf)
    nabla_sdf = []
    for xi in range(input_dim):
        d_dxi_sdf = diff_utils.compute_op('f_{x_' + str(xi) + '}', sdf , background_points) # Compute ∂φ/∂xi ~ n_xi
        nabla_sdf.append(torch.squeeze(d_dxi_sdf))

    nabla_sdf = torch.stack(nabla_sdf, dim=-1)
    loss = weight * ((torch.linalg.norm(nabla_sdf, dim=-1) - 1.0)**2).mean()  # Minimise(|∇ φ(x)|^2 - 1)^2  
    return loss



def off_surface_loss(batch, model, weight = 1, exp_weight = -1e2, point_opts = {'num_points': 1000, 'domain_bounds': [-2, 2]}):
    '''
    Function used to penalise the field at points that are not
    on the isosurface. 
    See https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/loss_functions.py#L229    
    '''
    input_dim = model.basis.dim[0]
    num_points = point_opts['num_points']
    domain = point_opts['domain_bounds']


    # generate a meshgrid of points to enforce the eikonal term on.
    
    background_points = torch.rand((num_points, input_dim), 
                                        device=model.device) * (domain[1] - domain[0])  + domain[0]
    
    background_points.requires_grad = True
    sdf = model(background_points)

    off_surf_loss = torch.exp(exp_weight * torch.abs(sdf))

    return  weight * off_surf_loss.mean()




