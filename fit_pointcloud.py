'''
2D Implicit Surface Fitting
'''
import torch
import pytorch_lightning as pl
from glob import glob
import os
import nbf, network, data, diff_utils

import geom_utils as gu
from   geom_loss import eikonal_loss, off_surface_loss

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shape", type = str, help="shape_name")
parser.add_argument("-n", "--number-points", help="number_points")
parser.add_argument("-p", "--pretrain", help="pretrained_weights")

args = parser.parse_args()
config = vars(args)


def main(shape, num_point, pretrained_ckpt):
    print(shape)
    print(num_point)
    # Use a seed for reproducibility
    seed = 2
    pl.seed_everything(seed)

    # Specify the data

    # Generic dataloader / generator for pointclouds
    pointset_data = data.PointSetData(  pointset = {'shape_name': str(shape), # choose from predefined shapes or provide file path and set load_ext to True
                                                    'shape_opts': {'num_points': num_point, 
                                                                'normal_constraints': 'grad', # normal_constraints: 'pseudo' or 'grad
                                                                }, 
                                                    'load_ext': False, 
                                                    'external_file_opts': None},

                                        training_point_opts = {'num_points': 1000})


    # Optional: pass a pretrained model
    if pretrained_ckpt != 'None':
        pretrained_ckpt = max(glob('lightning_logs/version_{}/checkpoints/*.ckpt'.format(pretrained_ckpt)), key=os.path.getmtime)
        print(pretrained_ckpt)
    else:
        pretrained_ckpt = None
   



    basis = network.init_basis("hypernet_kernel_geom", pointset_data, kernel_type = 'Gaussian', init_sigma=50)
    # Specify training parameters
    max_epochs = 50
    smooth_exp = 0



    # Initialize NBF
    if pretrained_ckpt is None:
        NBF = nbf.NeuralBasisField(basis=basis,
            constraint_pt=pointset_data.constraint_pt, constraint_value=pointset_data.constraint_value,
            k_list=pointset_data.k_list, op_list=pointset_data.op_list, smooth_exp=smooth_exp, 
            cond_num_weight=1e-1, lr = 1e-5, use_regression=False, loss_fn = eikonal_loss)
    else:
        NBF = nbf.NeuralBasisField.load_from_checkpoint(pretrained_ckpt, basis=basis,
            constraint_pt=pointset_data.constraint_pt, constraint_value=pointset_data.constraint_value,
            k_list=pointset_data.k_list, op_list=pointset_data.op_list, smooth_exp=smooth_exp, 
            cond_num_weight=1e-1, lr = 1e-5, use_regression=False, loss_fn = eikonal_loss)

        pointset_data = data.PointSetData(  pointset = {'shape_name': str(shape), # choose from predefined shapes or provide file path and set load_ext to True
                                                        'shape_opts': {'num_points': num_point, 
                                                                    'normal_constraints': 'grad', # normal_constraints: 'pseudo' or 'grad'
                                                                    }, 
                                                    'load_ext': False, 
                                                    'external_file_opts': None},

                                        training_point_opts = {'num_points': 1000, 'close_circle' : True})
            # Reinit data module
        # pointset_data = data.PointSetData(pointset_data.type)
        NBF.constraint_value = pointset_data.constraint_value
        NBF.constraint_pt = pointset_data.constraint_pt

    clbks = [pl.callbacks.ModelCheckpoint(save_weights_only=False, monitor='train/condition_number', mode='min')]
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1, callbacks=clbks, log_every_n_steps=1, num_sanity_val_steps=-1)
    # trainer.validate(NBF, pointset_data)  # check the mse at 0th epoch


    # Visualize the initial reconstruction
    with torch.inference_mode(False):
        gu.twod_visualisation_panel(NBF, pointset_data)
        gu.paper_visualisation(NBF, pointset_data, 'final_paper_{}_{}_prior'.format(shape, num_point))

    # Train the NBF
    trainer.fit(NBF, pointset_data)

    with torch.inference_mode(False):
        gu.twod_visualisation_panel(NBF, pointset_data)
        gu.paper_visualisation(NBF, pointset_data, 'final_paper_{}_{}_post'.format(shape, num_point))


if __name__ == '__main__':
    print('Running Config: {}'.format(config))
    shape = str(config['shape']).strip()
    number = int(config['number_points'])
    pretrained = str(config['pretrain']).strip()
    main(shape, number, pretrained)