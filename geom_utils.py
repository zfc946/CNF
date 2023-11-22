'''
Geometry utility functions
'''

import numpy as np
from skimage import measure
import open3d as o3d

import torch
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from scipy.spatial import distance
import os 
from tqdm import tqdm

# Setup matplotlib style
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

plt.rcParams["font.family"] = "cmr10"
matplotlib.rcParams["axes.formatter.use_mathtext"] = True

o3d_window_name = 'Neural Kernel Fields | 3D Surface Reconstruction'




def pointset_bounds(points):

    a = points.detach().cpu().numpy()
    bounds = np.array([[np.min(a[:,0]), np.max(a[:,0])],
                       [np.min(a[:,1]), np.max(a[:,1])],
                       [np.min(a[:,2]), np.max(a[:,2])]])

    return bounds

def eval_field(NBF, grid_size = 160, batch_size = 1, bounds = None):
    '''
    Function used to generate point values from a neural field 
    on a 3D grid.
    '''
    if bounds is None:
        bounds = np.array([[-1.2, 1.2], 
                           [-1.2, 1.2], 
                           [-1.2, 1.2]])



    eval_x = torch.linspace(bounds[0][0], bounds[0][1], grid_size, requires_grad=True)
    eval_y = torch.linspace(bounds[1][0], bounds[1][1], grid_size, requires_grad=True)
    eval_z = torch.linspace(bounds[2][0], bounds[2][1], grid_size, requires_grad=True)
    mesh_x,mesh_y, mesh_z= torch.meshgrid([eval_x, eval_y, eval_z], indexing='xy')
    eval_points = torch.vstack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()]).T
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NBF = NBF.to(device)
   
    len_train = len(eval_points)
    batches = int(len_train / batch_size)
    batch_id = range((batches))
    out = []
    with torch.inference_mode(False):
        print('Evaluating field...')
        for i in tqdm(batch_id):
            if i == batches - 1:
                x = eval_points[i*batch_size:,:].to(device)
                y = NBF(x)
                out.append(y.detach().cpu().numpy())
            else:
                x = eval_points[i*batch_size:(i+1)*batch_size,:].to(device)
                y = NBF(x)
                out.append(y.detach().cpu().numpy())
    
    field_eval = np.vstack(out)
    field_eval = np.reshape(field_eval, (grid_size, grid_size, grid_size))
    return field_eval

def generate_mesh(sdf: np.ndarray, ):
    '''
    Function performs marching cubes on an input volume 
    and returns the vertices, faces, and normals of the mesh.

    '''

    verts, faces, normals, values = measure.marching_cubes(sdf, level = 0.0, spacing=(2.4/99, 2.4/99, 2.4/99))

    # Create an open3d mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(verts) - np.array([1.2, 1.2, 1.2]))
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.compute_triangle_normals()

    return mesh


def nn_distace(points):
    out = distance.cdist(points, points)
    out[out == 0] = 1
    dists = np.min(out, axis = 1)
    avg = np.mean(dists)
    return avg


def generate_pointset(points):

    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    
    points = points.astype(np.float32)
    pointset_obj = o3d.geometry.PointCloud()
    pointset_obj.points = o3d.utility.Vector3dVector(points)
    
    return pointset_obj

def visualize_mesh_and_pointset(mesh, pcd):
    '''
    Function visualizes an open3d mesh object
    '''
    # mesh = mesh.translate(-mesh.get_center() + pcd.get_center())
    o3d.visualization.draw_geometries([mesh,pcd],  window_name = o3d_window_name, mesh_show_back_face = True, mesh_show_wireframe = True)
    # o3d.visualization.RenderOption.point_size = 1000.0
    return None


def visualize_mesh(mesh):
    '''
    Function visualizes an open3d mesh object
    '''
    
    o3d.visualization.draw_geometries([mesh], window_name = o3d_window_name, mesh_show_back_face = True, mesh_show_wireframe = True)
    
    return None

def construct_vis():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    return vis


def save_mesh(mesh, filename):
    '''
    Function saves an open3d mesh object to a .ply file
    '''
    o3d.io.write_triangle_mesh(filename, mesh)

    return None


def gen_test_sdf(delta = 0.01):
    '''
    Function generates a test signed distance field of the armadillo mesh.
    ''' 
    step = int(1.0 / delta)


    # Load the armadillo mesh from open3d
    armadillo_data = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(armadillo_data.path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a raycasting scene -> this is the object that will be used to 
    # compute the signed distance
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  

    # Find the bounds of the mesh 
    min_bound = mesh.vertex.positions.min(0).numpy()
    max_bound = mesh.vertex.positions.max(0).numpy()
    
    # Produce a mesh grid of the query volume.
    x, y, z = np.meshgrid(np.linspace(min_bound[0], max_bound[0], step),
                          np.linspace(min_bound[1], max_bound[1], step), 
                          np.linspace(min_bound[2], max_bound[2], step), indexing='xy')
    
    # Flatten the mesh grid and convert to float32 for query
    flattened_volume = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    flattened_volume = flattened_volume.astype(np.float32)

    # Compute the signed distance
    signed_distance = scene.compute_signed_distance(flattened_volume)

    # Reshape the signed distance to the original mesh grid shape
    signed_distance_volume  = signed_distance.numpy().reshape(step, step, step)

    return signed_distance_volume


def twod_visualisation_panel(NBF, data, save_fig = None, limits = [(-2.2, 2.2),(-2.2, 2.2)], grid_size = 50):
    '''
    Helper function to visualise the 2D SDF
    produced by the NBF network.
    '''
    device = torch.device('cuda:0')
    NBF = NBF.to(device)
    constraint_pt = data.constraint_pt
    train_pt = data.train_pt
    tmp_x, tmp_y = torch.linspace(limits[0][0], limits[0][1], grid_size, requires_grad=True), torch.linspace(limits[1][0], limits[1][1], grid_size, requires_grad=True)
    X,Y = torch.meshgrid([tmp_x, tmp_y], indexing='xy')
    eval_points = torch.vstack([X.ravel(), Y.ravel()]).T
    
    eval_points = eval_points.to(device)
    eval_points.requires_grad_(True)
    
    output = NBF(eval_points)
    output = output.reshape(grid_size, grid_size)

    output_grad = torch.autograd.grad(output, eval_points, grad_outputs=torch.ones_like(output, requires_grad = True), retain_graph=True)[0]
    norm = torch.norm(output_grad, dim = 1)

    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    
    axs[0,0].contour( X.detach().numpy(), Y.detach().numpy(), output.detach().cpu().numpy(), levels=0, colors = 'black', zorder=20)
    axs[0,0].legend(loc = 'upper right')
    axs[0,0].scatter(constraint_pt[:,0].detach().cpu().numpy(), constraint_pt[:, 1].detach().cpu().numpy(), c='blue', alpha = 0.2, label = 'Constraint Points', zorder=10)
    # axs[0,0].scatter(train_pt[:,0].detach().cpu().numpy(), train_pt[:, 1].detach().cpu().numpy(), c='r', alpha = 0.1, label = 'Train Points', zorder = 0)
    axs[0,0].legend(loc = 'upper right')
    axs[0,0].set_title(' $\Phi(x) = 0$')

    field = axs[0,1].contourf(X.detach().numpy(), Y.detach().numpy(), output.detach().cpu().numpy(), levels=10, cmap = 'RdYlGn')
    axs[0,1].scatter(constraint_pt[:,0].detach().cpu().numpy(), constraint_pt[:, 1].detach().cpu().numpy(), c='black', alpha = 0.1, label = 'Constraint Points')
    axs[0,1].set_title(' $\Phi(x)$')
    plt.colorbar(field, ax = axs[0,1])


    axs[1,0].quiver(X.detach().numpy(),Y.detach().numpy(), output_grad[:,0].detach().cpu().numpy(), output_grad[:,1].detach().cpu().numpy())
    axs[1,0].set_title(r'$\nabla \Phi(x)$')


    norm_grad = axs[1,1].imshow(norm.reshape(grid_size, grid_size).detach().cpu().numpy(), origin='lower')
    axs[1,1].set_title(r'$|\nabla \Phi(x)|$')
    plt.colorbar(norm_grad, ax = axs[1,1])
    plt.show()


def obj2pointset(local_path, shape_opt_dict):
    '''
    Util function called in data generation to convert an obj file (already downloaded)
    to a point set, with the number of points specified in the shape_opt_dict.
    '''


    mesh = o3d.io.read_triangle_mesh(local_path)
    mesh.compute_vertex_normals()
   
    sample_points = mesh.sample_points_uniformly(number_of_points = shape_opt_dict['num_points'])
    sample_points_high = mesh.sample_points_uniformly(number_of_points = shape_opt_dict['num_points']*10)
    points = np.asarray((sample_points.points))
    point_normals = (np.asarray(sample_points.normals))
    max = np.max(points)
    points = points/max
    sample_points_high = np.asarray((sample_points_high.points))
    sample_points_high = sample_points_high/max
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')    
    ax.quiver(points.T[0], points.T[1], points.T[2], point_normals.T[0], point_normals.T[1], point_normals.T[2], length=0.1, normalize=True)
    plt.show()

    return points, point_normals, sample_points_high 


def paper_visualisation(NBF, data, shapename, limits = [(-3, 3),(-3, 3)], grid_size = 60):
    MYDIR = ("fig_constructor")
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        
    current_working_directory = os.getcwd()
    output_dir = os.path.join(current_working_directory, MYDIR)
    device = torch.device('cuda:0')
    NBF = NBF.to(device)
    constraint_pt = data.constraint_pt
    constraint_val = data.constraint_value
    train_pt = data.train_pt
    tmp_x, tmp_y = torch.linspace(limits[0][0], limits[0][1], grid_size, requires_grad=True), torch.linspace(limits[1][0], limits[1][1], grid_size, requires_grad=True)
    X,Y = torch.meshgrid([tmp_x, tmp_y], indexing='xy')
    eval_points = torch.vstack([X.ravel(), Y.ravel()]).T
    
    eval_points = eval_points.to(device)
    eval_points.requires_grad_(True)
    
    output = NBF(eval_points)
    output = output.reshape(grid_size, grid_size)

    output_grad = torch.autograd.grad(output, eval_points, grad_outputs=torch.ones_like(output, requires_grad = True), retain_graph=True)[0]
    norm = torch.norm(output_grad, dim = 1)
    # print(output)
    np.save(os.path.join(output_dir, '{}_field.npy'.format(shapename)), np.array([X.detach().numpy(), Y.detach().numpy(), output.detach().cpu().numpy()]))
    np.save(os.path.join(output_dir, '{}_constraints.npy'.format(shapename)), np.array([constraint_pt[:,0].detach().cpu().numpy(), constraint_pt[:, 1].detach().cpu().numpy()]) )
    np.save(os.path.join(output_dir, '{}_constraints_val.npy'.format(shapename)), np.array([constraint_val.detach().cpu().numpy()]) )