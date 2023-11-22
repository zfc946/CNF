import numpy as np
import torch
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
import cv2
import torchvision
import os
from sklearn.neighbors import KDTree

def gen_toy_data(f, df=None, is_ODE=True, constraint_sr=128, training_sr=1000):
	# training_pt = torch.tensor([[1], [2], [3], [4], [5], [6]], dtype=torch.float, device='cuda')
	# training_value = torch.tensor([[1], [4], [9], [16], [25], [36]], dtype=torch.float, device='cuda')
	# constraint_pt = torch.tensor([[0], [8]], dtype=torch.float, device='cuda')
	# constraint_value = torch.tensor([[0], [64]], dtype=torch.float, device='cuda')

	# training_pt = torch.tensor([[1], [2], [3], [5], [6]], dtype=torch.float)
	# training_value = torch.tensor([[1], [4], [9], [25], [36]], dtype=torch.float)
	# constraint_pt = torch.tensor([[0], [4], [8]], dtype=torch.float)
	# constraint_value = torch.tensor([[0], [16], [64]], dtype=torch.float)

	# training_pt = torch.tensor([[1, 1], [2, 2], [3, 3], [5, 5], [6, 6]], dtype=torch.float)
	# training_value = torch.tensor([[1], [4], [9], [25], [36]], dtype=torch.float)
	# constraint_pt = torch.tensor([[0, 0], [4, 4], [8, 8]], dtype=torch.float)
	# constraint_value = torch.tensor([[0], [16], [64]], dtype=torch.float)

	# training_pt = torch.tensor([[1], [2], [3], [5], [6]], dtype=torch.float)
	# training_value = torch.tensor([[1, 1], [4, 4], [9, 9], [25, 25], [36, 36]], dtype=torch.float)
	# constraint_pt = torch.tensor([[0], [4], [8]], dtype=torch.float)
	# constraint_value = torch.tensor([[0, 0], [16, 16], [64, 64]], dtype=torch.float)

	# training_pt = torch.tensor([[1, 1], [2, 2], [3, 3], [5, 5], [6, 6]], dtype=torch.float)
	# training_value = torch.tensor([[1, 1], [4, 4], [9, 9], [25, 25], [36, 36]], dtype=torch.float)
	# constraint_pt = torch.tensor([[0, 0], [4, 4], [8, 8]], dtype=torch.float)
	# constraint_value = torch.tensor([[0, 0], [16, 16], [64, 64]], dtype=torch.float)

	# training_pt = torch.tensor([[]])
	# training_value = torch.tensor([[]])
	# constraint_pt = torch.unsqueeze(torch.linspace(0, 10, steps=30), 1)
	# constraint_value = torch.sin(0.5*constraint_pt)

	# training_pt = torch.unsqueeze(torch.linspace(2, 10, steps=1000), 1)
	# training_value = torch.zeros_like(training_pt)
	# constraint_pt = torch.unsqueeze(torch.linspace(2, 10, steps=128), 1)
	# constraint_value = f(constraint_pt)
	# k_list = []
	# op_list = []

	if is_ODE:
		# higher order constaints
		training_pt = torch.unsqueeze(torch.linspace(2, 10, steps=training_sr), 1)
		training_value = torch.zeros_like(training_pt)
		constraint_pt = torch.unsqueeze(torch.linspace(2, 10, steps=2), 1)
		constraint_value = f(constraint_pt)
		constraint_pt_grad = torch.unsqueeze(torch.linspace(2.1, 9.9, steps=constraint_sr, requires_grad = True), 1)
		constraint_value_grad = df(constraint_pt_grad)
		constraint_pt = torch.cat((constraint_pt, constraint_pt_grad))
		constraint_value = torch.cat((constraint_value, constraint_value_grad))
		k_list = [2]
		op_list = ["f_{x_0}"]

		d_in = constraint_pt.shape[1]
		d_out = constraint_value.shape[1]

		return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list
	
	else:
		training_pt = torch.unsqueeze(torch.linspace(2, 10, steps=training_sr), 1)
		training_value = f(training_pt)
		constraint_pt = torch.unsqueeze(torch.linspace(2, 10, steps=constraint_sr), 1)
		constraint_value = f(constraint_pt)
		k_list = []
		op_list = []

		d_in = constraint_pt.shape[1]
		d_out = constraint_value.shape[1]

		return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list

'''
Reference: Eq. 6 from https://arxiv.org/pdf/2210.07182.pdf
Code: https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/data_gen_NLE/AdvectionEq/advection_exact_Hydra.py
'''
def gen_advection_pdebench(f, op_f = None, beta=0.1, xL=0.0, xR=1.0, num_x=16, t0=0.0, tT=2.0, num_t=16, perturb=0, u0_shift=0):
	x = np.linspace(xL, xR, num_x, dtype=np.float32)
	t = np.linspace(t0, tT, num_t, dtype=np.float32)

	def set_function(x, t, beta, u0_shift):
		# return np.sin(2.*np.pi*(x - beta*t))
		return f(x, t, beta, u0_shift)

	x_grid, t_grid = np.meshgrid(x, t)
	u = set_function(torch.tensor(x_grid), torch.tensor(t_grid), beta, u0_shift)

	training_pt = torch.stack(torch.meshgrid(torch.linspace(xL, xR, num_x), torch.linspace(t0, tT, num_t), indexing='xy'), dim=-1).flatten(end_dim=1)
	training_value = torch.zeros_like(training_pt)

	constraint_pt_0 = torch.tensor(np.stack((x_grid[0], t_grid[0]), axis=-1))
	constraint_value_0 = torch.tensor(u[0]).unsqueeze(axis=-1)
	constraint_pt_1 = torch.tensor(np.stack((x_grid[1:], t_grid[1:]), axis=-1), requires_grad=True).flatten(end_dim=1)
	constraint_pt_1 = constraint_pt_1 + torch.randn_like(constraint_pt_1)*perturb
	if op_f == None:
		constraint_value_1 = torch.zeros(constraint_pt_1.shape[0], 1)
	else:
		constraint_value_1 = op_f(x_grid, t_grid, beta).flatten()[...,None]
	constraint_pt = torch.cat((constraint_pt_0, constraint_pt_1))
	constraint_value = torch.cat((constraint_value_0, constraint_value_1))
	constraint_gt = torch.cat((constraint_value_0, torch.tensor(u).flatten(end_dim=1).unsqueeze(1)))

	k_list = [constraint_pt_0.shape[0]]
	op_list = [f'f_{{x_1}} + {beta}f_{{x_0}}']
	d_in = constraint_pt_0.shape[1]
	d_out = constraint_value_0.shape[1]

	return constraint_pt, constraint_value, constraint_gt, training_pt, training_value, d_in, d_out, k_list, op_list

def gen_image(path, as_gray=False, resize_dim=None):
	'''
	resize_dim: tuple of the new shape (w, h, c)
	'''
	img =  cv2.imread(path, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR)
	num_channel = img.shape[2] if not as_gray else 1
	if resize_dim is not None:
		img = cv2.resize(img, resize_dim)
	img = torch.tensor(img, dtype=torch.float)
	xs = torch.linspace(0, img.shape[0]-1, steps=img.shape[0], dtype=torch.float)
	ys = torch.linspace(0, img.shape[1]-1, steps=img.shape[1], dtype=torch.float)
	x, y = torch.meshgrid(xs, ys)
	x = torch.unsqueeze(x, 2)
	y = torch.unsqueeze(y, 2)
	xy = torch.cat((x, y), 2)
	constraint_pt = torch.reshape(xy, (-1, 2))
	constraint_value = torch.reshape(img, (-1, num_channel))
	training_pt = torch.clone(constraint_pt)
	training_value = torch.clone(constraint_value)
	d_input = constraint_pt.shape[1]
	d_out = num_channel
	k_list = []
	op_list = []

	return constraint_pt, constraint_value, training_pt, training_value, d_input, d_out, k_list, op_list

def gen_sphere_samples(d_samples = 100, d_samples_train = 100000):

	def sample_spherical(npoints, ndim=3):
		vec = np.random.randn(ndim, npoints)
		vec /= np.linalg.norm(vec, axis=0)
		return vec

	# generate constraint points
	xi, yi, zi = sample_spherical(d_samples)
	xi = torch.tensor([xi], dtype=torch.float)
	yi = torch.tensor([yi], dtype=torch.float)
	zi = torch.tensor([zi], dtype=torch.float)

	rand_scale = torch.rand(d_samples, 1)
	samples = torch.cat((xi, yi, zi), 0)
	constraint_pt = torch.swapaxes(samples, 0, 1)*rand_scale
	constraint_value = torch.zeros(d_samples, 1)*rand_scale - 1

	# xi, yi, zi = sample_spherical(d_samples)
	# xi = torch.tensor([xi], dtype=torch.float)
	# yi = torch.tensor([yi], dtype=torch.float)
	# zi = torch.tensor([zi], dtype=torch.float)

	# samples = torch.cat((xi, yi, zi), 0)
	# constraint_pt2 = torch.swapaxes(samples, 0, 1)*1.5
	# constraint_value2 = torch.ones(d_samples, 1)*1.5 - 1

	# constraint_pt = torch.cat((constraint_pt, constraint_pt2), 0)
	# constraint_value = torch.cat((constraint_value, constraint_value2), 0)

	# generate training points
	xi, yi, zi = sample_spherical(d_samples_train)
	xi = torch.tensor([xi], dtype=torch.float)
	yi = torch.tensor([yi], dtype=torch.float)
	zi = torch.tensor([zi], dtype=torch.float)

	rand_scale = torch.rand(d_samples_train, 1)
	samples = torch.cat((xi, yi, zi), 0)
	training_pt = torch.swapaxes(samples, 0, 1)*rand_scale
	training_value = torch.ones(d_samples_train, 1)*rand_scale - 1

	return constraint_pt, constraint_value, training_pt, training_value, 3

def plot_1d(x, y, y_hat, epoch, mse_fn = torch.nn.functional.mse_loss, save_image = False, logger = None, plot_name = '1d.png'):
	plt.clf()
	plt.plot(x.squeeze().detach().cpu().numpy(), y.squeeze().detach().cpu().numpy(), label='Ground_truth')
	plt.plot(x.squeeze().detach().cpu().numpy(), y_hat.squeeze().detach().cpu().numpy(), label='Fitting')
	plt.legend()
	plt.title(f'Fitting error (MSE)\n{mse_fn(y_hat, y.squeeze(0)).item():.9f} at epoch {epoch}')

	if save_image:
		plt.savefig(plot_name)
		print("figure saved as ", plot_name)

	if logger is not None:
		log_plt_figure(plt.gcf(), 'Fitting', logger, epoch)

def visualize_advection(fn, beta=0.1, xlims=(0, 1), tlims=(0, 2), u0_shift=0):
	def set_function(x, t, beta, u0_shift):
		return np.sin(2.*np.pi*(x - beta*t)) + u0_shift

	nx, nt = 32, 32
	x = np.linspace(*xlims, nx, dtype=np.float32)
	t = np.linspace(*tlims, nt, dtype=np.float32)
	x_grid, t_grid = np.meshgrid(x, t)
	test_pt = torch.tensor(np.stack((x_grid, t_grid), axis=-1), device=fn.device)

	u = set_function(x_grid, t_grid, beta, u0_shift)

	fig, ax = plt.subplots(1, 3)
	# gt, = ax[0].plot(x, u[0].squeeze())
	ax[0].imshow(u)
	u_hat = fn(test_pt.reshape(-1, 2)).detach().squeeze().cpu().numpy().reshape(nx, nt)
	# pred, = ax[1].plot(x, u_hat)
	ax[1].imshow(u_hat)
	ax[2].imshow((u-u_hat)**2, vmax=5, vmin=0)
 
	fig.suptitle(r'Advection $\beta = 0.1$')
 
	# Reference for metrics: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py
	def compute_mse(u, test_pt):
		test_pt_flat = test_pt.reshape((test_pt.shape[0]*test_pt.shape[1], test_pt.shape[2]))
		u_hat = fn(test_pt_flat).detach().squeeze().cpu().numpy().reshape(nx, nt)
		err_mean = np.sqrt(np.mean((u - u_hat)**2, axis=0))
		err_rmse = err_mean.mean()
		return err_rmse

	def compute_relative_error(u, test_pt):
		test_pt_flat = test_pt.reshape((test_pt.shape[0]*test_pt.shape[1], test_pt.shape[2]))
		u_hat = fn(test_pt_flat).detach().squeeze().cpu().numpy().reshape(nx, nt)
		err_mean = np.sqrt(np.mean((u - u_hat)**2, axis=0))
		nrm = np.sqrt(np.mean(u**2, axis=0))
		err_nrmse = (err_mean/nrm).mean()
		return err_nrmse

	print(f"RMSE = {compute_mse(u, test_pt)}")
	print(f"Relative error = {compute_relative_error(u, test_pt)}")

	# TODO: better visualisation from NeurIPS submission

	ax[0].set_title('Ground truth')
	ax[1].set_title('Prediction')

	# Animate the plot
	# def animate(i):
	# 	gt.set_ydata(u[i].squeeze())
	# 	u_hat = fn(test_pt[i]).detach().squeeze().cpu().numpy()
	# 	pred.set_ydata(u_hat)
	# 	return gt, pred

	# ani = animation.FuncAnimation(fig, animate, frames=nt, interval=50, blit=False, repeat_delay=1000)

	# writer = animation.PillowWriter(fps=15, bitrate=1800)
	# ani.save("movie_advection.gif", writer=writer)
	# print("Animation saved")
	plt.savefig('advection_vis.png')

def log_plt_figure(fig, label, logger, epoch):
	fig.canvas.draw()
	image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	logger.experiment.add_image(label, image_from_plot, dataformats='HWC', global_step=epoch)
	fig.clear()

def plot_image(data, model, plot_name='res.png'):
    x = data.val_dataset.pt
    img_gt = data.val_dataset.value
    img_fit = model(x)
    mse = torch.nn.functional.mse_loss(img_fit, img_gt).item()
    img_fit = img_fit.detach().numpy()
    
    n_channel = img_fit.shape[1]
    n = int(np.sqrt(img_fit.shape[0]))
    img_fit = np.resize(img_fit, (n, n, n_channel))
    img_gt = np.resize(img_gt, (n, n, n_channel))
    is_gray = (img_gt.shape[2] == 1)

    # create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # show the first image in the first subplot
    ax[0].imshow(img_gt/255, vmin=0, vmax=1, cmap = 'gray' if is_gray else None)
    ax[0].set_title('Ground Truth')

    # show the second image in the second subplot
    ax[1].imshow(img_fit/255, vmin=0, vmax=1, cmap = 'gray' if is_gray else None)
    ax[1].set_title('Fitting')

    # set the title of the entire figure
    fig.suptitle(f'Fitting error (MSE)\n{mse:.10f} at epoch {model.current_epoch}')

    plt.savefig(plot_name)
    print("figure saved as ", plot_name)

def plot_implicit_(fn, bbox=(-2.5,2.5)):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    plt.show()
    plt.savefig('implicit.png')

def plot_implicit(implicit_fn):

	def fn(xi, yi, zi):
		shape_in = xi.shape
		x = np.reshape(xi, (-1))
		y = np.reshape(yi, (-1))
		z = np.reshape(zi, (-1))
		if x.shape == (1,):
			x = np.ones_like(y)*x
			shape_in = yi.shape
		elif y.shape == (1,):
			y = np.ones_like(x)*y
		elif z.shape == (1,):
			z = np.ones_like(x)*z
		x = torch.tensor([x], dtype=torch.float)
		y = torch.tensor([y], dtype=torch.float)
		z = torch.tensor([z], dtype=torch.float)
		xyz = torch.cat((x, y, z), 0)
		xyz = torch.swapaxes(xyz, 0, 1)

		sd = implicit_fn(xyz).detach().numpy()

		return np.reshape(sd, shape_in)

	plot_implicit_(fn)

def plot_parametric_surface(fn, period=(1,1)):
	fig = plt.figure()
	num = 20
	u = np.linspace(0, period(0), endpoint=True, num=num)
	v = np.linspace(0, period(1), endpoint=True, num=num)
	u, v = np.meshgrid(u, v)
	u, v = u.flatten(), v.flatten()

	uv = torch.tensor(np.concatenate([np.expand_dims(u, 1),np.expand_dims(v, 1)],1), dtype=torch.float).cuda()
	xyz = fn(uv)
	x = xyz[:,0].cpu().detach().numpy()
	y = xyz[:,1].cpu().detach().numpy()
	z = xyz[:,2].cpu().detach().numpy()

	tri = mtri.Triangulation(u, v)
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
	ax.set_zlim(-2, 2)
	plt.show()
	plt.savefig('explicit.png')

def	load_image(image_name):
	'''
	Args: image_name : str 
	'''
	cwd = os.getcwd()
	img_path = os.path.join(cwd, 'test_images', image_name + '.png')
	image = cv2.imread(img_path, cv2.IMREAD_COLOR)
	return image

def gen_constrain_pts(img, const_pattern, ratio=0.1, step=40):
	'''
	args: img shape = (H, W, C)
		  const_pattern  str
	'''
	img_shape = np.shape(img)

	convert_tensor = torchvision.transforms.ToTensor()
	image_tensor = convert_tensor(img) # convert image to torch tensor
	image_flat = image_tensor.flatten()  # flatten to make index the image easier.
										# NOTE: indexing should now only be done on 0 -> (511 * 511)
	if const_pattern == 'eye':
		eye_pix_x, eye_pix_y = torch.meshgrid(torch.arange(145,204,2), torch.arange(46,79, 2))
		tmp_constraint_pt = torch.vstack([eye_pix_x.ravel(), eye_pix_y.ravel()]).T

		constraint_pt = tmp_constraint_pt/(img_shape[1] - 1) # Normalise the coordinates to [0,1]

		x_long = tmp_constraint_pt.type(torch.long) 

		# print(type(img_shape))
		# print(type(x_long))
		ind = x_long[:,0] +img_shape[1]* x_long[:,1] 
		
		constraint_val = torch.unsqueeze(image_flat[ind], dim = 1) # obtain the grayscale image values
	
	elif const_pattern == 'handcrafted':
		tmp_x, tmp_y = (torch.Tensor([94, 430, 200, 220, 232]),
			       				     torch.Tensor([316,217, 70, 23, 285]))
		
		tmp_constraint_pt = torch.vstack([tmp_x.ravel(), tmp_y.ravel()]).T

		constraint_pt = tmp_constraint_pt/(img_shape[1] - 1)

		x_long = tmp_constraint_pt.type(torch.long) 

		# print(type(img_shape))
		# print(type(x_long))
		ind = x_long[:,0] +img_shape[1]* x_long[:,1] 
		
		constraint_val = torch.unsqueeze(image_flat[ind], dim = 1) # obtain the grayscale image values
	elif const_pattern == 'ratio':
		n_total = img.size
		ids = np.random.choice(np.arange(n_total), int(n_total*ratio))
		# constraint_pt = np.stack([ids // img.shape[0], ids % img.shape[0]]).T
		# constraint_val = image_tensor[:, constraint_pt[:,0], constraint_pt[:,1]].T
		# constraint_pt = constraint_pt / (np.array(img.shape[:2]) - 1)[None, :] # normalize
		# constraint_pt = torch.from_numpy(constraint_pt)

		tmp_pt = torch.from_numpy(np.stack([ids // img.shape[0], ids % img.shape[0]]).T)
		constraint_pt = tmp_pt/(torch.tensor(img_shape[:2]) - 1) # Normalise the coordinates to [0,1]
		x_long = tmp_pt.type(torch.long) 
		ind = x_long[:,0] +img_shape[1]* x_long[:,1] 

		constraint_val = torch.unsqueeze(image_flat[ind], dim = 1) # obtain the grayscale image values

	else:
		tmp_x, tmp_y = torch.meshgrid(torch.arange(0,512,step), torch.arange(0,512,step))
		tmp_pt = torch.vstack([tmp_x.ravel(), tmp_y.ravel()]).T
		constraint_pt = tmp_pt/(img_shape[1] - 1) # Normalise the coordinates to [0,1]
		x_long = tmp_pt.type(torch.long) 
		ind = x_long[:,0] +img_shape[1]* x_long[:,1] 

		constraint_val = torch.unsqueeze(image_flat[ind], dim = 1) # obtain the grayscale image values

	return constraint_pt, constraint_val

def gen_training_pts(img, step = 1):
	convert_tensor = torchvision.transforms.ToTensor()
	image_tensor = convert_tensor(img) # convert image to torch tensor
	image_flat = image_tensor.flatten()  # flatten to make index the image easier.
										 # NOTE: indexing should now only be done on 0 -> (511 * 511)
	image_shape = np.shape(img)
	pix_x, pix_y = torch.meshgrid(torch.arange(0,image_shape[0] ,step), torch.arange(0,image_shape[1], step))
	tmp_train_pt = torch.vstack([pix_x.ravel(), pix_y.ravel()]).T

	train_pt = tmp_train_pt/(image_shape[1] - 1)

	x_long = tmp_train_pt.type(torch.long) 
	ind = x_long[:,0] +image_shape[1]* x_long[:,1] 

	train_val = torch.unsqueeze(image_flat[ind], dim = 1) # obtain the grayscale image values
	return train_pt, train_val

def gen_Fermat(step = 64):
	constraint_pt = torch.tensor([[-1], [1]], dtype=torch.float)
	constraint_value = torch.tensor([[1, 2], [10, 10]], dtype=torch.float)
	training_pt = torch.unsqueeze(torch.linspace(-1, 1, steps=step), 1)
	# training_value = torch.zeros([step, 2])
	training_value = torch.stack([training_pt[:,0], torch.sin(training_pt[:,0])], 1)
	k_list = []
	op_list = []
	d_in = constraint_pt.shape[1]
	d_out = constraint_value.shape[1]
	return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list

def plot_Fermat(model, t, filename="optical_path.pdf", logger=None, epoch=None, travel_time=0):
	p = model(t).cpu().detach().numpy()
	plt.clf()
	plt.plot(p[:,0], p[:,1])
	plt.xlabel('x-axis')
	plt.ylabel('y-axis')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.title(f'Optical path\n Total travel time {travel_time:.4f} at epoch {epoch}')
 
	if logger is not None:
		log_plt_figure(plt.gcf(), 'Fitting', logger, epoch)
	else:
		plt.savefig(filename)


def get_field_points(field_points, constraint_pts, constraint_pts_normals):



	pts = constraint_pts.detach().cpu().numpy()
	pts_tree = KDTree(pts)

	field_points_query = field_points.detach().cpu().numpy()
	
	dist, ind = pts_tree.query(field_points_query, k=1)
	ind = torch.from_numpy(ind)
	
	
	distance = torch.from_numpy(dist)
	near_pts = constraint_pts[ind]
	near_pts = torch.squeeze(near_pts, dim = 1)


	
	nearest_normals = constraint_pts_normals[ind]
	nearest_normals = torch.squeeze(nearest_normals, dim = 1)
	# print(nearest_normals)
	out = torch.einsum('ij,ij->i', field_points - near_pts, nearest_normals)
	# print(out)
	out_bool = torch.where(out > 0, torch.ones_like(out), -torch.ones_like(out))


	return torch.einsum('ij,ij->i', torch.unsqueeze(out_bool, dim = 1), distance).float()



def gen_pointset_ndarray(points, point_normals, shape_opts, training_point_opts, points_high = None):
	
	constraint_pt_in = torch.from_numpy(points).float()
	constraint_pts_hd = torch.from_numpy(points_high).float()
	constraint_val = torch.unsqueeze(torch.zeros_like(constraint_pt_in[:, 0]), 1)
	constraint_pts_normals =  torch.from_numpy(point_normals).float()
	
	if shape_opts['normal_constraints'] == 'pseudo':
		eps = shape_opts['pseudo_eps']
		constraint_pts_1 = constraint_pt_in + eps * constraint_pts_normals
		constraint_pts_2 = constraint_pt_in - eps * constraint_pts_normals
		# constraint_pts_3 = constraint_pt_in + 1* constraint_pts_normals
		constraint_value_1 = torch.unsqueeze(torch.zeros_like(constraint_pt_in[:, 0]) + eps, 1)
		constraint_value_2 = torch.unsqueeze(torch.zeros_like(constraint_pt_in[:, 0]) - eps, 1)
		# constraint_value_3 = torch.unsqueeze(torch.zeros_like(constraint_pt_in[:, 0]) + 1, 1)
		constraint_pts = torch.cat([constraint_pt_in, constraint_pts_1, constraint_pts_2]) #constraint_pts_3])
		constraint_val = torch.cat([constraint_val, constraint_value_1, constraint_value_2])# constraint_value_3])
		constraint_pts = torch.cat([constraint_pt_in, constraint_pts_1, constraint_pts_2])
		constraint_val = torch.cat([constraint_val, constraint_value_1, constraint_value_2])
		diff_order = torch.zeros_like(constraint_val)
	elif shape_opts['normal_constraints'] == 'grad':
		constraint_pts_0 = constraint_pt_in.clone().requires_grad_(True)
		constraint_pts_1 = constraint_pt_in.clone().requires_grad_(True)
		constraint_pts_2 = constraint_pt_in.clone().requires_grad_(True)
		constraint_pts_3 = constraint_pt_in.clone().requires_grad_(True)
		
		constraint_value_1 = torch.unsqueeze((constraint_pts_normals[:, 0]), 1)
		constraint_value_2 = torch.unsqueeze((constraint_pts_normals[:, 1]), 1)
		constraint_value_3 = torch.unsqueeze((constraint_pts_normals[:, 2]), 1)

		constraint_value_0 = torch.zeros_like(constraint_value_1)
		constraint_pts = torch.cat([constraint_pts_0, constraint_pts_1, constraint_pts_2, constraint_pts_3])
		constraint_val = torch.cat([constraint_value_0, constraint_value_1, constraint_value_2, constraint_value_3])
		diff_order = torch.cat([torch.zeros_like(constraint_value_0), torch.ones_like(constraint_value_1), torch.ones_like(constraint_value_2) + 1,  torch.ones_like(constraint_value_3) + 2])

	tmp_x, tmp_y, tmp_z = torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10)
	X,Y, Z = torch.meshgrid([tmp_x, tmp_y, tmp_z], indexing='xy')
	field_points = torch.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

	field_values = get_field_points(field_points, constraint_pt_in, constraint_pts_normals)

	field_values = torch.unsqueeze(field_values, 1)

	# constraint_pts = torch.cat([constraint_pts, field_points])
	# constraint_val = torch.cat([constraint_val, field_values])
	# tmp_x, tmp_y, tmp_z = torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), torch.linspace(0, 1, 100)
	# X,Y, Z = torch.meshgrid([tmp_x, tmp_y, tmp_z], indexing='xy')

	training_pt = constraint_pts_hd
	training_value = torch.unsqueeze(torch.zeros_like(training_pt[:,0]), 1)

	d_in = 3
	d_out = 1
	k_list = []
	op_list = []

	return constraint_pts, constraint_val, training_pt, training_value, d_in, d_out, k_list, op_list, diff_order



def gen_pointset(shape, shape_opts, training_point_opts):
	'''
	Helper function to generate pointset for implicit function
	reconstruction.
	'''

	if shape == 'semi-circle':
		'''
		Construct a 2D semi-circle with N points.
		'''
		d_in = 2 # input dimension
		d_out = 1 # output dimension
		
		num_points = shape_opts['num_points']
		normal_constraints = shape_opts['normal_constraints']


		# Generate points on a semi-circle with N points
		theta = torch.linspace(0, torch.pi, num_points, requires_grad = True)
		constraint_pt = torch.vstack([torch.cos(theta), torch.sin(theta)])
		constraint_value = torch.unsqueeze(torch.zeros_like(theta) , 1)

		# Generate normal constraints
		d_d_theta = torch.vstack([-torch.sin(theta),  torch.cos(theta)])
		if normal_constraints == 'grad':
			constraint_pt_1 = constraint_pt.clone() #+ 0.001 * d_d_theta.clone()
			grad_x_val = torch.unsqueeze(-torch.sin(theta), 1)

			constraint_pt_2 = constraint_pt.clone() #- 0.001 * d_d_theta.clone()
			grad_y_val = torch.unsqueeze(torch.cos(theta), 1)
			
			k_list = [int(num_points), 2 * int(num_points)]
			op_list = ["f_{x_0}", "f_{x_1}"]

		elif normal_constraints == 'pseudo':
			eps = shape_opts['pseudo_eps']
			constraint_pt_1= constraint_pt.clone() + eps * d_d_theta.clone()
			grad_x_val = torch.unsqueeze(torch.zeros_like(theta) + eps, 1)

			constraint_pt_2 = constraint_pt.clone() - eps * d_d_theta.clone()
			grad_y_val = torch.unsqueeze(torch.zeros_like(theta) - eps, 1)
			k_list = []
			op_list = []

		# Concatenate the stationary kernel points and values into tensor
		constraint_pt = torch.cat((constraint_pt.T, constraint_pt_1.T, constraint_pt_2.T))
		constraint_value = torch.cat((constraint_value, grad_x_val, grad_y_val))
		
		# Generate training points, generates more points from the zero level set
		# of the implicit function that can be used for training.

		num_training_points = training_point_opts['num_points']
		
		try:
			close_circle = training_point_opts['close_circle']
		except KeyError:
			close_circle = None

		if close_circle != None:
			theta_training = torch.linspace(0, torch.pi, int(num_training_points/2))
			training_pt = torch.vstack([torch.cos(theta_training), torch.sin(theta_training)]).T
			training_pt_1_x = torch.linspace(-1, 1, int(num_training_points/2))
			training_pt_1_y = torch.linspace(0,0,  int(num_training_points/2))
			training_pt_1 = torch.vstack([training_pt_1_x, training_pt_1_y]).T
			training_pt = torch.cat((training_pt, training_pt_1))
		else:
			theta_training = torch.linspace(0, torch.pi, int(num_training_points))
			training_pt = torch.vstack([torch.cos(theta_training), torch.sin(theta_training)]).T

		training_value = torch.unsqueeze(torch.zeros_like(training_pt[:,0]) , 1)

		return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list
	
	
	if shape == 'circle':
		'''
		Construct a 2D semi-circle with N points.
		'''
		d_in = 2 # input dimension
		d_out = 1 # output dimension
		
		num_points = shape_opts['num_points'] + 1
		normal_constraints = shape_opts['normal_constraints']


		# Generate points on a semi-circle with N points
		theta = torch.linspace(0, 2*torch.pi, num_points, requires_grad = True)
		theta = theta[:-1]
		constraint_pt = torch.vstack([2*torch.cos(theta), 2*torch.sin(theta)])
		constraint_value = torch.unsqueeze(torch.zeros_like(theta) , 1)

		# Generate normal constraints
		d_d_theta = torch.vstack([-torch.sin(theta),  torch.cos(theta)])
		if normal_constraints == 'grad':
			constraint_pt_1 = constraint_pt.clone() #+ 0.001 * d_d_theta.clone()
			grad_x_val = torch.unsqueeze(torch.cos(theta), 1)

			constraint_pt_2 = constraint_pt.clone() #- 0.001 * d_d_theta.clone()
			grad_y_val = torch.unsqueeze(torch.sin(theta), 1)
			
			k_list = [int(len(theta)), 2 * int(len(theta))]
			op_list = ["f_{x_0}", "f_{x_1}"]

		elif normal_constraints == 'pseudo':
			eps = shape_opts['pseudo_eps']
			constraint_pt_1= constraint_pt.clone() + eps * d_d_theta.clone()
			grad_x_val = torch.unsqueeze(torch.zeros_like(theta) + eps, 1)

			constraint_pt_2 = constraint_pt.clone() - eps * d_d_theta.clone()
			grad_y_val = torch.unsqueeze(torch.zeros_like(theta) - eps, 1)
			k_list = []
			op_list = []

		# Concatenate the stationary kernel points and values into tensor
		constraint_pt = torch.cat((constraint_pt.T, constraint_pt_1.T, constraint_pt_2.T))
		constraint_value = torch.cat((constraint_value, grad_x_val, grad_y_val))
		
		# Generate training points, generates more points from the zero level set
		# of the implicit function that can be used for training.

		num_training_points = training_point_opts['num_points']
		
		try:
			close_circle = training_point_opts['close_circle']
		except KeyError:
			close_circle = None

		if close_circle != None:
			theta_training = torch.linspace(0, torch.pi, int(num_training_points/2))
			training_pt = torch.vstack([torch.cos(theta_training), torch.sin(theta_training)]).T
			training_pt_1_x = torch.linspace(-1, 1, int(num_training_points/2))
			training_pt_1_y = torch.linspace(0,0,  int(num_training_points/2))
			training_pt_1 = torch.vstack([training_pt_1_x, training_pt_1_y]).T
			training_pt = torch.cat((training_pt, training_pt_1))
		else:
			theta_training = torch.linspace(0, torch.pi, int(num_training_points))
			training_pt = torch.vstack([torch.cos(theta_training), torch.sin(theta_training)]).T

		training_value = torch.unsqueeze(torch.zeros_like(training_pt[:,0]) , 1)

		return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list

	if shape == 'parallel_lines':
		d_in = 2
		d_out = 1

		constraint_pt = torch.tensor([[-.5, 1.5], 
                       					[.5 , 1.5], 
                       				[.5, -1.5],
                      			 [-.5,-1.5],
								 [-.5, 1.5], 
                       					[.5 , 1.5], 
                       				[.5, -1.5],
                      			 [-.5,-1.5],
								 [-.5, 1.5], 
                       					[.5 , 1.5], 
                       				[.5, -1.5],
                      			 [-.5,-1.5]]).type(torch.float32)
		constraint_pt = constraint_pt.requires_grad_(True)
		print(constraint_pt)
		constraint_value = torch.unsqueeze(torch.tensor([0, 0, 0, 0, 0,0,0,0, 1,1, -1, -1]), 1).type(torch.float32)
		print(constraint_value)
		constraint_value = constraint_value.requires_grad_(True)
		k_list = [4, 8]
		op_list = ["f_{x_0}", "f_{x_1}"]

		constraint_pt = constraint_pt
		constraint_value = constraint_value
		


		# [TRAINING POINTS]
		# Generate training points, generates more points from the zero level set

		num_training_samples = training_point_opts['num_points']
		
		side_samples = int(num_training_samples/4)

		x_neg = torch.linspace(-2, 0, side_samples)
		x_pos = torch.linspace( 0, 2, side_samples)
		
		
		
		training_pt  =  torch.vstack([torch.vstack([x_neg, (x_neg  + 2)]).T, #LT
                              		  torch.vstack([x_neg, (-x_neg - 2)]).T, #LB
                                      torch.vstack([x_pos, (-x_pos + 2)]).T, #RT
                                      torch.vstack([x_pos, (x_pos  - 2)]).T])#RB
		
		# training points are assumed to be sampled from the zero level set.
		training_value = torch.unsqueeze(torch.zeros_like(training_pt[:, 0]), 1)
		
		return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list
	
	if shape == 'line':
		d_in = 2
		d_out = 1
		x = torch.linspace(-1.5, 1.5, 9)
		points = torch.vstack([x, -x]).T
		points1 = torch.vstack([x, -x]).T
		constraint_pt = torch.cat([points, points1, points1]).type(torch.float32)
		constraint_pt = constraint_pt.requires_grad_(True)
		print(constraint_pt)

		constraint_value = torch.unsqueeze(torch.cat([torch.zeros(9), torch.from_numpy(np.array([np.sqrt(2)/2]*18))]), 1).type(torch.float32)
		print(constraint_value)
		constraint_value = constraint_value.requires_grad_(True)
		k_list = [9, 18]
		op_list = ["f_{x_0}", "f_{x_1}"]

		constraint_pt = constraint_pt
		constraint_value = constraint_value
		


		# [TRAINING POINTS]
		# Generate training points, generates more points from the zero level set

		num_training_samples = training_point_opts['num_points']
		
		side_samples = int(num_training_samples/4)

		x_neg = torch.linspace(-2, 0, side_samples)
		x_pos = torch.linspace( 0, 2, side_samples)
		
		
		
		training_pt  =  torch.vstack([torch.vstack([x_neg, (x_neg  + 2)]).T, #LT
                              		  torch.vstack([x_neg, (-x_neg - 2)]).T, #LB
                                      torch.vstack([x_pos, (-x_pos + 2)]).T, #RT
                                      torch.vstack([x_pos, (x_pos  - 2)]).T])#RB
		
		# training points are assumed to be sampled from the zero level set.
		training_value = torch.unsqueeze(torch.zeros_like(training_pt[:, 0]), 1)
		
		return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list


	if shape == 'triangle':

		d_in = 2
		d_out = 1

		constraint_pt = torch.tensor([
					# [0,    1.5], 
                    #    [-1.5 , -1.5], 
                    #    [1.5,   -1.5],
		       			[-0.75, -1.5],
                       [0.75, -1.5],
		       			[-1.125, -0.75], 
                       [-0.375, 0.75],
                       [1.125, -0.75], 
                       [0.375, 0.75],
                       [-0.75, 0],
                       [-0. , -1.5],
                       [0.75, 0],
                        [-0.75, 0],
                       [-0. , -1.5],
                       [0.75, 0],
		       			[-0.75, -1.5],
                       [0.75, -1.5],
		       			[-1.125, -0.75], 
                       [-0.375, 0.75],
                       [1.125, -0.75], 
                       [0.375, 0.75],
                         [-0.75, 0],
                       [-0. , -1.5],
                       [0.75, 0],
					   [-0.75, -1.5],# constraint
                       [0.75, -1.5],
		       			[-1.125, -0.75], 
                       [-0.375, 0.75],
                       [1.125, -0.75], 
                       [0.375, 0.75],
					   
					   
					   ]).type(torch.float32)
		constraint_pt = constraint_pt.requires_grad_(True)
		print(constraint_pt)
		constraint_value = torch.unsqueeze(torch.tensor([ 0,0,0, 0, 0,0,0,0,0,-1/np.linalg.norm([-1, 2/3]), 0, 1/np.linalg.norm([1, 2/3]),0,0,-1/np.linalg.norm([-1, 2/3]), -1/np.linalg.norm([-1, 2/3]),1/np.linalg.norm([1, 2/3]), 1/np.linalg.norm([1, 2/3]),   2/3 *(1/np.linalg.norm([1, 2/3])), -1, 2/3 *(1/np.linalg.norm([1, 2/3])), -1, -1, 2/3 *(1/np.linalg.norm([1, 2/3])), 2/3 *(1/np.linalg.norm([1, 2/3])), 2/3 *(1/np.linalg.norm([1, 2/3])) , 2/3 *(1/np.linalg.norm([1, 2/3]))  ]),  1).type(torch.float32)
		print(constraint_value)
		constraint_value = constraint_value.requires_grad_(True)
		k_list = [9, 9+9]
		op_list = ["f_{x_0}", "f_{x_1}"]

		constraint_pt = constraint_pt
		constraint_value = constraint_value
		


		# [TRAINING POINTS]
		# Generate training points, generates more points from the zero level set

		num_training_samples = training_point_opts['num_points']
		
		side_samples = int(num_training_samples/4)

		x_neg = torch.linspace(-2, 0, side_samples)
		x_pos = torch.linspace( 0, 2, side_samples)
		
		
		
		training_pt  =  torch.vstack([torch.vstack([x_neg, (x_neg  + 2)]).T, #LT
                              		  torch.vstack([x_neg, (-x_neg - 2)]).T, #LB
                                      torch.vstack([x_pos, (-x_pos + 2)]).T, #RT
                                      torch.vstack([x_pos, (x_pos  - 2)]).T])#RB
		
		# training points are assumed to be sampled from the zero level set.
		training_value = torch.unsqueeze(torch.zeros_like(training_pt[:, 0]), 1)
		
		return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list
	
	
	if shape == 'diamond':

		d_in = 2 # input dimension
		d_out = 1 # output dimension

		# Generate points on a diamond with N points
		num_points = shape_opts['num_points']
		normal_constraints = shape_opts['normal_constraints']

		side_samples = int((num_points + 8)/4)
		x_neg = torch.linspace(-2, 0, side_samples, requires_grad=True)[1:-1]
		x_pos = torch.linspace( 0, 2, side_samples, requires_grad=True)[1:-1]
		
		
		
		constraint_pt = torch.vstack([torch.vstack([x_neg, (x_neg  + 2)]).T, #LT
                              		  torch.vstack([x_neg, (-x_neg - 2)]).T, #LB
                                      torch.vstack([x_pos, (-x_pos + 2)]).T, #RT
                                      torch.vstack([x_pos, (x_pos  - 2)]).T]) #RB

		constraint_value = torch.unsqueeze(torch.zeros_like(constraint_pt[:, 0]), 1)


		grad_x = torch.vstack([torch.ones_like(x_neg) *-1/np.sqrt(2),
							   torch.ones_like(x_neg) *-1/np.sqrt(2), 
							   torch.ones_like(x_neg) * 1/np.sqrt(2),
							   torch.ones_like(x_neg) * 1/np.sqrt(2)]).flatten()

		grad_y = torch.vstack([torch.ones_like(x_neg) * 1/np.sqrt(2),
							   torch.ones_like(x_neg) *-1/np.sqrt(2), 
							   torch.ones_like(x_neg) * 1/np.sqrt(2),
							   torch.ones_like(x_neg) *-1/np.sqrt(2),]).flatten()
		
		# Generate normal constraints
		if normal_constraints == 'grad':

			constraint_pt_1 = constraint_pt.clone()
			constraint_pt_1_val = torch.unsqueeze(grad_x, 1)

			constraint_pt_2 = constraint_pt.clone()
			constraint_pt_2_val = torch.unsqueeze(grad_y, 1)

			k_list = [int(len(constraint_pt)), int(len(constraint_pt)) + int(len(constraint_pt))]
			op_list = ["f_{x_0}", "f_{x_1}"]

		elif normal_constraints == 'pseudo':
			eps = shape_opts['pseudo_eps']
			normal_vector = torch.vstack([grad_x, grad_y]).T

			constraint_pt_1 = constraint_pt.clone() + eps * normal_vector
			constraint_pt_1_val = torch.unsqueeze(torch.zeros_like(constraint_pt[:, 0]) + eps, 1)

			constraint_pt_2 = constraint_pt.clone() - eps * normal_vector
			constraint_pt_2_val = torch.unsqueeze(torch.zeros_like(constraint_pt[:, 0]) - eps, 1)
			
			k_list = []
			op_list = []

		constraint_pt = torch.cat([constraint_pt, constraint_pt_1, constraint_pt_2])
		constraint_value = torch.cat([constraint_value, constraint_pt_1_val, constraint_pt_2_val])
		


		# [TRAINING POINTS]
		# Generate training points, generates more points from the zero level set

		num_training_samples = training_point_opts['num_points']
		
		side_samples = int(num_training_samples/4)

		x_neg = torch.linspace(-2, 0, side_samples)
		x_pos = torch.linspace( 0, 2, side_samples)
		
		
		
		training_pt  =  torch.vstack([torch.vstack([x_neg, (x_neg  + 2)]).T, #LT
                              		  torch.vstack([x_neg, (-x_neg - 2)]).T, #LB
                                      torch.vstack([x_pos, (-x_pos + 2)]).T, #RT
                                      torch.vstack([x_pos, (x_pos  - 2)]).T])#RB
		
		# training points are assumed to be sampled from the zero level set.
		training_value = torch.unsqueeze(torch.zeros_like(training_pt[:, 0]), 1)
		
		return constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list


	

