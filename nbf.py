import torch
import functorch
from torch import optim, nn, Tensor
import pytorch_lightning as pl
import numpy as np
import diff_utils, helper
import abc
from sklearn.neighbors import KDTree
import time
from torch.nn import ModuleList, ParameterList


class Basis(pl.LightningModule, abc.ABC):
	def __init__(self, num_basis, dim, requires_vmap=True):
		super().__init__()
		self.register_buffer('num_basis', torch.tensor(num_basis))
		self.dim = dim
		self.requires_vmap = requires_vmap

	@abc.abstractclassmethod
	def forward(self, x: Tensor, i: Tensor):
		"""
		takes in a tensor x: (batch_size, d_in) and output a tensor evaluating x on the i-th basis function
		return: (batch_size, d_out)
		"""

class KernelBasis(Basis):
	def __init__(self, sample_pt, encoder, num_basis, dim, kernel_type="Gaussian", init_sigma=1.0, per_basis_sigma=False):
		super().__init__(num_basis, dim)
		self.encoder = encoder
		self.register_buffer('sample_pt', sample_pt)
		self.kernel_type = kernel_type
		self.per_basis_sigma = per_basis_sigma
		if kernel_type == "Gaussian":
			if self.per_basis_sigma:
				# TODO: sigma1 and sigma2 is ugly, need an alternative
				# Wrapping with torch.ModuleList spits out strange errors due to vmap
				assert dim[0] == 2
				self.sigma1 = nn.Parameter(torch.tensor([init_sigma]*num_basis, dtype=torch.float32))
				self.sigma2 = nn.Parameter(torch.tensor([init_sigma]*num_basis, dtype=torch.float32))
			else:
				self.sigma = nn.Parameter(torch.tensor([init_sigma], dtype=torch.float32))

	def forward(self, x: Tensor, i: Tensor):
		self.i = i
		return self.kernel(self.sample_pt[i], x)[..., None].repeat(1, self.dim[1])
	
	def kernel(self, x1, x2):
		if self.kernel_type == "Gaussian":
			return self.kernel_gaussian(x1, x2)
		elif self.kernel_type == "dot":
			return self.kernel_dot(x1, x2)
		else:
			RuntimeError(f'Unknown kernel type: {self.kernel_type}')

	def kernel_gaussian(self, x1, x2):
		if self.per_basis_sigma:
			f1, f2 = self.encoder(x1), self.encoder(x2)
			inv = torch.cat((1/self.sigma1[self.i], 1/self.sigma2[self.i]))
			return torch.exp((-0.5*((f2 - f1)[...,None].permute(0,2,1) * inv) @ (f2 - f1)[...,None]).squeeze())
		else:
			return torch.exp(-(torch.norm(self.encoder(x1) - self.encoder(x2), dim = 1)**2)/(2*self.sigma**2))

	def kernel_dot(self, x1, x2):
		return (self.encoder(x1) * self.encoder(x2)).sum(dim=1)
	
class FFNBasis(Basis):
	# def __init__(self, sample_pt: Tensor, dim: int, scale: float, n_freqs=256):
	# 	super().__init__(sample_pt.shape[0], dim)
	# 	self.n_freqs = n_freqs
	# 	self.d_input = sample_pt.shape[-1]
	# 	self.B = torch.randn((sample_pt.shape[-1], n_freqs)) * scale
	# 	self.d_output = 2 * n_freqs + self.d_input

	# def forward(self, x: Tensor, i: Tensor):
	# 	out = [x]
	# 	x = x @ self.B.to(x.device)
	# 	x = 2*np.pi*x
	# 	return torch.cat(out + [torch.sin(x), torch.cos(x)], dim=-1)
	
	def __init__(self, mnet, hnet, num_basis, dim):
		super().__init__(num_basis, dim)
		self.mnet = mnet
		self.hnet = hnet
		self.basis = HyperNetwork(self.hnet, self.mnet)
		self.register_buffer('one_hot', torch.eye(num_basis))

	def forward(self, x: Tensor, i: Tensor):
		z_hnet = self.one_hot[i]
		res = self.basis(z_hnet, x)
		return res

class KernelBasis_LocalSupport(Basis):
	'''
	Variant of the kernel basis that uses a local support for the kernel
	such that the memory requirement is reduced.
	'''
	def __init__(self, sample_pt, encoder, num_basis, dim, kernel_type = "Gaussian", init_sigma = 1.0, dist = 0.2):
		super().__init__(num_basis, dim)
		self.encoder = encoder
		self.register_buffer('sample_pt', sample_pt)
		self.kernel_type = kernel_type
		self.register_buffer('support', torch.tensor(dist, dtype=torch.float32))

		if kernel_type == "Gaussian":
			self.sigma = nn.Parameter(torch.tensor([init_sigma], dtype=torch.float32))
		self.vmap_kernel = functorch.vmap(self.kernel)
	
	def forward(self, x: Tensor, i: Tensor):
		return self.vmap_kernel(self.sample_pt[i], x)
	
	def kernel(self, x1, x2):
		if self.kernel_type == "Gaussian":
			return self.kernel_gaussian(x1, x2)
		elif self.kernel_type == "dot":
			return self.kernel_dot(x1, x2)
		else:
			RuntimeError(f'Unknown kernel type: {self.kernel_type}')

	def kernel_gaussian(self, x1, x2):
		support_density = torch.exp(-torch.norm(x1 - x2, dim = 1)/(2*self.support**2))
		return torch.exp(-(torch.norm(self.encoder(x1) - self.encoder(x2), dim = 1)**2)/(2*self.sigma**2)) * support_density

	def kernel_dot(self, x1, x2):
		support_density = torch.exp(-torch.norm(x1 - x2, dim = 1)/(2*self.support**2))
		return (self.encoder(x1) * self.encoder(x2)).sum(dim=1)*support_density



class PolynomialBasis(Basis):
	def __init__(self, num_basis, dim):
		super().__init__(num_basis, dim)
		self.a = nn.Parameter(torch.tensor([0, 0.01, 0, 0.01], dtype=torch.float32))
		self.b = nn.Parameter(torch.tensor([1, 2, 1, 4], dtype=torch.float32))
		self.c = nn.Parameter(torch.tensor([0, 3, 1, 1], dtype=torch.float32))
  
	def forward(self, x: Tensor, i: Tensor):
		p1 = self.a[2*i]*x**2 + self.b[2*i]*x + self.c[2*i]
		p2 = self.a[2*i+1]*x**2 + self.b[2*i+1]*x + self.c[2*i+1]
		return torch.cat((p1, p2), 1)

class HypernetBasis(Basis):
	def __init__(self, mnet, hnet, num_basis, dim):
		super().__init__(num_basis, dim)
		self.mnet = mnet
		self.hnet = hnet
		self.basis = HyperNetwork(self.hnet, self.mnet)
		self.register_buffer('one_hot', torch.eye(num_basis))

	def forward(self, x: Tensor, i: Tensor):
		z_hnet = self.one_hot[i]
		res = self.basis(z_hnet, x)
		return res
	
class NetBasis(Basis):
	'''
	Use part output of the network as basis
	'''
	def __init__(self, net, num_basis, dim):
		super().__init__(num_basis, dim, requires_vmap=False)
		self.net = net

	def forward(self, x: Tensor, i: Tensor):
		res = self.net(x[0,...]) #[i*self.dim[-1]: (i+1)*self.dim[-1]]

		res = res.reshape([x.shape[0], -1, self.dim[-1]])
		res.permute([1,0,2])
		return res

class SingleBasis(Basis):
	def __init__(self, mnet, num_basis, dim):
		super().__init__(num_basis, dim)
		self.mnet = mnet
		if num_basis != 1:
			raise NotImplementedError()

	def forward(self, x: Tensor, i: Tensor):
		res = self.mnet(x)
		return res
	
# class SeparateBasis(Basis):
# 	def __init__(self, nets, num_basis, dim):
# 		super().__init__(num_basis, dim)
# 		self.nets = ModuleList(nets)

# 	def forward(self, x: Tensor, i: Tensor):
# 		res = self.nets[i](x)
# 		return res
	
class SeparateBasis(Basis):
	def __init__(self, net_list, num_basis, dim):
		super().__init__(num_basis, dim)

		# Obtain functional form and number of layers from 0th network 
		func, params = functorch.make_functional(net_list[0])
		self.net_func = torch.vmap(func)
		self.params_lst = [[] for _ in range(len(params))]

		# Iterate through nets and collect per-layer parameters
		for net in net_list:
			func, params = functorch.make_functional(net)
			# TODO: stronger check to ensure architecture is the same
			assert len(params) == len(self.params_lst), 'All basis networks must have the same architecture'
			for j, layer in enumerate(params):
				self.params_lst[j].append(layer)

		# Each layer should be a batched tensor
		for j, params in enumerate(self.params_lst):
			self.params_lst[j] = nn.Parameter(torch.stack(params).to('cuda'))
		
		# self.params_lst = ParameterList(self.params_lst)
		pass

	def forward(self, x: Tensor, i: Tensor):
		return self.net_func(self.params_lst, x)[i]

class HypernetKernelBasis(Basis):
	def __init__(self, sample_pt, mnet, hnet, num_basis, dim, kernel_type = "Gaussian", init_sigma = 1.0):
		super().__init__(num_basis, dim)
		self.mnet = mnet
		self.hnet = hnet
		self.basis = HyperNetwork(self.hnet, self.mnet)
		self.register_buffer('one_hot', torch.eye(num_basis))
		self.register_buffer('sample_pt', sample_pt)
		self.kernel_type = kernel_type
		if kernel_type == "Gaussian":
			self.sigma = nn.Parameter(torch.tensor([init_sigma], dtype=torch.float32))

	def forward(self, x: Tensor, i: Tensor):
		return self.kernel(x, i)[..., None].repeat(1, self.dim[1])
	
	def kernel(self, x, i):
		if self.kernel_type == "Gaussian":
			return self.kernel_gaussian(self.sample_pt[i], x, i)
		elif self.kernel_type == "dot":
			return self.kernel_dot(self.sample_pt[i], x, i)
		else:
			RuntimeError(f'Unknown kernel type: {self.kernel_type}')

	def kernel_gaussian(self, x1, x2, i):
		z_hnet = self.one_hot[i]
		return torch.exp(-(torch.norm(self.basis(z_hnet, x1) - self.basis(z_hnet, x2), dim = 1)**2)/(2*self.sigma**2))

	def kernel_dot(self, x1, x2, i):
		z_hnet = self.one_hot[i]
		return (self.basis(z_hnet, x1) * self.basis(z_hnet, x2)).sum(dim=1)

class NeuralBasisField(pl.LightningModule):
	def __init__(self, basis: Basis, constraint_pt, constraint_value, k_list, op_list, 
		  smooth_exp=1, lr = None, smooth_weight = 1e0, cond_num_weight = 1e0, use_regression = False, loss_fn = None):
		"""
		Args:
			basis: a basis function 
			constraint_pt: coordinates of the constraint points, has shape of [number_of_constraints, coord_size]
			constraint_value: value of the constraint points, has shape of [number_of_constraints, value_size]
			op_list: a list of linear operators specified in LaTeX format
			k_list: a list of the starting index of the constraint points for each linear operation specified in op_list
			smooth_exp: exponent to compute smoothness loss, set to 0 if smoothness regularization is not needed
			smooth_weight: weight for smoothness loss
			cond_num_weight: weight for condition number loss
			use_regression: flag indicating whether a regression loss should be included in training
			loss_fn: user-specificed additional training loss
		"""
		super().__init__()
		# self.save_hyperparameters()
		# Only learnable (nn.Parameter or nn.Module) or non-learnable (registered buffer) parameters
		# will be automatically sent to device by PL.
		self.register_buffer('constraint_pt', constraint_pt)
		self.register_buffer('constraint_value', constraint_value)
		self.register_buffer('num_constraint', torch.tensor(constraint_pt.shape[0]))
		self.register_buffer('num_basis', basis.num_basis)
		self.k_list = k_list
		self.op_list = op_list
		self.basis = basis
		if self.basis.requires_vmap:
			self.basis_vmap = functorch.vmap(self.basis)
		else:
			self.basis_vmap = self.basis

		self.smooth_exp = smooth_exp
		self.smooth_weight = smooth_weight
		self.cond_num_weight = cond_num_weight
		self.use_regression = use_regression
		self.loss_fn = loss_fn

		self.train_time = 0.

		if lr is not None:
			self.lr = lr
		else:
			self.lr = 1e-4
			
		self.mse_fn = nn.functional.mse_loss

	def backward(self, loss, optimizer=None, optimizer_idx=None):
		# with torch.autograd.set_detect_anomaly(True):
		loss.backward(retain_graph=True) #todo: comment

	def compute_matrix(self):
		basis_id = torch.arange(self.num_basis)[..., None]  # basis_id: (num_basis, 1)

		x_vec = self.constraint_pt[None].repeat(self.num_basis, 1, 1)  # x_vec: (num_basis, num_constraint, d_in)
		X = self.basis_vmap(x_vec, basis_id) # X: (num_basis, num_constraint, d_out)

		for i in range(len(self.k_list)):
			X_grad = diff_utils.compute_op(self.op_list[i], X , x_vec)
			if i+1<len(self.k_list):
				X[:,self.k_list[i]:self.k_list[i+1],:] = X_grad[:,self.k_list[i]:self.k_list[i+1],:]
			else:
				X[:,self.k_list[i]:,:] = X_grad[:,self.k_list[i]:,:]
		return X.permute(2,1,0)  # return: (d_out, num_constraint, num_basis)

	def compute_weights(self, X):
		beta = torch.linalg.solve(X, self.constraint_value.transpose(0,1))
		return beta.transpose(0,1)

	def forward(self, x: Tensor):
		X = self.compute_matrix()
		beta = self.compute_weights(X) # beta: (num_basis, d_out)

		basis_id = torch.arange(self.num_basis)[..., None]  # basis_id: (num_basis, 1)
		x_vec = x[None].repeat(self.num_basis, 1, 1)  # x_vec: (num_basis, batch_size, d_in)
		mat = self.basis_vmap(x_vec, basis_id)  # mat: (num_basis, batch_size, d_out)

		mat = mat.permute(1,0,2)  # mat: (batch_size, num_basis, d_out)

		weight_sum = torch.einsum('ijb,jb->ib', mat, beta) # weight_sum: (batch_size, d_out)

		return weight_sum

	def training_step(self, batch, batch_idx):
		x, y, *_ = batch

		if self.use_regression:
			y_hat = self(x)
			loss_regression = nn.functional.mse_loss(y_hat, y)

			self.log("train/regression_loss", loss_regression)
		
		loss_smoothness = 0 if self.smooth_exp == 0 else torch.linalg.norm(torch.diff(self(x), dim=0), ord=self.smooth_exp)
		self.log("train/smoothness_loss", loss_smoothness)

		X = self.compute_matrix().squeeze(0)

		loss_cond_num = torch.linalg.cond(X).mean()
		self.log("train/condition_number", loss_cond_num)

		if self.loss_fn is not None:
			loss_additional = self.loss_fn(batch, self)
			self.log("train/additional_loss", loss_additional)
		
		loss = self.cond_num_weight*loss_cond_num + self.smooth_weight*loss_smoothness
		if self.use_regression:
			loss = loss + loss_regression
		if self.loss_fn is not None:
			loss = loss + loss_additional

		self.log("train/loss", loss)

		return loss

	def validation_step(self, batch, batch_idx):
		x, y, *_ = batch
		with torch.inference_mode(False): 
			x = x.clone().squeeze(0)
			y_hat = self(x)
		self.log('val/mse', self.mse_fn(y_hat, y.squeeze(0)).item())
		self.test_step(batch, batch_idx)

	def test_step(self, batch, batch_idx):
		x, y, *_ = batch
		with torch.inference_mode(False):
			y_hat = self(x.clone().squeeze(0)).squeeze()
			x = x.clone().squeeze()
			y = y.clone().squeeze()
			if x.dim() == 1 and y.dim() == 1:
				helper.plot_1d(x, y, y_hat, self.current_epoch, mse_fn = self.mse_fn, save_image = False, logger = self.logger)

	def configure_optimizers(self):
		params = self.parameters()
		if isinstance(self.basis, SeparateBasis):
			params = list(params) + self.basis.params_lst
		optimizer = optim.Adam(params, lr=self.lr)
		return optimizer
  
	def on_train_epoch_start(self):
		self.time_stamp = time.time()
	
	def on_train_epoch_end(self):
		epoch_time = time.time() - self.time_stamp
		self.train_time += epoch_time


class NeuralBasisField_LocalKernelsWeighted(pl.LightningModule):
	def __init__(self, basis: Basis, constraint_pt, constraint_value, k_list, op_list, 
	      smooth_exp=1, loss_singularity_fn='cond', lr = None, smooth_weight = 1e0, singularity_weight = 1e0, use_regression = False, loss_fns = None, nn_size = None, points = None , normals = None, data = None, diff_order = None, avg_dist = 0.01):
		"""
		Variant of the NeuralBasisField class that uses local kernels with weights to approximate the function.

		Args:
			basis: a basis function 
			constraint_pt: coordinates of the constraint points, has shape of [number_of_constraints, coord_size]
			constraint_value: value of the constraint points, has shape of [number_of_constraints, value_size]
			op_list: a list of linear operators specified in LaTeX format
			k_list: a list of the starting index of the constraint points for each linear operation specified in op_list
			smooth_exp: exponent to compute smoothness loss, set to 0 if smoothness regularization is not needed
			loss_singularity_fn: pick either "ind" (pairwise independence) or "cond" (condition number)
			smooth_weight: weight for smoothness loss
			singularity_weight: weight for singularity loss
			use_regression: flag indicating whether a regression loss should be included in training
			loss_fn: user-specificed additional training loss
		"""
		super().__init__()
		
		# Only learnable (nn.Parameter or nn.Module) or non-learnable (registered buffer) parameters
		# will be automatically sent to device by PL.
		
		self.register_buffer('constraint_pt', constraint_pt)
		self.register_buffer('constraint_value', constraint_value)
		self.register_buffer('num_constraint', torch.tensor(constraint_pt.shape[0]))
		self.register_buffer('num_basis', basis.num_basis)
		
		self.constraint_pt_kdtree = KDTree(constraint_pt.detach().cpu().numpy(), leaf_size=2)
		
		self.k_list = k_list
		self.op_list = op_list
		self.basis = basis
		
		self.support_variance = torch.tensor([4 * avg_dist]).to('cuda')
		self.register_buffer('support', torch.sqrt(self.support_variance))
		self.basis.support = self.support

		self.basis_vmap = functorch.vmap(self.basis)
		
		self.diff_order = diff_order
		self.smooth_exp = smooth_exp
		self.smooth_weight = smooth_weight
		self.loss_singularity_fn = loss_singularity_fn
		self.singularity_weight = singularity_weight
		self.use_regression = use_regression
		self.loss_fns = loss_fns

		if lr is not None:
			self.lr = lr
		else:
			if self.loss_singularity_fn == 'ind':
				self.lr = 1e-5
			elif self.loss_singularity_fn == 'cond':
				self.lr = 1e-7
			elif self.loss_singularity_fn == 'svd':
				self.lr = 1e-6
			elif self.loss_singularity_fn == 'wsv':
				self.lr = 1e-4
			
		self.mse_fn = nn.functional.mse_loss

	def backward(self, loss, optimizer=None, optimizer_idx=None):
		with torch.autograd.set_detect_anomaly(True):
			loss.backward(retain_graph=True) #todo: comment
		
	

	def compute_local_constraints(self, x: Tensor):
		checksize = len(self.constraint_pt)

		qps = x.detach().cpu().numpy()
		dist, ind = self.constraint_pt_kdtree.query(qps, checksize)

		dist = torch.from_numpy(dist).to(x.device)[0] # matrix: (N, K
		ind = torch.from_numpy(ind).to(x.device)[0]   # matrix: (N, K)
		

		tmp = (3 * self.support - dist) > 0
		num_true = torch.sum(tmp.bool())
		

		# MEMORY GUARD: Prevents the total number of constraints from
		# exceeding 60 - the maximum number of constraints that can be
		# handled by the GPU.

		if num_true > 60:
			num_true = 60

		self.nn_size = num_true
	
		ind = ind[0:num_true]
		
	

		if num_true == 0:
			self.nn_size = 0
			compact_suppprt = False
			local_constraints = None
			ind = None

			return compact_suppprt, local_constraints, ind
		
		local_constraints = self.constraint_pt[ind] # matrix: (N, K, d_in)

		compact_suppprt = True

		return compact_suppprt, local_constraints, ind
	


	def compute_local_matrices(self):
		
		qp_id = torch.unsqueeze(self.constraint_index, dim = 2)
		x_vec = torch.unsqueeze(self.local_constraints, dim = 1)
		x_vec = x_vec.repeat(1, self.nn_size, 1, 1)		

		return self.basis_vmap(x_vec, qp_id)
	
	
	def solve_linear_system(self, Xs):
		constraint_values = torch.squeeze(self.constraint_value[self.constraint_index], dim = 2).type(torch.DoubleTensor).to(self.device)
		coeffs = torch.linalg.solve(Xs, constraint_values)
		return coeffs
	
	def compute_query_basis(self, x: Tensor):
		x_vec = torch.unsqueeze(x, dim = 1)
		x_vec = x_vec.repeat(1,self.nn_size,1)

		qp_id = torch.unsqueeze(self.constraint_index, dim = 2)
		query_basis = torch.squeeze(self.basis_vmap(x_vec, qp_id), dim = 2)

		return query_basis
	


	def forward(self, x: Tensor):

		# x : Tensor that is batched. (N, d_in)
		outputs = []
		for x_i in x:#tqdm(x, total = len(x)):
		# Currently only support batch size of 1 
		# hence, loop over all inputs.
			x_i = x_i.unsqueeze(0)

			# (1) Compute the local constraints_points for each x_i in x
			compact_support, self.local_constraints, self.constraint_index = self.compute_local_constraints(x_i)

			if compact_support == False:
				outputs.append(torch.ones(1, device= self.device) + 1e5)
				continue
			self.constraint_index = torch.unsqueeze(self.constraint_index, 0)

			self.local_constraints = torch.unsqueeze(self.local_constraints, 0)
			# (2) Compute the local matrix X_i for each set of local constraint points c_i
			Xs = self.compute_local_matrices() # Xs: (N, K, K)

			# (3) Then for each X_i in Xs we solve the linear system
			coeffs = self.solve_linear_system(Xs) # coeffs: (N, K)


			# (4) Query the local kernel basis at each x_i in x
			x_basis = self.compute_query_basis(x_i)

			# (5) Compute the linear combination of the local kernel basis
			f_x = torch.einsum('...i,...i->...', x_basis, coeffs) # weight_sum: (batch_size, d_out)

			outputs.append(f_x)
		f_x = torch.hstack(outputs)

		return torch.unsqueeze(f_x, dim = -1)

	def training_step(self, batch, batch_idx):

		x, y, *_ = batch

		if self.use_regression:
			y_hat = self(x)
			loss_regression = nn.functional.mse_loss(y_hat, y)
			self.log("train/regression_loss", loss_regression)
		
		loss_smoothness = 0 if self.smooth_exp == 0 else torch.linalg.norm(torch.diff(self(x), dim=0), ord=self.smooth_exp)
		self.log("train/smoothness_loss", loss_smoothness)

		X = self.compute_local_matrices().squeeze()

		loss_additional = []
		if self.loss_fns is not None:
			for i in range(len(self.loss_fns)):
				loss_additional.append(self.loss_fns[i](batch, self))
				self.log("train/additional_loss_{}".format(i), loss_additional[-1])
		
		loss = self.smooth_weight*loss_smoothness
		
		if self.use_regression:
			loss = loss + loss_regression *1e2
		
		if self.loss_fns is not None:
			for i in range(len(loss_additional)):
				loss = loss + loss_additional[i]
		
		self.log("train/loss", loss)

		return loss

	def validation_step(self, batch, batch_idx):
		x, y, *_ = batch
		with torch.inference_mode(False): 
			x = x.clone().squeeze(0)
			y_hat = self(x)
		self.log('val/mse', self.mse_fn(y_hat, y.squeeze(0)).item())
		self.test_step(batch, batch_idx)

	def test_step(self, batch, batch_idx):
		x, y, *_ = batch
		with torch.inference_mode(False):
			y_hat = self(x.clone().squeeze(0)).squeeze()
			x = x.clone().squeeze()
			y = y.clone().squeeze()
			if x.dim() == 1 and y.dim() == 1:
				helper.plot_1d(x, y, y_hat, self.current_epoch, mse_fn = self.mse_fn, save_image = False, logger = self.logger)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.lr)
		return optimizer

'''Reference: https://mfkasim1.github.io/2022/06/09/functorch-hypernet/'''
class HyperNetwork(torch.nn.Module):
	def __init__(self, hypnet: torch.nn.Module, synthnet: torch.nn.Module):
		'''
		hypnet: the network that takes x and produces the parameters of synthnet
		synthnet: the network that takes z and produces h		
		'''
		super().__init__()
		s_func, s_params0 = functorch.make_functional(synthnet)

		# store the information about the parameters
		self._sp_shapes = [sp.shape for sp in s_params0]
		self._sp_offsets = [0] + list(np.cumsum([sp.numel() for sp in s_params0]))

		# make the synthnet_func to accept batched parameters
		synthnet_func = functorch.vmap(s_func)
		# a workaround of functorch's bug #793
		self._synthnet_batched_func = [synthnet_func]
		self._hypnet = hypnet
		self._s_func = [s_func]

	def forward_vmap(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		'''z: (batch_size, nz), x: (batch_size, nx)'''
		params = self._hypnet(z)  # params: (batch_size, nparams_tot)

		# rearrange params to have the same shape as the synthnet params, except on the batch dimension
		params_lst = []
		for i, shape in enumerate(self._sp_shapes):
			j0, j1 = self._sp_offsets[i], self._sp_offsets[i + 1]
			params_lst.append(params[..., j0:j1].reshape(-1, *shape))

		# apply the function to the batched parameters and z
		h = self._synthnet_batched_func[0](params_lst, x)
		return h
	
	def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		'''z: (nz), x: (nx)'''
		params = self._hypnet(z)  # params: (batch_size, nparams_tot)

		# rearrange params to have the same shape as the synthnet params, except on the batch dimension
		params_lst = []
		for i, shape in enumerate(self._sp_shapes):
			j0, j1 = self._sp_offsets[i], self._sp_offsets[i + 1]
			params_lst.append(params[..., j0:j1].reshape(*shape))

		# apply the function to the batched parameters and z
		h = self._s_func[0](params_lst, x)
		return h
	