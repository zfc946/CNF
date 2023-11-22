import torch
from torch import nn
import nbf
import functorch
import numpy as np
from collections import OrderedDict

class PositionalEncoder(nn.Module):
	r"""
	Sine-cosine positional encoder for input points.
	"""
	def __init__(
	self,
	d_input: int,
	n_freqs: int,
	log_space: bool = False
	):
		super().__init__()
		self.d_input = d_input
		self.n_freqs = n_freqs
		self.log_space = log_space
		self.d_output = d_input * (1 + 2 * self.n_freqs)
		self.embed_fns = [lambda x: x]

		# Define frequencies in either linear or log scale
		if self.log_space:
			freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
		else:
			freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

		# Alternate sin and cos
		for freq in freq_bands:
			self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
			self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

	def forward(
	self,
	x
	) -> torch.Tensor:
		r"""
		Apply positional encoding to input.
		"""
		return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

class FFN(nn.Module):
    def __init__(self, d_input: int, scale: float, n_freqs=256):
        super().__init__()
        self.n_freqs = n_freqs
        self.d_input = d_input
        self.B = torch.randn((d_input, n_freqs)) * scale
        self.d_output = 2 * n_freqs + self.d_input

    def forward(self, x):
        out = [x]
        x = x @ self.B.to(x.device)
        x = 2*np.pi*x
        return torch.cat(out + [torch.sin(x), torch.cos(x)], dim=-1)

class  Encoder(nn.Module):
    def __init__(self, pe, n_layers=1, n_hidden=512, n_out=512, activation=nn.ReLU()):
        super().__init__()
        self.n_hidden = n_hidden
        self.pe = pe

        nets = [nn.Linear(pe.d_output, n_hidden),
                activation]
        for _ in range(n_layers):
            nets.append(nn.Linear(n_hidden, n_hidden))
            nets.append(activation)
        nets.append(nn.Linear(n_hidden, n_out))
        
        self.l1 = nn.Sequential(*nets)

    def forward(self, x):
        return self.l1(self.pe(x))

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
       
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., input_enc = lambda x: x):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        self.input_enc = input_enc

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(self.input_enc(coords))
        return output        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    
def init_basis(basis_type, data_module, kernel_type="Gaussian", init_sigma=1.0, per_basis_sigma=False, identity_encoder=False):
    '''
    Quick initialization of a basis network given its type and data module
    '''
    if per_basis_sigma or identity_encoder:
        assert basis_type == "kernel" and kernel_type == "Gaussian"
    if basis_type == "hypernet":
        # Specify the main network
        n_hidden = 400
        mnet = torch.nn.Sequential(torch.nn.Linear(data_module.dim[0], n_hidden),
                                torch.nn.ReLU(),
                                torch.nn.Linear(n_hidden, n_hidden),
                                torch.nn.ReLU(),
                                torch.nn.Linear(n_hidden, data_module.dim[1]))

        # Specify the hypernetwork
        _, mnet_params = functorch.make_functional(mnet)
        n_mnet_params = sum([p.numel() for p in mnet_params])
        hnet = torch.nn.Sequential(
                torch.nn.Linear(len(data_module.constraint_pt), 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_mnet_params))

        # Specify the hypernet basis function
        basis = nbf.HypernetBasis(mnet, hnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    if basis_type == "hypernet_pe":
        # Specify the main network
        n_freqs = 5
        mnet = Encoder(PositionalEncoder(data_module.dim[0], n_freqs), n_out=data_module.dim[1])

        # Specify the hypernetwork
        _, mnet_params = functorch.make_functional(mnet)
        n_mnet_params = sum([p.numel() for p in mnet_params])
        hnet = torch.nn.Sequential(
                torch.nn.Linear(len(data_module.constraint_pt), 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_mnet_params))

        # Specify the hypernet basis function
        basis = nbf.HypernetBasis(mnet, hnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "hypernet_siren":
        # Specify the main network
        mnet = Siren(data_module.dim[0], hidden_features=21, hidden_layers=1, out_features=data_module.dim[1], outermost_linear=True)

        # Specify the hypernetwork
        _, mnet_params = functorch.make_functional(mnet)
        n_mnet_params = sum([p.numel() for p in mnet_params])
        hnet = torch.nn.Sequential(
                torch.nn.Linear(len(data_module.constraint_pt), 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_mnet_params))

        # Specify the hypernet basis function
        basis = nbf.HypernetBasis(mnet, hnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "singlebasis_siren":
        # Specify the main network
        mnet = Siren(data_module.dim[0], hidden_features=21, hidden_layers=1, out_features=data_module.dim[1], outermost_linear=True)

        # Specify the hypernet basis function
        basis = nbf.SingleBasis(mnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "singlebasis_ffn":
        # Specify the main network
        n_freqs = 256
        mnet = Encoder(FFN(data_module.dim[0], n_freqs=n_freqs, scale=18), n_layers=4, n_hidden=256, n_out=data_module.dim[1])

        # Specify the hypernet basis function
        basis = nbf.SingleBasis(mnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "sepbasis_siren":
        # Specify the main network
        nets = []
        for _ in range(data_module.constraint_pt.shape[0]):
            nets.append(Siren(data_module.dim[0], hidden_features=21, hidden_layers=1, out_features=data_module.dim[1], outermost_linear=True))

        # Specify the hypernet basis function
        basis = nbf.SeparateBasis(nets, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "singlebasis_siren_large":
        # Specify the main network
        mnet = Siren(data_module.dim[0], hidden_features=64, hidden_layers=3, out_features=data_module.dim[1], outermost_linear=True)

        # Specify the hypernet basis function
        basis = nbf.SingleBasis(mnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "hypernet_ffn":
        # Specify the main network
        n_freqs = 256
        mnet = Encoder(FFN(data_module.dim[0], n_freqs=n_freqs, scale=18), n_layers=4, n_hidden=256, n_out=data_module.dim[1])

        # Specify the hypernetwork
        _, mnet_params = functorch.make_functional(mnet)
        n_mnet_params = sum([p.numel() for p in mnet_params])
        hnet = torch.nn.Sequential(
                torch.nn.Linear(len(data_module.constraint_pt), 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_mnet_params))

        # Specify the hypernet basis function
        basis = nbf.HypernetBasis(mnet, hnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim)
    elif basis_type == "kernel_ffn":
        # Specify the encoder for the kernel function
        n_freqs = 64
        encoder = Encoder(FFN(data_module.dim[0], n_freqs=n_freqs, scale=18), n_layers=1, n_hidden=256)

        # Specify the kernel basis function
        basis = nbf.KernelBasis(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma)
    elif basis_type == "kernel":
        # Specify the encoder for the kernel function
        n_freqs = 3
        encoder = nn.Identity() if identity_encoder else Encoder(PositionalEncoder(data_module.dim[0], n_freqs))

        # Specify the kernel basis function
        basis = nbf.KernelBasis(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma, per_basis_sigma=per_basis_sigma)
    
    elif basis_type == "local_kernel":
        # TODO: Do we need to support per-basis sigma?
        n_freqs = 0
        encoder = Encoder(PositionalEncoder(data_module.dim[0], n_freqs))

        # Specify the kernel basis function
        basis = nbf.KernelBasis_LocalSupport(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma)
    
    elif basis_type == "hypernet_kernel":
        # Specify the main network
        n_freqs = 3
        mnet = Encoder(PositionalEncoder(data_module.dim[0], n_freqs))

        # Specify the hypernetwork
        _, mnet_params = functorch.make_functional(mnet)
        n_mnet_params = sum([p.numel() for p in mnet_params])
        hnet = torch.nn.Sequential(
                torch.nn.Linear(len(data_module.constraint_pt), 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_mnet_params))

        # Specify the hyperkernel basis function
        basis = nbf.HypernetKernelBasis(data_module.constraint_pt, mnet = mnet, hnet = hnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma)
    elif basis_type == "hypernet_kernel_geom":
         # Specify the main network
        n_freqs = 0
        mnet = Encoder(PositionalEncoder(data_module.dim[0], n_freqs), n_hidden = 1200, n_out = 800, activation = nn.Softplus(beta = 10))

        # Specify the hypernetwork
        _, mnet_params = functorch.make_functional(mnet)
        n_mnet_params = sum([p.numel() for p in mnet_params])
        hnet = torch.nn.Sequential(
                torch.nn.Linear(len(data_module.constraint_pt), 100),
                torch.nn.Softplus(8),
                torch.nn.Linear(100, n_mnet_params))

        # Specify the hyperkernel basis function
        basis = nbf.HypernetKernelBasis(data_module.constraint_pt, mnet = mnet, hnet = hnet, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type=kernel_type, init_sigma=init_sigma)
    # elif basis_type == "kernel_siwn":
    #     encoder = SimpleMLP(SwinIREncoder(data_module.torch_image))

    #     basis = nbf.KernelBasis(data_module.constraint_pt, encoder, num_basis = data_module.constraint_pt.shape[0], dim=data_module.dim, kernel_type = "dot")
    else:
        RuntimeError(f'Unknown basis type: {basis_type}')

    return basis
