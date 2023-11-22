import helper
from pytorch_lightning import LightningDataModule
import torch
import numpy as np
import torch.utils.data as D
import multiprocessing
import os 
import wget 
from brdf import fastmerl, coords, common
import pandas as pd

class ToyDataset(D.Dataset):
	"""toy dataset"""

	def __init__(self, pt, value, train=True):
		"""
		Args:
		pt: coordinates of the input pt, has size of [number_of_constraints, coord_size]
		value: value of the input pt, has size of [number_of_constraints, value_size]
		"""
		self.pt = pt
		self.value = value
		self.train = train

	def __len__(self):
		return self.pt.shape[0] if self.train else 1

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		return (self.pt[idx, :], self.value[idx, :]) if self.train else (self.pt, self.value)

class ToyDataModule(LightningDataModule):
    functions = {'sin': (torch.sin, torch.cos),
                 'sin2': (lambda x: torch.sin(0.5*x), lambda x: 0.5*torch.cos(0.5*x)),
                 'quadratic': (lambda x: x**2/10, lambda x: 2*x/10)}
    def __init__(self, type='sin', is_ODE=True, constraint_sr=128, training_sr=1000, baseline=False):
        super().__init__()
        self.type = type
        f, df = self.functions[type]
        constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list = helper.gen_toy_data(f, df, is_ODE, constraint_sr, training_sr)
        self.constraint_pt = constraint_pt
        self.constraint_value = constraint_value
        self.dim = (d_in, d_out)
        self.k_list = k_list
        self.op_list = op_list
        # self.train_dataset = ToyDataset(constraint_pt, constraint_value) if baseline else ToyDataset(training_pt, training_value)
        self.train_dataset = ToyDataset(training_pt, training_value)
        self.num_workers = multiprocessing.cpu_count()

        # Test/val
        # merged from param's commit, not sure if expected
        if is_ODE:
            self.test_pt = torch.linspace(constraint_pt[0].item(), constraint_pt[1].item(), steps=255).unsqueeze(dim=1)
            self.val_dataset = ToyDataset(constraint_pt[2:], f(constraint_pt[2:]), train=False) # remove boundary constraint points for validation
        else:
            self.test_pt = torch.linspace(training_pt[0].item(), training_pt[-1].item(), 2000).unsqueeze(dim=1)
            self.val_dataset = ToyDataset(self.test_pt, f(self.test_pt), train=False)

        self.test_dataset = ToyDataset(self.test_pt, f(self.test_pt), train=False)


    def train_dataloader(self):
        return D.DataLoader(self.train_dataset, batch_size=64, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return D.DataLoader(self.val_dataset, batch_size=1, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return D.DataLoader(self.test_dataset, batch_size=1, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)
    
class ToyImageDataModule(LightningDataModule):
    def __init__(self, filepath, as_gray=False, resize=None):
        super().__init__()
        constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list = helper.gen_image(filepath, as_gray, resize)
        self.constraint_pt = constraint_pt
        self.constraint_value = constraint_value
        self.dim = (d_in, d_out)
        self.k_list = k_list
        self.op_list = op_list
        self.train_dataset = ToyDataset(training_pt, training_value)
        self.num_workers = multiprocessing.cpu_count()

        # Test/val
        self.test_dataset = ToyDataset(constraint_pt, constraint_value, train=False)
        self.val_dataset = self.test_dataset

    def train_dataloader(self):
        return D.DataLoader(self.train_dataset, batch_size=64, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return D.DataLoader(self.val_dataset, batch_size=1, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return D.DataLoader(self.test_dataset, batch_size=1, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

class FermatDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list = helper.gen_Fermat()
        self.constraint_pt = constraint_pt
        self.constraint_value = constraint_value
        self.training_pt = training_pt
        self.dim = (d_in, d_out)
        self.k_list = k_list
        self.op_list = op_list
        self.train_dataset = ToyDataset(training_pt, training_value)
        self.num_workers = multiprocessing.cpu_count()

        # Test/val
        self.test_dataset = ToyDataset(training_pt, training_value, train=False)
        self.val_dataset = self.test_dataset

    def train_dataloader(self):
        return D.DataLoader(self.train_dataset, batch_size=640, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return D.DataLoader(self.val_dataset, batch_size=1, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return D.DataLoader(self.test_dataset, batch_size=1, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

class AdvectionDataModule(LightningDataModule):
    def __init__(self, type='sin', beta=0.1, xL=0.0, xR=1.0, num_x=16, t0=0.0, tT=2.0, num_t=16, perturb=0, u0_shift=0):
        super().__init__()
        if type == 'sin':
            def f(x, t, beta, u0_shift):
                return torch.sin(2.*torch.pi*(x - beta*t)) + u0_shift
        else:
            RuntimeError(f'Unknown kernel type: {type}')
        constraint_pt, constraint_value, constraint_gt, training_pt, training_value, d_in, d_out, k_list, op_list = helper.gen_advection_pdebench(f, beta=beta, xL=xL, xR=xR, num_x=num_x, t0=t0, tT=tT, num_t=num_t, perturb=perturb, u0_shift=u0_shift)
        self.constraint_pt = constraint_pt
        self.constraint_value = constraint_value
        self.constraint_gt = constraint_gt
        self.dim = (d_in, d_out)
        self.k_list = k_list
        self.op_list = op_list
        self.train_dataset = ToyDataset(training_pt, training_value)
        self.num_workers = multiprocessing.cpu_count()

        # validation set
        self.val_dataset = ToyDataset(constraint_pt[16:], f(constraint_pt[16:,0], constraint_pt[16:,1], beta, u0_shift=u0_shift), train=False) # remove boundary constraint points for validation

        # test set 
        x = torch.linspace(xL, xR, 50, dtype=torch.float)
        t = torch.linspace(t0, tT, 50, dtype=torch.float)
        x_grid, t_grid = torch.meshgrid(x, t)
        u = f(x_grid, t_grid, beta, u0_shift=u0_shift)
        self.test_pt = torch.stack((x_grid, t_grid), axis=-1).flatten(end_dim=1)
        self.test_value = torch.zeros(u.flatten().shape[0], 1)
        self.test_dataset = ToyDataset(self.test_pt, self.test_value, train=False)

    def train_dataloader(self):
        return D.DataLoader(self.train_dataset, batch_size=64, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return D.DataLoader(self.val_dataset, batch_size=1, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return D.DataLoader(self.test_dataset, batch_size=1, num_workers = self.num_workers, pin_memory=True, persistent_workers=True)

class ImageDataModule(LightningDataModule):
    def __init__(self, image_name ='baboon', const_pattern = 'eye'):
        super().__init__()

        img = helper.load_image(image_name) # returns the image read into mem: shape (H, W, C)
        
        constraint_pt, constraint_value = helper.gen_constrain_pts(img, const_pattern)
        
        # constraint_pt = constraint_pt[:1,:]
        # constraint_value = constraint_value[:1,:]

        training_pts, training_val = helper.gen_training_pts(img)
        
        print("Image Data Module Loaded".center(40, "-"))
        print('image name = {} | img shape = {}'.format(image_name+ '.png', np.shape(img)))
        print('Number of constraint_pt  = {}'.format(np.shape(constraint_pt)[0]))
        print('Number of training_pts  = {}'.format(np.shape(training_pts)[0]))

        d_in = 2
        d_out = 1 # Assume only grayscale images

        k_list = []
        op_list = []

        tmp = img.astype(np.float32) / 255
        self.gt_img = tmp
        tmp = np.transpose(tmp, (2, 0, 1))  # HCW-BGR to CHW-RGB
        self.torch_image = torch.from_numpy(tmp).float().unsqueeze(0)# CHW-RGB to NCHW-RGB


        self.constraint_pt = constraint_pt
        self.constraint_value = constraint_value
        self.dim = (d_in, d_out)
        self.k_list = k_list
        self.op_list = op_list
        self.train_dataset = ToyDataset(training_pts, training_val)
        self.num_workers = multiprocessing.cpu_count()
       
        # Test/val
        self.val_dataset = ToyDataset(training_pts, training_val , train=False) # remove the zero order constraint points for validation
        self.test_pt = training_pts
        
        test_pt, test_val = helper.gen_training_pts(img, 0.25)
        self.test_dataset = ToyDataset(test_pt, test_val)

    def train_dataloader(self):
        return D.DataLoader(self.train_dataset, batch_size=5000, num_workers = self.num_workers)

    def val_dataloader(self):
        return D.DataLoader(self.val_dataset, batch_size=10000, num_workers = self.num_workers)

    def test_dataloader(self):
        return D.DataLoader(self.test_dataset, batch_size=10000, num_workers = self.num_workers)
    

Xvars = ['hx', 'hy', 'hz', 'dx', 'dy', 'dz']
Yvars = ['brdf_r', 'brdf_g', 'brdf_b']
class MerlDataset(LightningDataModule):
    def __init__(self, merlPath, batchsize=5000, n_constraint=50, logscale=False, rvectors_path=None):
        super(MerlDataset, self).__init__()
                
        self.bs = batchsize
        self.BRDF = fastmerl.Merl(merlPath)

        self.reflectance_train = generate_nn_datasets(self.BRDF, nsamples=800000, pct=0.8, logscale=logscale)
        self.reflectance_test = generate_nn_datasets(self.BRDF, nsamples=800000, pct=0.2)
        # self.reflectance_test = generate_test_datasets(self.BRDF, ratio=0.01, logscale=logscale, rvectors_path=rvectors_path)

        self.k_list = []
        self.op_list = []

        constraints = generate_constraint_datasets(self.BRDF, nsamples=n_constraint, logscale=logscale)
        self.constraint_pt = torch.tensor(constraints[Xvars].values, dtype=torch.float32)
        self.constraint_value = torch.tensor(constraints[Yvars].values, dtype=torch.float32)
        self.dim = (6, 3)
        # self.num_workers = multiprocessing.cpu_count()
        self.num_workers = 1

        self.train_samples = torch.tensor(self.reflectance_train[Xvars].values, dtype=torch.float32)
        self.train_gt = torch.tensor(self.reflectance_train[Yvars].values, dtype=torch.float32)
        self.train_dataset = ToyDataset(self.train_samples, self.train_gt)

        self.val_samples = torch.tensor(constraints[Xvars].values, dtype=torch.float32)
        self.val_gt = torch.tensor(constraints[Yvars].values, dtype=torch.float32)
        self.val_dataset = ToyDataset(self.val_samples, self.val_gt)

        self.test_samples = torch.tensor(self.reflectance_test[Xvars].values, dtype=torch.float32)
        self.test_gt = torch.tensor(self.reflectance_test[Yvars].values, dtype=torch.float32)
        self.test_dataset = ToyDataset(self.test_samples, self.test_gt)

    def train_dataloader(self):
        return D.DataLoader(self.train_dataset, batch_size=5000, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return D.DataLoader(self.val_dataset, batch_size=5000, num_workers = self.num_workers)

    def test_dataloader(self):
        return D.DataLoader(self.test_dataset, batch_size=10000, num_workers = self.num_workers)

class PointSetData(LightningDataModule):
    '''
    Data module for pointset data / implicit function reconstruction.
    '''

    def __init__(self, 
                 pointset = {'shape_name': 'semi-circle', 'shape_opts': {'num_points': 24, 'normal_constraints': 'pseudo', 'pseudo_eps': 1e-2}, 
                             'load_ext': False, 'external_file_opts': None},
                 training_point_opts = {'num_points': 1000, 'noise': 0.1,  'seed': 2}):

        import geom_utils as gu
        super().__init__()
        
        if pointset['load_ext'] == True:

            
            ext_file = pointset['external_file_dir']
            shape_name = pointset['shape_name'].lower()
            cwd = os.getcwd()

            if not os.path.exists(cwd + '/3d_models'):
                 os.mkdir(cwd + '/3d_models')
             

            local_path = None 

            # Load external file

            if ext_file == 'github_repo':
                local_path = cwd + '/3d_models/'+pointset['shape_name'].lower() + '.obj'
                if not os.path.exists(cwd + '/3d_models/'+ shape_name+ '.obj'):
                    os.chdir(cwd + '/3d_models')
                    print('') 
                    base_url = 'https://github.com/alecjacobson/common-3d-test-models/raw/master/data/'
                    shape = pointset['shape_name'].lower()
                    url = base_url + shape + '.obj'
                    wget.download(url)
                    os.chdir(cwd)

            elif ext_file == 'local':
                local_path = pointset['external_file_dir']
            

            points, point_normals, points_high = gu.obj2pointset(local_path, pointset['shape_opts'])

            shape_opts = pointset['shape_opts']
            constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list, diff_order= helper.gen_pointset_ndarray(points, point_normals, shape_opts, training_point_opts, points_high = points_high)
            self.constraint_pt = constraint_pt
            self.constraint_value = constraint_value
            self.train_pt = training_pt
            self.train_value = training_value
            self.dim = (d_in, d_out)
            self.points = torch.from_numpy(points).float()

            self.k_list = k_list
            self.op_list = op_list
            self.diff_order = diff_order
            self.train_dataset = ToyDataset(training_pt, training_value, train=True)
            self.num_workers = multiprocessing.cpu_count()

            print(constraint_pt.shape)
        else:
            shape = pointset['shape_name']
            shape_opts = pointset['shape_opts']
            constraint_pt, constraint_value, training_pt, training_value, d_in, d_out, k_list, op_list, = helper.gen_pointset(shape, shape_opts, training_point_opts)
            
            
            # self.points = points
            # self.normals = normals
            self.constraint_pt = constraint_pt
            self.constraint_value = constraint_value
            self.train_pt = training_pt
            self.train_value = training_value
            self.dim = (d_in, d_out)
            self.k_list = k_list
            self.op_list = op_list
            # self.points = points
            # self.normals = normals

            self.train_dataset = ToyDataset(training_pt, training_value, train=True)
            self.num_workers = multiprocessing.cpu_count()

    def train_dataloader(self):
        return D.DataLoader(self.train_dataset, batch_size=512, num_workers = self.num_workers, shuffle=True)


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

def brdf_values(rvectors, brdf:fastmerl.Merl=None, model=None, logscale=False):
    if brdf is not None:
        rangles = coords.rvectors_to_rangles(*rvectors)
        brdf_arr = brdf.eval_interp(*rangles).T
    elif model is not None:
        # brdf_arr = model.predict(rvectors.T)        # nnModule has no .predict
        raise RuntimeError("Should not have entered that branch at all from the original code")
    else:
        raise NotImplementedError("Something went really wrong.")
    brdf_arr *= common.mask_from_array(rvectors.T).reshape(-1, 1)
    if logscale:
        brdf_arr += 1
        brdf_arr = np.where(brdf_arr>0, np.log(brdf_arr), -1e8)
    return brdf_arr

def generate_nn_datasets(brdf:fastmerl.Merl, nsamples=800000, pct=0.8, logscale=False):
    rangles = np.random.uniform([0, 0, 0], [np.pi / 2., np.pi / 2., 2 * np.pi], [int(nsamples * pct), 3]).T
    rangles[2] = common.normalize_phid(rangles[2])
    rvectors = coords.rangles_to_rvectors(*rangles)
    brdf_vals = brdf_values(rvectors, brdf=brdf, logscale=logscale)
    df = pd.DataFrame(np.concatenate([rvectors.T, brdf_vals], axis=1), columns=[*Xvars, *Yvars])
    df = df[(df.T != 0).any()]
    if not logscale:
        df = df.drop(df[df['brdf_r'] < 0].index)
    return df

def generate_constraint_datasets(brdf:fastmerl.Merl, nsamples=1, logscale=False):
    rangles = np.random.uniform([0, 0, 0], [np.pi / 2., np.pi / 2., 2 * np.pi], [int(nsamples), 3]).T

    rangles[0, :int(nsamples) // 2] = np.abs(np.random.normal(0, 0.1, int(nsamples) // 2))

    # if nsamples==1:
    rangles[0,:] = 0.

    rangles[2] = common.normalize_phid(rangles[2])
    rvectors = coords.rangles_to_rvectors(*rangles)
    brdf_vals = brdf_values(rvectors, brdf=brdf, logscale=logscale)
    df = pd.DataFrame(np.concatenate([rvectors.T, brdf_vals], axis=1), columns=[*Xvars, *Yvars])
    df = df[(df.T != 0).any()]
    if not logscale:
        df = df.drop(df[df['brdf_r'] < 0].index)
    return df

def generate_test_datasets(brdf:fastmerl.Merl, ratio=0.01, logscale=False, rvectors_path=None):
    rvectors = np.load(rvectors_path)
    rvectors = rvectors[np.arange(0, rvectors.shape[0], int(1/ratio))].T
    brdf_vals = brdf_values(rvectors, brdf=brdf, logscale=logscale)
    df = pd.DataFrame(np.concatenate([rvectors.T, brdf_vals], axis=1), columns=[*Xvars, *Yvars])
    # df = df[(df.T != 0).any()]
    # df = df.drop(df[df['brdf_r'] < 0].index)
    return df
