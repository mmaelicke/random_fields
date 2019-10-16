import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from itertools import product


class ConfigError(Exception):
    pass


def get_random_field(grid_shape, kernel, n_samples=1, random_state=None):    
    r, c = grid_shape
    xx, yy = np.mgrid[0:r-1:r*1j, 0:c-1:c*1j]

    gp = GaussianProcessRegressor(kernel, random_state=random_state)
    X = np.asarray([[_x, _y] for _x, _y in zip(xx.flatten(), yy.flatten())])
        
    field = gp.sample_y(X, n_samples=n_samples, random_state=random_state)
        
    return field
    
    
def build_kernel_from_config(config):
    # get kernel names
    kernelsConf = config.get('kernels')
    if not isinstance(kernelsConf, list):
        raise ConfigError('No kernel configuration found.')
    
    # get arguments
    args = config.get('args')
    if not isinstance(args, list) or any([not isinstance(_, dict) for _ in args]):
        raise ConfigError('Kernel args are not found or malformed.')
    
    # map the kernel names to classes
    knCls = [getattr(kernels, name) for name in kernelsConf]
    
    # get the attribute names and arguments for product
    attrs = []
    vals = []
    for i, arg  in enumerate(args):
        for k,v in arg.items():
            if len(v) > 1:
                attrs.append(k)
                vals.append(v)
                break

    for values in product(*vals):
        kernel_list = [Cls(
                    **{attrs[i]: values[i]}, # attibute that is changed
                    **{k:v for k,v in args[i].items() if k != attrs[i]}, # static attributes
                ) for i, Cls  in enumerate(knCls)]
        kernel = kernel_list[0]
        for i in range(1,len(kernel_list)):
            kernel += kernel_list[i]
        yield kernel, values