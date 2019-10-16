import json, os
import numpy as np
import progressbar

from lib import get_random_field, build_kernel_from_config


def load(test_case_file):
    with open(test_case_file, 'r') as fs:
        return json.load(fs)


def run(config, output_path):
    #create output path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # load config
    grid = (config.get('grid_size')['x'], config.get('grid_size')['y'])
    n_fields = config.get('fields_per_setting')
    random_state = config.get('random_state')
    
    # build a progressbar
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    prg = 0
    
    # generate kernels
    for kernel, values in build_kernel_from_config(config):
        fields = get_random_field(grid, kernel, n_fields, random_state)
        
        # update progrss
        bar.update(prg)
        prg+= 1
        
        # save each field
        for i in range(fields.shape[1]):
            f = fields[:,i].reshape(grid)
            filename = 'field_%d_%s.dat' % (i + 1, '_'.join([str(_) for _ in values]))
            np.savetxt(os.path.join(output_path, filename), f)        
    

if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        cases = sys.argv[1:]
    else:
        cases = ['test_set.json']
    
    # create output folder if not exists
    if not os.path.exists(os.path.join(os.getcwd(), 'output')):
        os.mkdir(os.path.join(os.getcwd(), 'output'))
    
    # for each case, create output and run
    for case in cases:
        print('Running %s' % case)
        path = os.path.join(os.getcwd(), 'output', case)
        run(load(os.path.join(os.getcwd(), case)), os.path.splitext(path)[0])
