import subprocess as sp
import itertools
import torch
from multiprocessing import Pool
import numpy as np
from datetime import datetime
import os


def worker(args):
    gpu = args[0]
    tasks = args[1]
    
    for cmd in tasks:
        cmd += " --gpu %d" % gpu
        run_cmd(cmd)
    return 0
       
def run_cmd(cmd):
    #cmd = ' '.join(args)
    print(cmd)
    sp.call(cmd, shell=True)
    return True

def run_sweep():
    global available_devices

    
    cur_time = datetime.now().strftime("%y%m%d%H%M")
    save_dir = "/home/dykim/ConvGNN-SSTA/save/%s" % cur_time

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    device_count = torch.cuda.device_count()
    dropout_list = [ 0.1 ]
    hidden_state_size_list = [ 95, 105, 115, 125 ] 
    optimizer_list = [ 'SGD', 'Adam', 'Adadelta', 'RMSprop' ]
    batch_list = [ 10, 20, 30, 40, 50 ]
    n_update_list = [ 1, 2, 3, 4, 5 ]
    n_epoch_list = [ 500 ]
    layer_list = [ '128 128 128', '128 256 512', '256 256 512' ]
    lr_list = [ 0.01, 0.001, 0.0001 ] #e-04, 5e-05, 1e-05 ]
    tasks = []
     
    for (dt, hss, opt, ne, nu, la, lr, bs) in itertools.product(dropout_list, hidden_state_size_list, optimizer_list, n_epoch_list, n_update_list, layer_list,
            lr_list, batch_list):
        shell = 'python'
        run_file = 'train.py'
        flags = ""
        flags += " --dropout {dt:} --hidden_state_size {hss:} --opt {opt:}".format(dt=dt, hss=hss, opt=opt)
        flags += " --n_epochs {ne:} --n_update {nu:} --layers {la:} --lr {lr:}".format(ne=ne, nu=nu, la=la, lr=lr)
        flags += " --save {save:} --batch {bs:}".format(save=save_dir, bs=bs) 
        args = [shell, run_file, flags]
        cmd = ' '.join(args)
        tasks.append(cmd)


    device_list = [ gpu for gpu in range(device_count) ]
    sliced_tasks = [tasks[::device_count+gpu] for gpu in range(device_count)]
   
    print(device_list)
    print(np.shape(sliced_tasks))
    args = [ (gpu, sliced_tasks[gpu]) for gpu in device_list ] 
        
    with Pool(processes=device_count) as pool:
        try:
            pool.map(worker, args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()




    




if __name__ == '__main__':
    run_sweep()

