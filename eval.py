import torch

import GAN
import utils

import numpy as np
import sys
import os

#============ PARSE ARGUMENTS =============

args = utils.setup_args()
args.save_name = args.save_file + args.env
print(args)

#============= MODEL INITIALIZATION ==============

# initialize generator
netG = GAN.Generator(args.nz, args.n_hidden)
netG.load_state_dict(torch.load(args.save_name+"_netG.pth"))
print("Generator loaded")

if torch.cuda.is_available():
    netG.cuda()
    print("Using GPU")

# load data
loader = utils.setup_data_loaders(args.batch_size, args.source_data_file, args.target_data_file)
print('Data loaded')
sys.stdout.flush()

netG.eval()

# loop over dataloader
for s_inputs, t_inputs in loader:
    s_inputs = Variable(s_inputs)
    if torch.cuda.is_available():
        s_inputs = s_inputs.cuda()
    s_generated, s_scale = netG(s_inputs)

    # save results to text files

    with open(args.save_name+"_rho.txt", 'ab') as f:
        np.savetxt(f, s_scale.cpu().data.numpy(), fmt='%f')

    with open(args.save_name+"_trans.txt", 'ab') as f:
        np.savetxt(f, s_generated.cpu().data.numpy(), fmt='%f')

    with open(args.save_name+"_source.txt", 'ab') as f:
        np.savetxt(f, s_inputs.cpu().data.numpy(), fmt='%f')

    with open(args.save_name+"_target.txt", 'ab') as f:
        np.savetxt(f, t_inputs.numpy(), fmt='%f')
