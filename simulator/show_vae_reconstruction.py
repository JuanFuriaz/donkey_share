#!/usr/bin/env python3
"""
Script to show vae reconstruction of environment

Usage:
    show_vae_reconstruction.py  [--vae=<vae>] 

Options:
    -h --help          Show this screen.
"""


import os
import time
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import _thread 
import traceback
import torch
import sys
from vae.vae import VAE
import torchvision.transforms.functional as F_
import donkeycar as dk


def show_reconstruction(cfg,args):
    vae_path=args['--vae']
    print('vae_path: %s'%vae_path)
    
    if not vae_path:
        print('Error: No vae path specified')
        sys.exit(0)
        
    # init vae 
    print('Initializing vae...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(image_channels=cfg.IMAGE_CHANNELS, z_dim=cfg.VARIANTS_SIZE)
    vae.load_state_dict(torch.load(vae_path, map_location=torch.device(device)))
    vae.to(device).eval()
    
    
    for i in range(cfg.TIME_STEPS):
    
        z = torch.load('z_tensor.pt')
        
        reconst = vae.decode(z)
        reconst = reconst.detach().cpu()[0].numpy()
        reconst = np.transpose(np.uint8(reconst*255),[1,2,0])
        
        reconst_image = F_.to_pil_image(reconst)
        imgplot = plt.imshow(reconst_image)

        plt.pause(0.05)
        
        time.sleep(0.1)


args = docopt(__doc__)
cfg = dk.load_config()

show_reconstruction(cfg,args)
