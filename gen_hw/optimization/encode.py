import time
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.psp import pSp
from tqdm import tqdm

from argparse import Namespace

import dlib 
from utils import align_face

def encode(args):
    ckpt = torch.load(args.encoder, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.encoder
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    # img_path = './data/ffhq/'
    # img_lst = os.listdir(img_path)
    # img_lst.sort()
    # for tf in tqdm(img_lst):
    #     img_file_path = os.path.join(img_path, tf)
    #     if os.path.isfile(img_file_path) and tf.split('.')[-1] == 'jpg' and tf[0] != '.':
    #         id = tf.split('.')[0]

    #         original_image = Image.open(img_file_path)
    #         original_image = original_image.convert("RGB")

    #         predictor = dlib.shape_predictor(args.model_paths['shape_predictor'])
    #         aligned_image = align_face(filepath=img_file_path, predictor=predictor) 
    #         aligned_image.resize(args.resize_dims)
    #         transformed_img = args.transform(aligned_image)

    #         with torch.no_grad():
    #             _, latents = net(transformed_img.unsqueeze(0).cuda(), randomize_noise=False, return_latents=True)

    #         torch.save(latents, './latent/latent_' + id + '.pt')
    in_dir = "results/ffhq/" + args.description.replace(' ', '_') + '/'
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    image_path = in_dir + str(args.image_id) + '.jpg'
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")

    predictor = dlib.shape_predictor(args.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    aligned_image.resize(args.resize_dims)
    transformed_img = args.transform(aligned_image)

    with torch.no_grad():
        _, latents = net(transformed_img.unsqueeze(0).cuda(), randomize_noise=False, return_latents=True)
    out_dir = "results/latent/ffhq/" + args.description.replace(' ', '_') + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(latents, out_dir + 'latent_' + str(args.image_id) + '.pt')

if __name__ == '__main__':
    from config import Config
    args = Config
    encode(args)