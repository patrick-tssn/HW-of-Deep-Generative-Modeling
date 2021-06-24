import os
from numpy import not_equal
import torch
import torchvision
from models.stylegan2.model import Generator
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import copy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

alpha_lst = [-1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    # StyleGAN 生成器
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load("./pretrained_models/stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    text_lst = ["A_blonde_man","A_Woman_Without_Makeup","a_person_with_purple_hair", "a_person_in_Surprised","A_man_with_a_beard"]
    for text in text_lst:
        for i in tqdm(range(68)):
            # 读取源图和目标图
            latent_init_path = "results/PCA/latent_src/latent_" + str(i) + ".pt"
            latent_final_path = "results/PCA/latent_des/" + text + "/latent_" + str(i) + ".pt"
            prefix = "results/PCA/img_result/" + text
            os.makedirs(prefix, exist_ok=True)
            prefix += "/latent_" + str(i)
            os.makedirs(prefix, exist_ok=True)
            latent_init = torch.load(latent_init_path).cuda()
            latent_final = torch.load(latent_final_path).cuda()
            img, _ = g_ema([latent_init], input_is_latent=True, randomize_noise=False)
            # img_cat = img
            torchvision.utils.save_image(img, prefix+"/0.jpg", normalize=True, range=(-1, 1))
            img = None

            # PCA
            latent_delta = latent_final - latent_init
            U, S, V = torch.svd(latent_delta)
            for j in range(latent_delta.shape[1]):
                u_j = torch.reshape(U[0, :, j], (18, 1))
                v_j = torch.reshape(V[0, :, j], (1, 512))
                latent_init[0] += u_j * v_j * S[0, j]
                img, _ = g_ema([latent_init], input_is_latent=True, randomize_noise=False)
                torchvision.utils.save_image(img, prefix+"/"+str(j + 1) +".jpg", normalize=True, range=(-1, 1))
                img = None
            img, _ = g_ema([latent_final], input_is_latent=True, randomize_noise=False)
            torchvision.utils.save_image(img, prefix+"/"+str(19)+".jpg", normalize=True, range=(-1, 1))
            img = None
            command = "ffmpeg -pattern_type glob -i './results/PCA/img_result/" + text + "/latent_" + str(i) + "/*.jpg' -filter_complex tile=10x2 ./results/PCA/img_result/img_clip/" + text + "_latent_" + str(i) +"_clip.jpg"
            os.system(command)
    # # 手动资源选择
    # text_lst = ["A_blonde_man","A_Woman_Without_Makeup","a_person_with_purple_hair", "a_person_in_Surprised","A_man_with_a_beard"]
    # for text in text_lst:
    #     for i in tqdm(range(68)):
    #         # 读取源图和目标图
    #         latent_init_path = "results/PCA/latent_src/latent_" + str(i) + ".pt"
    #         latent_final_path = "results/PCA/latent_des/" + text + "/latent_" + str(i) + ".pt"
    #         prefix = "results/PCA/img_result/" + text
    #         os.makedirs(prefix, exist_ok=True)
    #         prefix += "/latent_" + str(i)
    #         os.makedirs(prefix, exist_ok=True)
    #         latent_init = torch.load(latent_init_path).cuda()
    #         latent_final = torch.load(latent_final_path).cuda()
    #         ori_img, _ = g_ema([latent_init], input_is_latent=True, randomize_noise=False)
    #         # img_cat = img
    #         # torchvision.utils.save_image(img, prefix+"/0.jpg", normalize=True, range=(-1, 1))
    #         # img = None

    #         # PCA
    #         latent_delta = latent_final - latent_init
    #         U, S, V = torch.svd(latent_delta)
    #         for p in range(7):
    #             torchvision.utils.save_image(ori_img, prefix+"/"+str(p)+'_'+str(0)+".jpg", normalize=True, range=(-1, 1))
    #             latent_tmp_init = latent_init.clone().detach()
    #             for j in range(latent_delta.shape[1]):
    #                 u_j = torch.reshape(U[0, :, j], (18, 1))
    #                 v_j = torch.reshape(V[0, :, j], (1, 512))
    #                 latent_tmp_init[0] += u_j * v_j * S[0, j] * alpha_lst[p]
    #                 # img, _ = g_ema([latent_tmp_init], input_is_latent=True, randomize_noise=False)
    #                 # latent_tmp = latent_init.clone()
    #                 # latent_tmp[0] += u_j * v_j * S[0, j] * alpha_lst[p]
    #                 if (j % 4) == 0:
    #                     img, _ = g_ema([latent_tmp_init], input_is_latent=True, randomize_noise=False)
    #                     # if latent_test.data != latent_tmp.data:
    #                     #     print('Oops!')
    #                     # latent_tmp = None
    #                     # img_cat = torch.cat([img_cat, img])
    #                     torchvision.utils.save_image(img, prefix+"/"+str(p) + "_" + str(j // 4+1)+".jpg", normalize=True, range=(-1, 1))
    #                     img = None

    #             # 图片拼接
    #             # img, _ = g_ema([latent_init * alpha_lst[p]], input_is_latent=True, randomize_noise=False)
    #             # img_cat = torch.cat([img_cat, img])
    #             # torchvision.utils.save_image(img, prefix+"/"+str(p)+'_'+str(6)+".jpg", normalize=True, range=(-1, 1))
    #         command = "ffmpeg -pattern_type glob -i './results/PCA/img_result/" + text + "/latent_" + str(i) + "/*.jpg' -filter_complex tile=6x7 ./results/clip/"+ text+ "_latent_" + str(i) +"_clip.jpg"
    #         os.system(command)
    #         img = None
    #         # img_grid = ToPILImage()(
    #         #     make_grid(img_cat.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
    #         # img_grid.save(prefix + "/grid.jpg")
    #         # img_grid = None
    #         # print("latent_" + str(i) + " finished")

