import torchvision
from config import Config
args = Config()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from optimization.run_optimization import main
from optimization.encode import encode
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from utils import logging, logging_params

# img_path = './data/ffhq/'
# img_lst = os.listdir(img_path)
# img_lst.sort()
# for tf in img_lst:
#     img_file_path = os.path.join(img_path, tf)
#     if os.path.isfile(img_file_path) and tf.split('.')[-1] == 'jpg' and tf[0] != '.':
#         id = tf.split('.')[0]
#         args.image_id = int(id)
#         encode(args, id)

# encode(args)
out_dir = "results/contra/ffhq/" + args.description.replace(' ', '_') + '/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if args.l1_lambda > 0:
    out_dir += '/l1_'

result, res, img_gen = main(args)

# encode(args)

result_image = ToPILImage()(make_grid(result.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
h, w = result_image.size
result_image.resize((h // 2, w // 2))
result_image.save(out_dir + str(args.image_id) + "_contra.jpg")


out_dir = "results/process/ffhq/" + args.description.replace(' ', '_') + '/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if args.l1_lambda > 0:
    out_dir += '/l1_'

result_image = ToPILImage()(make_grid(res.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
h, w = result_image.size
result_image.resize((h // 2, w // 2))
result_image.save(out_dir + str(args.image_id) + "_process.jpg")
