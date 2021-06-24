import os, time
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--description", type=str, default="a person with Curly short hair", help="the text that guides the editing/generation")
parser.add_argument("--ckpt", type=str, default="./pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
parser.add_argument("--encoder", type=str, default="./pretrained_models/e4e_ffhq_encode.pt", help="pretrained e4e encoder")
parser.add_argument('--ir_se50_weights', default='./pretrained_models/model_ir_se50.pth', type=str, help="Path to facial recognition network used in ID loss")
parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
parser.add_argument("--lr_rampup", type=float, default=0.05)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--step", type=int, default=301, help="number of optimization steps")
parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"], help="choose between edit an image an generate a free one")
parser.add_argument("--l2_lambda", type=float, default=0.0025, help="weight of the latent distance (used for editing only)")
parser.add_argument("--l1_lambda", type=float, default=0.0002, help="weight of the latent distance (used for editing only)")
parser.add_argument("--id_lambda", type=float, default=0.000, help="weight of the face id distance (used for editing only)")
parser.add_argument("--latent_path", type=str, default='./latent/latent_6.pt', help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                                                                    "the mean latent in a free generation, and from a random one in editing. "
                                                                    "Expects a .pt format")
parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                    "not provided")
parser.add_argument("--save_intermediate_image_every", type=int, default=60, help="if > 0 then saves intermidate results during the optimization")
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("--gpu_id", type=str, default="6")
parser.add_argument("--create_video", type=bool, default=False)
parser.add_argument("--image_id", type=int, default=10)
parser.add_argument("--encode", type=bool, default=True)

parser.add_argument("--out", type=str, default="default")


args = parser.parse_args()

class Config():

    out = args.out

    mode = args.mode #@param ['edit', 'free_generation']

    description = args.description #@param {type:"string"}

    latent_path = './latent/latent_' + str(args.image_id) + '.pt'
    # latent_path = args.latent_path #@param {type:"string"}
    # latent_path = None

    step = args.step #@param {type:"number"}

    l2_lambda = args.l2_lambda #@param {type:"number"}

    l1_lambda = args.l1_lambda
    # l1_lambda = 0

    # d_lambda = args.id_lambda
    id_lambda = 0

    create_video = args.create_video #@param {type:"boolean"}

    gpu_id = args.gpu_id #@param {type:"string"}

    ckpt = args.ckpt

    stylegan_size = args.stylegan_size

    lr_rampup = args.lr_rampup

    lr = args.lr

    truncation = args.truncation

    save_intermediate_image_every = args.save_intermediate_image_every

    results_dir = args.results_dir

    ir_se50_weights = args.ir_se50_weights

    encoder = args.encoder

    image_id = args.image_id

    image_path = './results/ffhq/A_Woman_Without_Makeup/' + str(image_id) + '.png'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    resize_dims = (256, 256)

    model_paths = {
        'stylegan_ffhq': './pretrained_models/stylegan2-ffhq-config-f.pt',
        'ir_se50': './pretrained_models/model_ir_se50.pth',
        'shape_predictor': './pretrained_models/shape_predictor_68_face_landmarks.dat',
        'moco': './pretrained_models/moco_v2_800ep_pretrain.pth'
    }


    log_file = 'log-{time}.log'.format(time = time.strftime('%m-%d-%H:%M:%S'))
    log_path = os.path.join('./log_files', log_file)