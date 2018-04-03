from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 4
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'

config.VALID.hr_img_path_set5 = 'data2017/Set5/image_SRF_4/HR/'
config.VALID.lr_img_path_set5 = 'data2017/Set5/image_SRF_4/LR/'

config.VALID.hr_img_path_set14 = 'data2017/Set14/image_SRF_4/HR/'
config.VALID.lr_img_path_set14 = 'data2017/Set14/image_SRF_4/LR/'

config.VALID.hr_img_path_u100 = 'data2017/Urban100/image_SRF_4/HR/'
config.VALID.lr_img_path_u100 = 'data2017/Urban100/image_SRF_4/LR/'

config.VALID.hr_img_path_b100 = 'data2017/BSD100/image_SRF_4/HR/'
config.VALID.lr_img_path_b100 = 'data2017/BSD100/image_SRF_4/LR/'

config.VALID.hr_img_path_s80 = 'data2017/Sun-Hays80/image_SRF_8/HR/'
config.VALID.lr_img_path_s80 = 'data2017/Sun-Hays80/image_SRF_8/LR/'
def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))#dict to str
        f.write("\n================================================\n")
