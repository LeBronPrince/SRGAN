#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
from skimage import measure
import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config
import scipy.misc
###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs  #imgs is a nested list .

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:SRGAN_g
    #     print(im.shape)
    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2
    #d_loss=tf.reduce_mean(logits_real - logits_fake)
    tf.summary.scalar('d_loss',d_loss)
    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    #g_gan_loss=tf.reduce_mean(logits_fake)
    mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    tf.summary.scalar('mse_loss',mse_loss)
    g_loss = mse_loss + vgg_loss + g_gan_loss
    tf.summary.scalar('g_loss',g_loss)
    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)# train_only, printable
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    #g_optim_init = tf.train.RMSPropOptimizer(lr_v, decay=0.9).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    #g_optim = tf.train.RMSPropOptimizer(lr_v, decay=0.9).minimize(g_loss, var_list=g_vars)
    #d_optim = tf.train.RMSPropOptimizer(lr_v, decay=0.9).minimize(d_loss, var_list=d_vars)
    #clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01 , 0.01)) for var in d_vars]
    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:',sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    #tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit+'/_train_sample_96.png')
    #tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit+'/_train_sample_384.png')
    #tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan+'/_train_sample_96.png')
    #tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan+'/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate

    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init+1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0
        #merged = tf.merge_all_summaries()
        #writer = tf.summary.FileWriter("/home/wangyang/Desktop/code/SRGAN/visual",sess.graph)
        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
        print(log)
        tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
        ## quick evaluation on train set
        #if (epoch != 0) and (epoch % 10 == 0):
        #    out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
        #    print("[*] save images")
            #tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)

        ## save model
    #    if (epoch != 0) and (epoch % 10 == 0):
    #        tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/home/wangyang/Desktop/code/SRGAN/visual",sess.graph)
        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            ## update G
            errG, errM, errV, errA, _ ,result_loss= sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim , merged], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            #sess.run(clip_discriminator_var_op)
            writer.add_summary(result_loss,epoch*batch_size+idx)
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)

def convert_rgb_to_y(image, jpeg_mode=True, max_value=255.0):
	if len(image.shape) <= 2 or image.shape[2] == 1:
		return image

	if jpeg_mode:
		xform = np.array([[0.299, 0.587, 0.114]])
		y_image = image.dot(xform.T)
	else:
		xform = np.array([[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0]])
		y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

	return y_image

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path_set14, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path_set14, regx='.*.png', printable=False))
    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path_set14, n_threads=32)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path_set14, n_threads=32)
    size = valid_lr_imgs[13].shape
    t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
    #t_image = tf.placeholder('float32', [None, None, None, 3], name='input_image')
    # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)
    ###========================== DEFINE MODEL ============================###
    ssim_ave=0
    psnr_ave=0
    for imid in range(13,14):#len(valid_lr_imgs)
        # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        ssim_sum=0
        psnr_sum=0
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
            # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
        valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]
        # print(valid_lr_img.min(), valid_lr_img.max())
        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        print("took: %4.4fs" % (time.time() - start_time))
        print("LR size: %s /  generated HR size: %s" % (size, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir+'/set14/%d/valid_gen%d.png' % (imid,imid))
        tl.vis.save_image(valid_lr_img, save_dir+'/set14/%d/valid_lr%d.png' % (imid,imid))
        tl.vis.save_image(valid_hr_img, save_dir+'/set14/%d/valid_hr%d.png' % (imid,imid))
        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, save_dir+'/set14/%d/valid_bicubic%d.png' % (imid,imid))
        """
        valid_hr_img_float = valid_hr_img.astype('float32')
        ### count the metric for ssim and psnr

        im_bic = scipy.misc.imread(save_dir+'/valid_bicubic%d.png' % imid)
        im_gen = scipy.misc.imread(save_dir+'/valid_gen%d.png' % imid)

        im_bic_y = convert_rgb_to_y(im_bic,False)
        im_true_y = convert_rgb_to_y(valid_hr_img_float,False)
        im_gen_y = convert_rgb_to_y(im_gen,False)
        print(np.array(im_gen_y.shape))
        print(np.array(im_true_y.shape))

        ssim = measure.compare_ssim(np.array(im_true_y),np.array(im_gen_y),win_size=11,gradient=False,multichannel=False,gaussian_weights=True,full=False,dynamic_range=255)
        #ssim_sum += ssim
        psnr = measure.compare_psnr(np.array(im_true_y),np.array(im_gen_y),True,255)
        #psnr_sum += psnr
    #ssim_ave=ssim_sum/len(valid_lr_imgs)
    #psnr_ave=psnr_sum/len(valid_lr_imgs)
        print('the average SSIM is %4.4f' %ssim)
        print('the average PSNR is %4.4f' %psnr)
        """
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
