import numpy as np
import scipy.misc
from skimage import measure
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

def convert_rgb_to_ycbcr(image, jpeg_mode=True, max_value=255):
	if len(image.shape) < 2 or image.shape[2] == 1:
		return image

	if jpeg_mode:
		xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
		ycbcr_image = image.dot(xform.T)
		ycbcr_image[:, :, [1, 2]] += max_value / 2
	else:
		xform = np.array(
			[[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
			 [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
		ycbcr_image = image.dot(xform.T)
		ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
		ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

	return ycbcr_image
psnr_sum = 0
ssim_sum = 0
for i in range(5):
	image = scipy.misc.imread('/home/wangyang/桌面/SRGAN/samples/evaluate/set5/%d/valid_gen%d.png'%(i,i))
	image_true = scipy.misc.imread('/home/wangyang/桌面/SRGAN/samples/evaluate/set5/%d/valid_hr%d.png'%(i,i))
	y_image = convert_rgb_to_y(image,False)
	y_image_true = convert_rgb_to_y(image_true,False)
	print(np.array(y_image).shape)
	print(np.array(y_image_true).shape)
	ssim = measure.compare_ssim(np.array(y_image_true),np.array(y_image),win_size=11,gradient=False,multichannel=True,gaussian_weights=True,full=False,dynamic_range=255)
	psnr = measure.compare_psnr(np.array(y_image_true),np.array(y_image),True,255)
	ssim_sum = ssim_sum+ssim
	psnr_sum = psnr_sum+psnr
	print(ssim)
	print(psnr)
print("average ssim is %4.4f"%(ssim_sum/5.0))
print("average psnr is %4.4f"%(psnr_sum/5.0))
