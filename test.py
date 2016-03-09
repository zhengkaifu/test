from PIL import Image
import numpy as np
import pylab
import matplotlib as mpl
from skimage import color
from skimage import transform, filters
from augment import (img_augment, sample_img_augment_params, AugmentedInput,
                      SupervisedAugmentedInput)
from util import img_transform
def _resize(args):
    img, rescale_size, bbox = args
    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Smooth image before resize to avoid moire patterns
    scale = img.shape[0] / float(rescale_size)
    sigma = np.sqrt(scale) / 2.0
    img = filters.gaussian_filter(img, sigma=sigma, multichannel=True)
    img = transform.resize(img, (rescale_size, rescale_size, 3), order=3)
    img = (img*255).astype(np.uint8)
    return img


def _resize_augment(args):
    img, rescale_size, bbox = args
    augment_params = sample_img_augment_params(
        translation_sigma=2.00, scale_sigma=0.01, rotation_sigma=0.01,
        gamma_sigma=0.05, contrast_sigma=0.05, hue_sigma=0.01
    )
    img = img_augment(img, *augment_params, border_mode='nearest')
    img = _resize((img, rescale_size, bbox))
    return img

if __name__ == '__main__':
	img=np.array(Image.open('./000001.jpg').convert('L'))
	# a.show()
	# a.save('aa.jpg')
	_resize_augment((img,64,(40, 218-30, 15, 178-15)))
