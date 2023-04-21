import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

class SquarePad:
    def __call__(self, image):
        w, h = image.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        # padding = (hp, vp, hp, vp)
        padding = ((hp, hp), (vp, vp))
        img = np.pad(image, padding, 'constant')
        return img


class PassiveDegrad():
    def __init__(self, src_size, fake_size):
        self.src_size = src_size
        self.fake_size = fake_size

        src_h, src_w, _ = src_size
        fake_h, fake_w, _ = fake_size
        self.to_gray = True if len(src_size) == 3 else False
        self.reversed_fake_size = (fake_w, fake_h)
         
        scale = max(fake_h / src_h, fake_w / src_w)
        self.scale = scale

        crop_h = int(fake_h / scale)
        crop_w = int(fake_w / scale)
        skip_h = (src_h - crop_h) // 2
        skip_w = (src_w - crop_w) // 2
        self.crop_h, self.crop_w, self.skip_h, self.skip_w = crop_h, crop_w, skip_h, skip_w

    def __call__(self, img):
        if self.to_gray:
            img = img.convert('L')
        img = img.crop((self.skip_w, self.skip_h, self.skip_w+self.crop_w, self.skip_h+self.crop_h))
        img = img.resize(self.reversed_fake_size, resample=Image.BICUBIC)
        return np.array(img)

# visual test
if __name__=="__main__":
    

    transform_real = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_fake = transforms.Compose([
        PassiveDegrad([28,28,1], [12, 16, 1]),
        SquarePad(),
        transforms.ToTensor()
    ])
    
    mnist_real = datasets.CIFAR10(root="../../../datasets/CLSDatasets/CIFAR10", train=True, transform=transform_real, download=True)
    mnist_fake = datasets.CIFAR10(root="../../../datasets/CLSDatasets/CIFAR10", train=True, transform=transform_fake, download=True)

    img_real = [mnist_real.__getitem__(i)[0]*255 for i in range(5)]
    img_fake = [mnist_fake.__getitem__(i)[0]*255 for i in range(5)]

    img_reals = make_grid(img_real, nrow=5, padding=1, pad_value=255).numpy().transpose(1, 2, 0)[:,:, ::-1]
    img_fakes = make_grid(img_fake, nrow=5, padding=1, pad_value=255).numpy().transpose(1, 2, 0)[:, :, ::-1]

    cv2.imwrite("CIFAR10_reals_28x28.png", img_reals)
    cv2.imwrite("CIFAR10_fakes_16x16.png", img_fakes)  