import random
import numpy as np
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# Augmentation for Training
class Augmentation(object):
    def __init__(self, img_size=224, pixel_mean=[0., 0., 0.], pixel_std=[1., 1., 1.], jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def rand_scale(self, s):
        scale = random.uniform(1, s)
        if random.randint(0, 1): 
            return scale
        return 1./scale
    
    def resize_fn(self, video_tensor):
        nouvelle_taille = (self.img_size, self.img_size)
        resize = TF.Resize(nouvelle_taille)
        resize_video = resize(video_tensor)
        return resize_video
        
    def horizontal_flip(self, video_tensor):
        random_horizontal_flip = TF.RandomHorizontalFlip(p=1)
        random_horizontal_flip_video = random_horizontal_flip(video_tensor)
        return random_horizontal_flip_video   
    
    def normalisation(self, video_tensor):
        normalized_clip = F.normalize(video_tensor, self.pixel_mean, self.pixel_std)
        return normalized_clip
    
    def random_distort_image(self, video_tensor):
        dsat = self.rand_scale(self.saturation)
        color_jitter = TF.ColorJitter(saturation=dsat, hue=(-self.hue, self.hue))
        color_jitter_clip = color_jitter(video_tensor)
        return color_jitter_clip

    def random_crop(self, video_tensor, width, height):
        dw = int(width * self.jitter)
        dh = int(height * self.jitter)

        pleft = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        pbot = random.randint(-dh, dh)

        swidth = width - pleft - pright
        sheight = height - ptop - pbot

        # Random crop
        cropped_clip = F.crop(video_tensor, ptop, pleft, sheight, swidth)

        return cropped_clip, ptop, pleft, sheight, swidth

    def apply_bbox(self, target, ow, oh, dx, dy, sx, sy):
        sx, sy = 1./sx, 1./sy
        # apply deltas on bbox
        target[..., 0] = np.minimum(0.999, np.maximum(0, target[..., 0] / ow * sx - dx)) 
        target[..., 1] = np.minimum(0.999, np.maximum(0, target[..., 1] / oh * sy - dy)) 
        target[..., 2] = np.minimum(0.999, np.maximum(0, target[..., 2] / ow * sx - dx)) 
        target[..., 3] = np.minimum(0.999, np.maximum(0, target[..., 3] / oh * sy - dy)) 


        print("target apres apply deltas on bbox",target)
        # refine target
        refine_target = []
        for i in range(target.shape[0]):
            tgt = target[i]
            bw = (tgt[2] - tgt[0]) * ow
            bh = (tgt[3] - tgt[1]) * oh

            if bw < 1. or bh < 1.:
                continue
            
            refine_target.append(tgt)
        print( "target apres refine_target",refine_target)
        refine_target = np.array(refine_target).reshape(-1, target.shape[-1])

        return refine_target
    
    
    
    def __call__(self, video_clip, target):
        
        # video_clip=self.tensor_video_to_pil_list(video_clip)
        # Initialize Random Variables
        oh = video_clip.size(2) 
        ow = video_clip.size(3)


        print("target in transform###########################",target)
        # random crop
        video_clip, dx, dy, sx, sy = self.random_crop(video_clip, ow, oh)

        # resize
        video_clip = self.resize_fn(video_clip)

        
        # random flip
        flip = random.randint(0, 1)
        if flip:
            
            video_clip=self.horizontal_flip(video_clip)

        # distort
        video_clip = self.random_distort_image(video_clip)
        print("target in transform",target)
        print("target is none or not ", target is not None)
        # process target
        if target is not None:
            print('target not none da5let')
            print("type of target ",type(target),"   ",target)
            target = self.apply_bbox(target, ow, oh, dx, dy, sx, sy)
            print( "target apres apply_bbox type",type( target) , '   ' ,target)
            if flip:
                target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
                print( "target apres flip type",type( target) , '   ' ,target)
        else:
            target = np.array([])
            
        # to tensor
        video_clip = self.normalisation(video_clip)
        target = torch.as_tensor(target).float()

        return video_clip, target 
    
    
class BaseTransform(object):
    def __init__(self, img_size=224, pixel_mean=[0., 0., 0.], pixel_std=[1., 1., 1.]):
        self.img_size = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        
    def normalisation(self, video_tensor):
        normalized_clip = F.normalize(video_tensor, self.pixel_mean, self.pixel_std)
        return normalized_clip
    
    def resize_fn(self, video_tensor):
        nouvelle_taille = (self.img_size, self.img_size)
        resize = TF.Resize(nouvelle_taille)
        resize_video = resize(video_tensor)
        return resize_video
        
    def __call__(self, video_clip, target=None, normalize=True):
        
        oh = video_clip.size(2) 
        ow = video_clip.size(3)

        # resize
        video_clip = self.resize_fn(video_clip)

        # normalize target
        if target is not None:
            
            if normalize:
                target[..., [0, 2]] /= ow
                target[..., [1, 3]] /= oh

        else:
            target = np.array([])

        # to tensor
        video_clip = self.normalisation(video_clip)
        target = torch.as_tensor(target).float()

        return video_clip, target 
