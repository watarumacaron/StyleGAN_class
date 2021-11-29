import tensorflow as tf
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import matplotlib.pyplot as plt
from tqdm import tqdm

class StyleGAN:
    def __init__(self, pickle_path):
        self.pickle_path = pickle_path
        self.Gs, self.fmt = self.load_Gs()

    def load_Gs(self):
        tflib.init_tf()
        with open(self.pickle_path, 'rb') as f:
            _G, _D, Gs = pickle.load(f)
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        return Gs, fmt

    def generate_dlatent(self, seed):
        src_seeds = range(seed)
        src_latents = np.stack(np.random.RandomState(seed).randn(self.Gs.input_shape[1]) for seed in src_seeds)
        print("src_latents.shape : "+str(src_latents.shape))
        src_dlatents = self.Gs.components.mapping.run(src_latents, None,num_gpus=1)
        print("src_dlatents.shape : "+str(src_dlatents.shape))
        return src_dlatents

    def generate_image(self, dlatents):
        src_images = self.Gs.components.synthesis.run(dlatents, randomize_noise=False, num_gpus=1, output_transform=self.fmt)
        return src_images

    def plot_image(self, dlatents=None, npy_path=None):
        if npy_path is not None:
            dlatents = np.load(npy_path)
        for i in range(dlatents.shape[0]):
            image = self.generate_image(dlatents[i][np.newaxis, :, :])
            plt.imshow(image[0])
            plt.show()

    def save_image(self, dlatents, file_name, set_save=False, dlatents_name=None):
        bar = tqdm(range(dlatents.shape[0]))
        for i in bar:
            bar.set_description('Number of saved images %s' % i+1)
            image = self.generate_image(dlatents[i][np.newaxis, :, :])
            image = PIL.Image.fromarray(image[0])
            image.save(file_name + '/{num:06d}.jpg'.format(num = i))
        if set_save == True:
            if dlatents_name is not None:
                np.save(file_name + '/' + dlatents_name + '.npy', dlatents)
            else:
                np.save(file_name + '/dlatents.npy', dlatents)
