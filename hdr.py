#
#   High Dynamic Range Imaging
#   Written by Qhan
#   2018.4.11
#

from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import os.path as osp
import random

from utils import Weights, ImageAlignment, DebevecMethod, RobertsonMethod, ToneMapping
from timer import Timer


parser = ArgumentParser('High Dynamic Range Imaging')
parser.add_argument('indir', default='cksmh', nargs='?', help='Directory of input images with text file of shutter times.')
parser.add_argument('--savedir', default='res', type=str, help='Output directory.')
parser.add_argument('-n', default=150, type=int, help='Number of sampled points for Debevec method.')
parser.add_argument('-e', default=5, type=int, help='Number of epochs for Robertson method.')
parser.add_argument('-d', default=5, type=int, help='Number of depth for image alignment.')
parser.add_argument('-r', default=1, type=float, help='Resize ratio.')
parser.add_argument('-w', default='triangle', choices=['triangle', 'gaussian', 'uniform'], type=str, help='Weight type.')
parser.add_argument('--lambda', default=10, type=float, help='Lambda for Debevec method.')
parser.add_argument('--alpha', default=0.5, type=float, help='Alpha for photographic tonemapping.')
parser.add_argument('--seed', default=1028, type=int, help='Random seed.')
parser.add_argument('--hdr', default='debevec', choices=['debevec', 'robertson'], type=str, help='HDR algorithm.')
parser.add_argument('--tm', default='all', choices=['all', 'local', 'global', 'bilateral'], type=str, help='Tone mapping method.')
parser.add_argument('--no-plot', action='store_true', dest='no_plot', default=False, help='Plot results.')


t = Timer()

class HDR(Weights):
    
    def __init__(self):
        self.image_alignment = ImageAlignment()
        self.debevec_method = DebevecMethod()
        self.robertson_method = RobertsonMethod()
        self.tone_mapping = ToneMapping()
        self.colors = ['blue', 'green', 'red']

    def check_path(self, indir, savedir):
        if not osp.exists(indir) or not osp.isdir(indir):
            print('[Process] input directory not exists.')
            return False

        if not osp.exists(savedir):
            os.mkdir(savedir)

        return True

    def read_images(self, dirname, r):
        def is_image(filename):
            name, ext = osp.splitext(osp.basename(filename.lower()))
            return ext in ['.jpg', '.png', '.gif']
        
        images = []

        for filename in np.sort(os.listdir(dirname)):
            if is_image(filename):
                im = cv2.imread(osp.join(dirname, filename))
                h, w, c = im.shape
                images += [cv2.resize(im, (int(w * r), int(h * r)))]

        print('[Read] image shape:', images[0].shape)
        print('[Read] images: P =', len(images))

        return images
        
    def read_shutter_times(self, dirname):
        shutter_times, shutter_times_string = [], []

        with open(osp.join(dirname, 'shutter_times.txt'), 'r') as f:
            
            for line in f.readlines():
                line = line.replace('\n', '')
                shutter_times_string += [line]
                
                if '/' in line:
                    a, b = np.float32(line.split('/'))
                    shutter_times += [a/b]
                    
                else:
                    shutter_times += [np.float32(line)]
        
        print('[Read] shutter times:', shutter_times_string)
        return np.array(shutter_times, dtype=np.float32), shutter_times_string
        
    def sample_pixels(self, images, n_samples=150, random_seed=1208):
        print('[Sample] samples per image: N =', n_samples)

        height, width, channels = images[0].shape

        random.seed(random_seed)
        indices = np.array(random.sample(range(height * width), n_samples))

        xv = indices % width
        yv = indices // width

        return [[images[p][yv, xv, c] for p in range(len(images))] for c in range(channels)]
    
    def solve_alignment(self, images, d=4):
        for i in range(1, len(images)):
            print('\r[Alignment] %d' % (i + 1), end='')
            images[i] = self.image_alignment.fit(images[i], images[i-1], d)
        print()
        return images

    def solve_hdr(self, images, hdr_method, ln_st, n_samples, n_epochs, wtype):
        n_images = len(images)

        if hdr_method == 'debevec':
            samples_bgr = self.sample_pixels(images, n_samples)
            return [self.debevec_method.solve(sample, ln_st, n_samples, n_images, wtype) for sample in samples_bgr]
        
        elif hdr_method == 'robertson':
            h, w, channels = images[0].shape
            all_bgr = [[images[p][:, :, c] for p in range(n_images)] for c in range(channels)]
            init_G = [np.exp(np.arange(0, 1, 1 / 256))] * channels
            return self.robertson_method.solve(all_bgr, init_G, np.exp(ln_st), n_epochs)

        else:
            print('[HDR] unknown hdr method:', hdr_method)
            return None

    def solve_tm(self, radiance_bgr, tm_method, alpha, savedir):
        if tm_method in ['global', 'all']:
            ldr = self.tone_mapping.photographic_global(radiance_bgr, a=alpha)
            filepath = osp.join(savedir, "tonemap_global.png")
            cv2.imwrite(filepath, ldr)
        
        if tm_method in ['local', 'all']:
            ldr = self.tone_mapping.photographic_local(radiance_bgr, a=alpha)
            filepath = osp.join(savedir, "tonemap_local.png")
            cv2.imwrite(filepath, ldr)
        
        if tm_method in ['bilateral', 'all']:
            ldr = self.tone_mapping.durand_bilateral(radiance_bgr)
            filepath = osp.join(savedir, "tonemap_bilateral.png")
            cv2.imwrite(filepath, ldr)

        if tm_method not in ['global', 'local', 'bilateral', 'all']:
            print('[Tone Mapping] unknown tone mapping method:', tm_method)
            return None

        return ldr

    def compute_radiance(self, images, lnG_bgr, ln_st, wtype, savedir):
        P = len(images)
        image_shape = images[0].shape
        ln_radiance_bgr = np.zeros(image_shape).astype(np.float32)
        height, width, channels = image_shape

        for c in range(channels): # BGR channels
            W_sum = np.zeros([height, width], dtype=np.float32) + 1e-8
            ln_radiance_sum = np.zeros([height, width], dtype=np.float32)

            for p in range(P): # different shutter times
                im_1D = images[p][:, :, c].flatten()
                ln_radiance = (lnG_bgr[c][im_1D] - ln_st[p]).reshape(height, width)

                weights = self.get_weights(im_1D, wtype).reshape(height, width)
                w_ln_radiance = ln_radiance * weights
                ln_radiance_sum += w_ln_radiance
                W_sum += weights

            weighted_ln_radiance = ln_radiance_sum / W_sum
            ln_radiance_bgr[:, :, c] = weighted_ln_radiance
        
        radiance_bgr = np.exp(ln_radiance_bgr).astype(np.float32)
        cv2.imwrite(osp.join(savedir, 'radiance.hdr'), radiance_bgr)

        print('[Radiance]', radiance_bgr.shape)
        
        return radiance_bgr

    def plot_images(self, images, shutter_times_string, savedir):
        print('[Plot] input')
        P = len(images)
        b = np.ceil((np.sqrt(P))).astype(int)
        fig, ax = plt.subplots(b, b, figsize=(4 * b, 4 * b))
        
        for p in range(P): 
            ax[int(p / b), int(p % b)].imshow(cv2.cvtColor(images[p], cv2.COLOR_BGR2RGB))
            ax[int(p / b), int(p % b)].set_title(shutter_times_string[p])
            
        fig.savefig(osp.join(savedir, 'input_images.png'), bbox_inches='tight', dpi=256)
    
    def plot_response_curve(self, lnG_bgr, savedir):
        print('[Plot] response curve')
        channels = len(lnG_bgr)
        fig, ax = plt.subplots(1, channels, figsize=(5 * channels, 5))
        
        for c in range(channels):
            ax[c].plot(lnG_bgr[c], np.arange(256), c=self.colors[c])
            ax[c].set_title(self.colors[c])
            ax[c].set_xlabel('E: Log Exposure')
            ax[c].set_ylabel('Z: Pixel Value')
            ax[c].grid(linestyle=':', linewidth=1)
            
        fig.savefig(osp.join(savedir, 'response_curve.png'), bbox_inches='tight', dpi=256)

    def plot_radiance(self, radiance, savedir):
        print('[Plot] radiance')
        def fmt(x, pos): return '%.3f' % np.exp(x)

        height, width, channels = radiance.shape
        ln_radiance = np.log(radiance)

        plt.clf()
        fig, ax = plt.subplots(1, channels, figsize=(5 * channels, 5))

        for c in range(channels):
            im = ax[c].imshow(ln_radiance[:, :, c], cmap='jet')
            ax[c].set_title(self.colors[c])
            ax[c].set_axis_off()
            divider = make_axes_locatable(ax[c])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fmt))

        fig.savefig(osp.join(savedir, 'radiance.png'), bbox_inches='tight', dpi=256)
        
    def solve(self, indir, savedir, 
        hdr_method = 'debevec', 
        tm_method = 'local', 
        n_samples = 150, 
        n_epochs = 5, 
        n_depth = 5,
        wtype = 'triangle', 
        alpha = 0.5,
        plot = True,
        resize_ratio = 1):
        if not self.check_path(indir, savedir):
            return None

        # read images
        images = self.read_images(indir, r=resize_ratio)
        n_images = len(images)

        # image alignment
        images = self.solve_alignment(images, n_depth)

        # read shutter times
        st, st_str = self.read_shutter_times(indir)
        ln_st = np.log(st)
        
        # solve HDR, obtain response curve
        lnG_bgr = self.solve_hdr(images, hdr_method, ln_st, n_samples, n_epochs, wtype)
        
        # reconstruct radiance
        radiance_bgr = self.compute_radiance(images, lnG_bgr, ln_st, wtype, savedir)
        
        # convert HDR to LDR by tonemapping
        ldr = self.solve_tm(radiance_bgr, tm_method, alpha, savedir)
        
        # plot result
        if plot:
            self.plot_images(images, st_str, savedir)
            self.plot_response_curve(lnG_bgr, savedir)
            self.plot_radiance(radiance_bgr, savedir)
        
        return lnG_bgr, radiance_bgr, ldr


if __name__ == '__main__':
    args = vars(parser.parse_args())
    
    # Example Usage
    hdr = HDR()
    lnG, radiance, ldr = hdr.solve(
        indir = args['indir'],
        savedir = args['savedir'], 
        hdr_method = args['hdr'],
        tm_method = args['tm'],
        n_samples = args['n'],
        n_epochs = args['e'],
        n_depth = args['d'],
        wtype = args['w'],
        alpha = args['alpha'],
        plot = not args['no_plot'],
        resize_ratio = args['r'])