#
#   HDR Utilities
#   Written by Qhan
#   2018.4.11
#   2019.3.29
#

import cv2
import numpy as np


gaussian = lambda x, mu, s: 1 / (s * (2 * np.pi) ** (1/2)) * np.exp(-(x - mu) ** 2 / (2 * s ** 2))

class Weights():
    
    def get_weights(self, Z, wtype='triangle', mean=128, sigma=128):
        if wtype == 'triangle':
            weights = np.concatenate((np.arange(1, 129), np.arange(1, 129)[::-1]), axis=0)
            return weights[Z].astype(np.float32)

        elif wtype == 'gaussian':
            w = np.arange(256)
            return gaussian(w, mean, sigma)[Z].astype(np.float32) * 128

        elif wtype == 'uniform':
            return np.ones(Z.shape, dtype=np.float32) * 128

        else:
            print('[Weight] unknown weight type.')
            return None

        
#
#   Image Alignment (Pyramid Method)
#
class ImageAlignment():
    
    def __init__(self, threshold=4):
        self.thres = threshold

    def gradient_magnitude(self, I):
        r, l, t, b = np.zeros(I.shape), np.zeros(I.shape), np.zeros(I.shape), np.zeros(I.shape)
        r[:, :-1] = I[:, 1:]; r[:, -1] = I[:, -1]
        l[:, 1:] = I[:, :-1]; l[:, 0] = I[:, 0]
        t[1:] = I[:-1]; t[0] = I[0]
        b[:-1] = I[1:]; b[-1] = I[-1]
        Ix = r - l
        Iy = b - t
        return np.sqrt(Ix ** 2 + Iy ** 2).astype(np.uint8)

    def translation_matrix(self, dx, dy):
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        return M

    def find_shift(self, src, tar, x, y):
        h, w = tar.shape[:2]
        min_error = np.inf
        best_dx, best_dy = 0, 0
        Im_tar = self.gradient_magnitude(tar)
        Im_src = self.gradient_magnitude(src)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                Im_tmp_src = cv2.warpAffine(Im_src, self.translation_matrix(x + dx, y + dy), (w, h))

                error = np.sum(np.abs(np.sign(Im_tmp_src - Im_tar)))
                if error < min_error:
                    min_error = error
                    best_dx, best_dy = dx, dy

        return x + best_dx, y + best_dy

    def align(self, src, tar, depth):
        if depth == 0:
            dx, dy = self.find_shift(src, tar, 0, 0)

        else:
            h, w = src.shape[:2]
            half_src = cv2.resize(src, (w//2, h//2))
            half_tar = cv2.resize(tar, (w//2, h//2))
            prev_dx, prev_dy = self.align(half_src, half_tar, depth-1)
            dx, dy = self.find_shift(src, tar, prev_dx * 2, prev_dy * 2)

        return dx, dy

    def fit(self, src, tar, depth=4):
        h, w, c = tar.shape
        dx, dy = self.align(src, tar, depth)
        shift = self.translation_matrix(dx, dy)
        return cv2.warpAffine(src, shift, (w, h))


#
#   Paul Debevec's Method for HDR Imaging
#
class DebevecMethod(Weights):
    
    def __init__(self, wtype='triangle'):
        self.wtype = wtype
    
    def construct_A(self, cols, rows, W, ln_st, N, P, constraint, lamda):
        A = np.zeros([N * P + 255, 256 + N])
    
        A[cols, rows] = W

        for p in range(P):
            A[p * N : p * N + N, 256:] = -np.identity(N) * W[p * N : p * N + N]

        for i in range(254):
            A[N * P + i, i : i + 3] = np.array([1, -2, 1]) * np.abs(i - 127) * lamda

        A[-1, constraint] = 1

        return A.astype(np.float32)

    def construct_B(self, cols, rows, W, ln_st, N, P):
        B = np.zeros(N * P + 255)

        for p in range(P):
            B[p * N : p * N + N] = ln_st[p]

        B[cols] *= W

        return B.astype(np.float32)

    def construct_matrix(self, Z, ln_st, N, P, wtype, constraint=127, lamda=10):
        cols = np.arange(N * P)
        rows = np.array(Z).flatten()

        W = self.get_weights(rows, wtype)
        A = self.construct_A(cols, rows, W, ln_st, N, P, constraint, lamda)
        B = self.construct_B(cols, rows, W, ln_st, N, P)

        return A, B
    
    def solve(self, samples, ln_shutter_times, n_samples, n_images, wtype):
        A, B = self.construct_matrix(samples, ln_shutter_times, n_samples, n_images, wtype, constraint=127)
        
        A_inv = np.linalg.pinv(A)
        lnG = np.dot(A_inv, B)[:256]
        print('[Debevec] A inverse solved:', A_inv.shape)
        
        return lnG


#
#   Robertson' Method for HDR Imaging
#
class RobertsonMethod(Weights):
    
    def __init__(self, wtype='triangle'):
        self.wtype = wtype
        
    def fitE(self, Z, G, st):
        P = st.shape[0]
        Wz = self.get_weights(Z, wtype=self.wtype).reshape(P, -1) / 128
        Gz = G[Z].reshape(P, -1)

        upper = np.sum(Wz * Gz * st, axis=0).astype(np.float32)
        bottom = np.sum(Wz * st * st, axis=0).astype(np.float32)
        return upper / bottom
    
    def fitG(self, Z, G, E, st):
        P = st.shape[0]
        Z = Z.reshape(P, -1)
        Wz = self.get_weights(Z, wtype=self.wtype).reshape(P, -1) / 128
        Wz_Em_st = Wz * (E * st)

        for m in range(256):
            index = np.where(Z == m)
            upper = np.sum(Wz_Em_st[index]).astype(np.float32)
            lower = np.sum(Wz[index]).astype(np.float32)
            if lower > 0:
                G[m] = upper / lower

        G /= G[127]
        return G
    
    def solve(self, Z_bgr, initG, shutter_times, epochs=2):
        G_bgr = np.array(initG)
        st = shutter_times.reshape(-1, 1)
        colors = ['blue', 'green', 'red']

        for c in range(3):
            Z = np.array(Z_bgr[c])
            G = np.array(initG[c])

            for e in range(epochs):
                print('\r[Robertson] %s, epoch=%d' % (colors[c], e+1), end='    ')
                # Compute Ei (energy of each pixel)
                E = self.fitE(Z, G, st)
                # Compute Gm
                G = self.fitG(Z, G, E, st)

            G_bgr[c] = G

        print()

        return np.log(G_bgr).astype(np.float32)


#
#   Tone Mapping: Photographic Global / Local, Bilateral
#
class ToneMapping():
   
    def __init__(self):
        self.bgr_string = ['blue', 'green', 'red']
        
    def photographic_global(self, hdr, d=1e-6, a=0.5):
        print('[Photographic Global]')
        Lw = hdr
        Lw_ave = np.exp(np.mean(np.log(d + Lw)))
        Lm = (a / Lw_ave) * Lw
        Lm_max = np.max(Lm) # Lm_white
        Ld = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
        ldr = np.clip(np.array(Ld * 255), 0, 255)
        
        return ldr.astype(np.uint8)
        
    def gaussian_blurs(self, im, smax=25, a=0.5, fi=8.0, epsilon=0.01):
        cols, rows = im.shape
        blur_prev = im
        num_s = int((smax+1)/2)
        
        blur_list = np.zeros(im.shape + (num_s,))
        Vs_list = np.zeros(im.shape + (num_s,))
        
        for i, s in enumerate(range(1, smax+1, 2)):
            print('\r[Photographic Local] filter:', s, end=', ')
            blur = cv2.GaussianBlur(im, (s, s), 0)
            Vs = np.abs((blur - blur_prev) / (2 ** fi * a / s ** 2 + blur_prev))
            blur_list[:, :, i] = blur
            Vs_list[:, :, i] = Vs
        
        # 2D index
        print('find index...', end='')
        smax = np.argmax(Vs_list > epsilon, axis=2)
        smax[np.where(smax == 0)] = num_s
        smax -= 1
        
        # select blur size for each pixel
        print(', apply index...')
        I, J = np.ogrid[:cols, :rows]
        blur_smax = blur_list[I, J, smax]
        
        return blur_smax
        
    def photographic_local(self, hdr, d=1e-6, a=0.25):
        ldr = np.zeros_like(hdr, dtype=np.float32)
        Lw_ave = np.exp(np.mean(np.log(d + hdr)))
        
        for c in range(3):
            Lw = hdr[:, :, c]
            Lm = (a / Lw_ave) * Lw
            Ls = self.gaussian_blurs(Lm)
            Ld = Lm / (1 + Ls)
            ldr[:, :, c] = np.clip(np.array(Ld * 255), 0, 255)

        return ldr.astype(np.uint8)
    
    def durand_bilateral(self, hdr):
        ldr = np.zeros_like(hdr)

        b, g, r = 1, 40, 20

        Lw = (hdr[:, :, 0] * b + hdr[:, :, 1] * g + hdr[:, :, 2] * r) / (b + g + r)
        log_Lw = np.log10(Lw)
        log_base = cv2.bilateralFilter(log_Lw, 5, 15, 15)
        log_detail = log_Lw - log_base
        
        cf = 2 / (np.max(log_base) - np.min(log_base)) # compression factor
        log_Ld = cf * (log_base - np.max(log_base)) + log_detail
        Ld = np.power(10, log_Ld)

        ldr[:, :, 0] = (hdr[:, :, 0] / Lw) * Ld
        ldr[:, :, 1] = (hdr[:, :, 1] / Lw) * Ld
        ldr[:, :, 2] = (hdr[:, :, 2] / Lw) * Ld

        ldr = np.clip((ldr ** 0.3) * 255, 0, 255)
        
        return ldr.astype(np.uint8)
