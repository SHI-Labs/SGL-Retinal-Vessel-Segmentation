import random
import cv2
import numpy as np
import skimage.color as sc
from PIL import Image, ImageDraw
import torch
import math
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img, vessel, ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    
    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
    
    #Geometric Transformation
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    vessel = cv2.warpAffine(vessel,Rot_M,(cols,rows))
    vessel = cv2.warpAffine(vessel,Trans_M,(cols,rows))
    vessel = cv2.warpAffine(vessel,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def pixel_shuffle_ds(im):
    img_shape = im.shape
    im = np.array(im)
    H = img_shape[0]
    W = img_shape[1]
    C = img_shape[2]  #total channels
    out = np.zeros((int(H/2), int(W/2), 4*C))
    for i in range(C):
        out_tmp = np.concatenate((np.expand_dims(im[0:H:2, 0:W:2,i], axis=2),
                              np.expand_dims(im[0:H:2, 1:W:2,i], axis=2),
                              np.expand_dims(im[1:H:2, 1:W:2,i], axis=2),
                              np.expand_dims(im[1:H:2, 0:W:2,i], axis=2)), axis=2)
        out[:,:,i*4:i*4+4] = out_tmp
    return out

def add_AWGN(hr, level):
    w, h = hr.shape[:2]
    gauss = np.zeros((w, h, 3))
    for chn in range(3):
        gauss[:,:,chn] = np.random.normal(0, level, (w, h))
    noisy = hr + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy

# For inpainting
def process(image, opt=2):
    luma = cv2.cvtColor(image[:,:,::-1], cv2.COLOR_BGR2YUV)[:,:,0]
    if opt==1:
        equ = cv2.equalizeHist(luma) 
    else:
        clahe = cv2.createCLAHE(clipLimit = 3) 
        equ= clahe.apply(luma)
    return equ 

def stage_process(image, mask):
    m = mask.copy()
    m = m[:,:]
    vessel_seg = image * m
    non_vessel = image * (1 - m)
    vessel_seg = vessel_seg * 0.1
    non_vessel = cv2.GaussianBlur(non_vessel,(5,5), 0) 
    non_vessel = np.expand_dims(non_vessel, 2)
    return vessel_seg * m + non_vessel * (1 - m)

def compute_dismap(mask, ind=0.99, dual=False):
    dist = cv2.distanceTransform((mask*255).astype(np.uint8), cv2.DIST_L2, 5)
    expo = np.log(ind) * dist
    dist = np.exp(expo)
    if dual:
        dist2 = cv2.distanceTransform(((1-mask)*255).astype(np.uint8), cv2.DIST_L2, 5)
        expo2 = np.log(ind) * dist2
        dist2 = np.exp(expo2)
        dist = dist * dist2  #dual mode

    return dist


def ttsep(ve_map):
    filterSize =(3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)
    timage = cv2.erode(ve_map,kernel,iterations = 1) #erase thin vessel
    dia_timage = cv2.dilate(timage,kernel,iterations = 1)  #dilate the vessel again
    dia_timage = np.expand_dims(dia_timage, 2)
    thin = ve_map * (ve_map - dia_timage)  #thin vessel only
    thick = ve_map - thin  #the left thick map
    return thin, thick

def get_patch(hr, ve_r, ma, te_r, patch_size=256, deform = True, train=True, random_toggle=False):
    ve = np.zeros((hr.shape[:2]))
    ve[ve_r > 200] = 1
    ma[ma > 125] = 1
    te = te_r / 255.
    ih, iw = hr.shape[:2]
    if train:
        ih, iw = hr.shape[:2]
        ip = patch_size
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        #crop patch
        hr = hr[iy:iy + ip, ix:ix + ip, :]
        #hr = process(hr)
        ve = ve[iy:iy + ip, ix:ix + ip]
        te = te[iy:iy + ip, ix:ix + ip][..., None]
        ma = ma[iy:iy + ip, ix:ix + ip]
        ma = np.expand_dims(ma, 2)
        if deform and np.random.rand() >0.5:#< 0.5:  #whether to deform the inputs?
            im_merge = np.concatenate((hr, ve[...,None], te), axis=2)
            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)
            # The elaster augmentation after cropping may have some problems
            hr_t = im_merge_t[...,0:3]
            ve_t = im_merge_t[...,3]
            te_t = im_merge_t[...,4]
            ve_t[ve_t>0.5] = 1  #thresholding
            #hr_t = np.expand_dims(hr_t, 2)
            ve_t = ve_t[..., None]
            te_t[te_t>0.5] = 1  #whether this will influence the performance?
        else:
            hr_t = hr#np.expand_dims(hr, 2)
            ve_t = np.expand_dims(ve, 2)
            te_t = te
        #med = stage_process(hr_t, ve_t)
        #ret = [hr_t, ve_t, med, ma]
        dp = compute_dismap(ve_t, 0.99, True)
        if random_toggle == True:
            toggle_reg = dp.copy()
            toggle_reg[dp>=0.99] = 1
            toggle_reg[dp<0.99] = 0  #select the toggle regions
            a,b = dp.shape
            rand_toggle_map = np.random.rand(a,b)
            rand_toggle_map[rand_toggle_map>=0.3] = 1
            rand_toggle_map[rand_toggle_map<0.3] = 0
            rand_toggle_map *= toggle_reg
            rand_toggle_map = np.expand_dims(rand_toggle_map, 2)
            ve_t[np.logical_and(rand_toggle_map==1, ve_t==1)] = 0  #toggle
            ve_t[np.logical_and(rand_toggle_map==1, ve_t==0)] = 1  #toggle
        ve_thin, ve_thick = ttsep(ve_t)
        ret = [hr_t, ve_t, ma, te_t, dp, ve_thin, ve_thick]
    else:
    #if True:
        hr_img = np.zeros((608, 608, 3))
        hr_img[0:ih, 0:iw, :] = hr#process(hr)
        ve_img = np.zeros((608, 608))
        ve_img[0:ih, 0:iw] = ve
        ma_img = np.zeros((608, 608))
        ma_img[0:ih, 0:iw] = ma
        #hr_img = np.expand_dims(hr_img, 2)
        ve_img = np.expand_dims(ve_img, 2)
        ma_img = np.expand_dims(ma_img, 2)
        #med = stage_process(hr_img, ve_img)
        dp_img = compute_dismap(ve_img, 0.99, True)
        ve_thin, ve_thick = ttsep(ve_img)
        ret = [hr_img, ve_img, ma_img, ve_img, dp_img, ve_thin, ve_thick]
        #ret = [hr_img, ve_img, med, ma_img]
    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        #if n_channels == 1 and c == 3:
        #    img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        #elif n_channels == 3 and c == 1:
        #    img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255, single_test=False):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        if single_test:
            np_transpose = np.expand_dims(np_transpose, 0)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(hr,ve,ma, te, hflip=True, rot=True):
    #current version does not support augmentation
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    def _augment(hr, ve, ma, te):
        if hflip: 
            hr = hr[:, ::-1, :]
            ve = ve[:, ::-1]
            te = te[:, ::-1]
            ma = ma[:, ::-1]
        if vflip: 
            hr= hr[::-1, :, :]
            ve= ve[::-1, :]
            te= te[::-1, :]
            ma = ma[::-1, :]
        if rot90: 
            hr = hr.transpose(1, 0, 2)
            ve = ve.transpose(1, 0)
            te = te.transpose(1, 0)
            ma = ma.transpose(1,0)
        
        return hr, ve, ma, te

    return _augment(hr, ve, ma, te)

def raw_augment(*args):
    '''
    This codes partially borrow from Megvii Paper
    augment the images in all the images inside the batch
    '''
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    transpose = random.random() < 0.5
    def flip(bayer):
        if vflip and hflip:
            bayer = bayer[::-1, ::-1]
            bayer = bayer[1:-1, 1:-1]
        elif vflip:
            bayer = bayer[::-1]
            bayer = bayer[1:-1] 
        elif hflip:
            bayer = bayer[:, ::-1]
            bayer = bayer[:, 1:-1]  
        if transpose:
            bayer = np.transpose(bayer, (1, 0, 2))
        return bayer
    return [flip(a) for a in args] 

#generate mask
def add_mask(image):
    w, h = image.shape[:2]
    mask = brush_stroke_mask(w, h)  #(w, h, 1)
    mask_t = np.tile(mask, (1,1,3))
    result = image * (1- mask)
    return result, mask, image 
   


def brush_stroke_mask(W, H):
    """Generate random brush and stroke mask.
    Return a mask of (1, 1, W, H)
    partially fork from Jiahui Yu's Codes
    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    def generate_mask(W, H):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (W, H, 1))
        return mask

    return generate_mask(W, H)





