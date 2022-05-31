import skimage.io as io
import skvideo.io as vio
import numpy as np
import os
import cv2


def __polish_reading(img, required_dtype, required_nc, channel_dim):  # non-functional, banning being used externally
    # channel_dim is 2 for image and 3 for video

    img = np.array(img)
    if required_dtype == 'float':
        if img.dtype is np.dtype('float'):
            pass
        elif img.dtype is np.dtype('uint8'):
            img = img.astype(np.float32) / 255.
        else:
            raise NotImplementedError
    elif required_dtype == 'uint8':
        if img.dtype is np.dtype('float'):
            img = img * 255.
            img = img.astype(np.uint8)
        elif img.dtype is np.dtype('uint8'):
            pass
        else:
            raise NotImplementedError
    elif required_dtype == 'uint16':
        if img.dtype is np.dtype(np.uint16):
            pass
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError('Unknown required_dtype: %s' % required_dtype)

    img = np.squeeze(img)
    if required_nc == 3:
        if img.ndim == channel_dim:
            img = np.expand_dims(img, axis=channel_dim)
        if img.shape[2] == 1:
            img = np.concatenate((img, img, img), axis=channel_dim)
    elif required_nc == 1:
        assert img.ndim == channel_dim
    else:
        raise NotImplementedError

    return img


def imread(image_path, required_dtype='float', required_nc=3):  # functional
    img = io.imread(image_path, as_gray=required_nc==1)
    img = __polish_reading(img, required_dtype, required_nc, 2)
    return img


def vread(video_path, num_frames=0, required_dtype='float', required_nc=3):  # functional
    video = vio.vread(video_path, num_frames=num_frames, as_gray=required_nc==1)
    video = __polish_reading(video, required_dtype, required_nc, 3)
    return video


def gifWrite(gif_path, rgbT):
    assert len(rgbT.shape) == 4 and rgbT.shape[3] == 3  # T * imageHeight * imageWidth * 3
    assert 0. <= rgbT.mean() <= 1.  # assume rgbT is between [0, 1]
    import imageio  # so python 2 cannot call this function
    writer = imageio.get_writer(gif_path, mode='I')
    for t in range(rgbT.shape[0]):
        writer.append_data((255. * rgbT[t]).astype(np.uint8))
    writer.close()
