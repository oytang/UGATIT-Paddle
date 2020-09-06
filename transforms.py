import cv2
import math
import numbers
import traceback
import numpy as np
from paddle.fluid.layers import reshape, image_resize, flip, transpose
from paddle.fluid.dygraph import to_variable
from collections.abc import Iterable
from collections.abc import Sequence

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *data):
        for f in self.transforms:
            try:
                # multi-fileds in a sample
                if isinstance(data, Sequence):
                    data = f(*data)
                # single field in a sample, call transform directly
                else:
                    data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Resize(object):

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, img):
        input = reshape(img, shape=(1, img.shape[0], img.shape[1], img.shape[2]))
        out = image_resize(input, out_shape=self.size, data_format='NHWC')[0]
        return out

class RandomResizedCrop(object):

    def __init__(self,
                 output_size,
                 scale=(1.0, 1.0),
                 ratio=(1.0, 1.0)):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        assert (scale[0] <= scale[1]), "scale should be of kind (min, max)"
        assert (ratio[0] <= ratio[1]), "ratio should be of kind (min, max)"
        self.scale = scale
        self.ratio = ratio

    def _get_params(self, image, attempts=10):
        height, width, _ = image.shape
        area = height * width

        for _ in range(attempts):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = tuple(math.log(x) for x in self.ratio)
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                x = np.random.randint(0, width - w + 1)
                y = np.random.randint(0, height - h + 1)
                return x, y, w, h

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        x = (width - w) // 2
        y = (height - h) // 2
        return x, y, w, h

    def __call__(self, img):
        x, y, w, h = self._get_params(img)
        cropped_img = img[y:y + h, x:x + w]
        cropped_img = reshape(cropped_img, shape=(1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
        out = image_resize(input=cropped_img, out_shape=self.output_size, data_format='NHWC')[0]
        return out

class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            return flip(img, dims=[1]) # horizontal flip
        return img

class Normalize(object):

    def __init__(self, mean=0.0, std=1.0):
        if isinstance(mean, numbers.Number):
            mean = [mean, mean, mean]

        if isinstance(std, numbers.Number):
            mean = [std, std, std]

        self.mean = to_variable(np.array(mean, dtype=np.float32).reshape(len(mean), 1, 1))
        self.std = to_variable(np.array(std, dtype=np.float32).reshape(len(std), 1, 1))

    def __call__(self, img):
        return (img - self.mean) / self.std

class Permute(object):

    def __init__(self, mode="CHW", to_rgb=True):
        assert mode in [
            "CHW"
        ], "Only support 'CHW' mode, but received mode: {}".format(mode)
        self.mode = mode
        self.to_rgb = to_rgb

    def __call__(self, img):
        if self.to_rgb:
            img = img[..., ::-1]
        if self.mode == "CHW":
            CHW = transpose(img, (2, 0, 1))
            return CHW
        return img