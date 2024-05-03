"""Streaming images and labels from datasets created with dataset_tool.py."""
from __future__ import print_function

import os
import os.path
import zipfile
import pickle as pkl
import random
import math

import PIL.Image
import cv2
import numpy as np

from training.dataset import Dataset

try:
    import pyspng
except ImportError:
    pyspng = None


class PairedMyxo(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 resize_by=1.,
                 mode="random",
                 pair_mode="same",
                 normalize=True,
                 norm_min=-1,
                 norm_max=1,
                 num_output=2,
                 return_name=False,
                 label_dict="/home/xavier/Documents/dataset/Welch/trainingset2/InceptionV3-labels.pkl",
                 use_rgb=False,
                 brightness_norm=.5,
                 brightness_mean=107.2,
                 brightness_std=5.8,
                 flip_rate=.5,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._zipfile = None
        self.resize_by = resize_by
        self.mode = mode
        self.pair_mode = pair_mode
        self.return_name = return_name
        self.label_dict = label_dict
        self.use_rgb = use_rgb
        self.brightness_norm = brightness_norm
        self.brightness_mean = brightness_mean
        self.brightness_std = brightness_std
        self.flip_rate = flip_rate
        self.crop_size = resolution
        self.crop_diag = np.sqrt(np.square(resolution) * 2)
        self.normalize = normalize
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.num_output = num_output
        self._image_labels = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if self.label_dict:
            label_dict = pkl.load(open(self.label_dict, "rb"))
            all_images = {}
            for phenotype in label_dict:
                for image_name in label_dict[phenotype]:
                    all_images[image_name] = phenotype
            self._image_fnames = []
            self._image_labels = []
            for image_name in all_images:
                if image_name in self._all_fnames:
                    if self._file_ext(image_name) in PIL.Image.EXTENSION:
                        self._image_fnames.append(image_name)
                        self._image_labels.append(all_images[image_name])
                else:
                    raise IOError(f'The image file {image_name} does not exist')
            self._image_labels = np.array(self._image_labels)
            del all_images
        else:
            self._image_fnames = sorted(
                fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + [1, self.resolution, self.resolution]
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    @property
    def resolution(self):  # The function becomes redundant.
        return self.crop_size

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = cv2.imread(os.path.join(self._path, fname),
                                   cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED loads the image as is, including alpha channel if present

        # Convert from BGR to RGB (OpenCV loads images in BGR)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Handle uint16 images
        if image.dtype == np.uint16:
            image = np.uint8(image / 256)

        # Resize if necessary
        if self.resize_by != 1.:
            shape = (int(image.shape[1] * self.resize_by), int(image.shape[0] * self.resize_by))
            image = cv2.resize(image, shape, cv2.INTER_LANCZOS4)

        # Adjust brightness
        if random.random() < self.brightness_norm:
            target = np.random.normal(self.brightness_mean, self.brightness_std)
            obj_v = np.mean(image)
            value = target - obj_v
            image = cv2.add(image, value)

        # Convert to/from RGB/Grayscale based on use_rgb flag
        if self.use_rgb and len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif not self.use_rgb and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def _load_raw_labels(self):
        label_dict = pkl.load(open(self.label_dict, "rb"))
        name2label = {}
        for item in label_dict:
            for img_name in label_dict[item]:
                name2label[img_name] = item
        labels = [name2label[fname] for fname in self._image_fnames]
        return np.array(labels)

    def rotate_image(self, image):
        # Get the image size
        image_size = image.shape
        # print(image_size)
        rot_images = []
        angles = np.random.uniform(-180, 180, self.num_output)
        for i in range(self.num_output):
            angle = angles[i]
            new_len = np.max([abs(self.crop_diag * np.cos((angle + 45) / 180 * np.pi)),
                              abs(self.crop_diag * np.sin((angle + 45) / 180 * np.pi))])
            new_len = int(math.ceil(new_len))
            x_sample = np.random.randint(new_len // 2, image_size[0] - np.ceil(new_len / 2))
            y_sample = np.random.randint(new_len // 2, image_size[1] - np.ceil(new_len / 2))

            x_min = min([x_sample, 0])
            y_min = min([y_sample, 0])

            cropped_img = image[x_min:x_min + new_len, y_min:y_min + new_len]
            # Vertical flip
            if random.random() < self.flip_rate:
                cropped_img = cropped_img[::-1, :]
            # Compute rotation matrix
            h, w = cropped_img.shape[:2]
            cropped_center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(cropped_center, angle, 1)
            # Rotate the image
            rot_img = cv2.warpAffine(cropped_img, rotation_matrix, (w, h))
            # rot_img = np.array(PIL.Image.fromarray(image[x_min:x_min + new_len, y_min:y_min + new_len]).rotate(angle))
            # ndimage.rotate(image[x_min:x_min + new_len, y_min:y_min + new_len], angle, reshape=True, mode='reflect')
            rot_img_shape = rot_img.shape
            img_center = [rot_img_shape[0] // 2, rot_img_shape[1] // 2]
            rot_img = rot_img[img_center[0] - self.resolution // 2:img_center[0] + self.resolution // 2,
                      img_center[1] - self.resolution // 2:img_center[1] + self.resolution // 2]
            rot_images.append(rot_img)
        rot_images = np.array(rot_images)
        return rot_images

    def rot90_image(self, image):
        # Get the image size
        image_size = image.shape
        rot_images = []
        for i in range(self.num_output):
            x_sample = np.random.randint(0, image_size[0] - self.resolution)
            y_sample = np.random.randint(0, image_size[1] - self.resolution)
            rot_img = image[x_sample:x_sample + self.resolution, y_sample:y_sample + self.resolution]
            rot_img = np.rot90(rot_img, k=np.random.randint(4))
            if np.random.rand() > 0.5:
                rot_img = np.fliplr(rot_img)
            rot_images.append(rot_img)
        rot_images = np.array(rot_images)
        return rot_images

    def crop_image(self, image):
        image_size = image.shape
        img_center = [image_size[0] // 2, image_size[1] // 2]
        img_out = image[img_center[0] - self.resolution // 2:img_center[0] + self.resolution // 2,
                  img_center[1] - self.resolution // 2:img_center[1] + self.resolution // 2]
        img_out = np.repeat(img_out[np.newaxis, :], self.num_output, axis=0)
        return img_out

    def lrcrop(self, image):
        image_size = image.shape
        img_center = [image_size[0] // 2, image_size[1] // 2]
        img_out = []
        for i in range(self.num_output):
            if i % 2:
                img_tmp = image[:self.resolution,
                          img_center[1] - self.resolution // 2:img_center[1] + self.resolution // 2]
            else:
                img_tmp = image[-self.resolution:,
                          img_center[1] - self.resolution // 2:img_center[1] + self.resolution // 2]
            img_out.append(img_tmp)
        img_out = np.array(img_out)
        return img_out

    def get_label(self, idx):
        label = self._image_labels[self._raw_idx[idx]]
        return label.copy()

    def __getitem__(self, idx, eps=1e-8):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8

        if self.mode == "random":
            images = self.rotate_image(image)
        elif self.mode == "rot90":
            images = self.rot90_image(image)
        elif self.mode == 'lrcrop':
            images = self.lrcrop(image)
        else:
            images = self.crop_image(image)
        if self.pair_mode == "replicate":
            # Get the index of the image replicate
            fname = self._image_fnames[self._raw_idx[idx]]
            # print(fname)
            fname_scope = int(fname.split("Scope")[-1][:2])
            pairroot = os.path.dirname(os.path.dirname(fname))
            pair_scopes = [int(scope[-2:]) for scope in os.listdir(os.path.join(self._path, pairroot))]
            pair_scopes.remove(fname_scope)
            rep_scope = random.choice(pair_scopes)
            rep_name = fname.replace("cope%.2d" % fname_scope, "cope%.2d" % rep_scope)
            rep_name = rep_name.replace("cope%d" % fname_scope, "cope%d" % rep_scope)
            rep_idx = self._image_fnames.index(rep_name)
            rep_image = self._load_raw_image(rep_idx)
            if self.mode == "random":
                rep_images = self.rotate_image(rep_image)
            elif self.mode == "rot90":
                rep_images = self.rot90_image(rep_image)
            else:
                rep_images = self.crop_image(rep_image)
            images = np.concatenate([images, rep_images], axis=0)

        # image1 = (image1[np.newaxis, :] - image1.mean()) / (image1.std() + eps)
        # image2 = (image2[np.newaxis, :] - image2.mean()) / (image2.std() + eps)
        # Normalize images to [0, 1]
        if len(images.shape) == 4:
            images = images.transpose(0, 3, 1, 2)
        else:
            images = images[:, np.newaxis, :]

        if self.normalize:
            images = images / (255 / (self.norm_max - self.norm_min)) + self.norm_min
        if self._xflip[idx]:
            assert images.ndim == 4  # BCHW
            images = images[:, :, :, ::-1]
        label = self._image_fnames[idx] if self.return_name else self.get_label(idx)

        # img_items = [image_item.copy() for image_item in images]
        return *images, label

# ----------------------------------------------------------------------------
