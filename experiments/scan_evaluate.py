import sys
import pylidc as pl
import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
from preprocessing import scan_index_split
from keras.models import load_model
import pickle

sys.path.append('../')
from keras_retinanet import models
from keras_retinanet.utils.eval import _get_detections as get_fast_detection
from keras_retinanet.preprocessing.generator import Generator

FAST_DETECTION_PARM = {
    'path': 'working_models/augmented_smallanchor64.h5',
    'backbone': 'p3d',
    'convert_model': True
}

FPR_MODEL_PATH = 'fp_reduction/working_models/conv3d_amsgrad_rop_v2_098.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def generate_fpr_batch(volume, boxes, z):
    def extract_patch(volume, x, y, z, patch_size=(64, 64, 16)):
        xs = np.arange(-patch_size[0]//2, patch_size[0]//2, dtype=np.int) + x
        ys = np.arange(-patch_size[1]//2, patch_size[1]//2, dtype=np.int) + y
        zs = np.arange(-patch_size[2]//2, patch_size[2]//2, dtype=np.int) + z

        patch = volume.take( xs, mode='wrap', axis=0).take(
                            ys, mode='wrap', axis=1).take(
                            zs, mode='warp', axis=2)
        return patch

    batch = list()
    for d in boxes:
        patch = extract_patch(volume, int((d[0]+d[2])/2), int((d[1]+d[3])/2), int(z))
        patch = (patch - 418.) / 414.
        batch.append(patch)
    batch = np.array(batch)[..., np.newaxis]
    return batch

class ScanGenerator:
    def __init__(self, volume):
        def preprocess_image(image):
            """ Preprocess image and its annotations.
            """
            MEAN, STD = 174., 825.
            # image = (image - image.mean()) / image.std()
            image = (image - MEAN) / STD
            return image

        self.volume = volume
        self.z_indices = list(range(16//2, self.volume.shape[2]-16//2))
        self.preprocess_image = preprocess_image

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return 0

    def label_to_name(self, label):
        """ Map label to name.
        """
        return 'nodule'

    def size(self):
        return len(self.z_indices)

    def load_image(self, image_index, repeat=False):
        """ Load an image at the image_index.
        """
        z = self.z_indices[image_index]
        return self.volume[:,:,z-16//2: z+16//2].reshape(512, 512, 16, 1)

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return image, 1.

def predict(set, index):
    index_scan = scan_index_split(1018)[{'train': 0, 'valid': 1, 'test': 2}[set]][index]
    scans = pl.query(pl.Scan).filter()

    volume = scans[index_scan].to_volume()

    # load the models
    print('Loading model, this may take a second...')
    fast_detection_model = models.load_model(
                            FAST_DETECTION_PARM['path'],
                            backbone_name=FAST_DETECTION_PARM['backbone'],
                            convert=FAST_DETECTION_PARM['convert_model'])
    # fast_detection_model.summary()

    fpr_model = load_model(FPR_MODEL_PATH)
    # fpr_model.summary()

    fpr_model = keras.utils.multi_gpu_model(fpr_model)

    fast_detection_generator = ScanGenerator(volume)
    fast_detection = get_fast_detection(
                            generator=fast_detection_generator,
                            model=fast_detection_model,
                            score_threshold=0.05,
                            max_detections=60,
                            verbose=True,
                            save_path=None,
                            do_draw_annotations=False)

    result = list()
    for z in tqdm(np.arange(len(fast_detection)) + 8, desc='false positive reduction'):
        batch = generate_fpr_batch(volume, fast_detection[z-8][0], z)
        fpr_score = fpr_model.predict(batch, 64)
        for d, fs in zip(fast_detection[z-8][0], fpr_score[:, 0]):
            result.append([d[0], d[1], d[2] ,d[3], z, d[4], fs])
    result = np.array(result)
    result = result[result[:, -1].argsort()[::-1]]
    with open('result.pl', 'wb') as f:
        pickle.dump(result, f)

    return index_scan, volume, result


if __name__ == '__main__':
    predict('test', 85)
