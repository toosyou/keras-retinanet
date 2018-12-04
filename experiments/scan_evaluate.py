import sys
import pylidc as pl
import numpy as np
import keras
import tensorflow as tf
from tqdm import tqdm
import keras
from keras import Model
from keras.backend.tensorflow_backend import set_session
from preprocessing import scan_index_split
from keras.models import load_model
import pickle
import itertools

sys.path.append('../')
from keras_retinanet import models
from keras_retinanet.utils.eval import _get_detections as get_fast_detection
from keras_retinanet.preprocessing.generator import Generator

sys.path.append('/home/toosyou/projects/LungTumor')
import data_util

FAST_DETECTION_PARM = {
    'path': 'working_models/augmented_smallanchor64.h5',
    'backbone': 'p3d',
    'convert_model': True
}

FPR_MODEL_PATH = 'fp_reduction/working_models/resnet3d_amsgrad_097.h5'
# FPR_MODEL_PATH = 'fp_reduction/model_checkpoints/95.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def false_positive_reduction(volume, fast_detection, fpr_model):
    def extract_patch(volume, x, y, z, patch_size=(64, 64, 16)):
        xs = np.arange(-patch_size[0]//2, patch_size[0]//2, dtype=np.int) + x
        ys = np.arange(-patch_size[1]//2, patch_size[1]//2, dtype=np.int) + y
        zs = np.arange(-patch_size[2]//2, patch_size[2]//2, dtype=np.int) + z

        patch = volume.take( ys, mode='wrap', axis=0).take(
                                xs, mode='wrap', axis=1).take(
                                zs, mode='warp', axis=2)
        return patch

    # remove predictions outside the lung
    lung_mask = data_util.lung_mask(volume, times_dilation=20, times_erosion=15, verbose=True)
    if lung_mask.any():
        for z in np.arange(len(fast_detection)) + 8:
            tmp_detection = list()
            for d in fast_detection[z-8][0]:
                if lung_mask[int((d[1]+d[3])/2), int((d[0]+d[2])/2), z]:
                    tmp_detection.append(d)
            fast_detection[z-8][0] = tmp_detection

    # generate batches for fpr
    result = list()
    batches = list()
    for z in tqdm(np.arange(len(fast_detection)) + 8, desc='generation fpr batches'):
        for d in fast_detection[z-8][0]:
            batches.append(extract_patch(volume, int((d[0]+d[2])/2), int((d[1]+d[3])/2), int(z)))
            result.append([d[0], d[1], d[2] ,d[3], z, d[4]])

    batches = np.array(batches)[..., np.newaxis]
    batches = (batches - 418.) / 414. # normalize

    # do the prediction
    print('making fpr predictions...')
    fpr_score = fpr_model.predict(batches, 128)
    for i, fs in enumerate(fpr_score[:, 0]):
        result[i].append(fs)

    # sort by the score
    result = np.array(result)
    result = result[result[:, -1].argsort()[::-1]]

    return result

# Malisiewicz et al.
def nms_2d(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, -1]

    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #按照score置信度降序排序
    order = scores.argsort()[::-1]

    keep = [] #保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i) #保留该类剩余box中得分最高的一个
        #得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    return dets[keep]

def group_nodule(result, nms2d_threshold=0.1, group3d_threshold=0.1):
    # group by z axis
    layer_result = np.array( [result[result[:, 4] == i] for i in range(8, int(result[:, 4].max()+1))] )

    nodules = list()
    last_boxes, last_box_nodule_index = None, None
    for i in range(layer_result.shape[0]):
        nmsed_boxes = nms_2d(layer_result[i], nms2d_threshold)
        if i == 0: # treat every predict as a nodule
            for b in nmsed_boxes:
                nodules.append([b])
            box_nodule_index = list(range(nmsed_boxes.shape[0]))
        else:
            box_nodule_index = [-1] * nmsed_boxes.shape[0]
            last_areas = (last_boxes[:, 2] - last_boxes[:, 0] + 1) * (last_boxes[:, 3] - last_boxes[:, 1] + 1)# (x2 - x1 + 1) * (y2 - y1 + 1)
            for index_box, b in enumerate(nmsed_boxes):
                # calcualte iou between this and last layer
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)

                xx1 = np.maximum(x1, last_boxes[:, 0])
                yy1 = np.maximum(y1, last_boxes[:, 1])
                xx2 = np.minimum(x2, last_boxes[:, 2])
                yy2 = np.minimum(y2, last_boxes[:, 3])

                #计算相交的面积,不重叠时面积为0
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
                ovr = inter / (areas + last_areas- inter)

                if ovr.max() > group3d_threshold: # merge to a existing nodule
                    index_max_overlap = ovr.argmax()
                    index_nodule = last_box_nodule_index[index_max_overlap]
                    nodules[index_nodule].append(b)
                    box_nodule_index[index_box] = index_nodule
                else: # new nodule
                    box_nodule_index[index_box] = len(nodules)
                    nodules.append([b])

        last_box_nodule_index = box_nodule_index
        last_boxes = nmsed_boxes

    # calculate the nodule score
    nodule_scores = np.zeros((len(nodules), 3))
    for index_nodule, n in enumerate(nodules):
        boxes = np.array(n).reshape((-1, 7))
        nodule_scores[index_nodule][0] = boxes[:, -2].mean()
        nodule_scores[index_nodule][1] = boxes[:, -1].mean()
        nodule_scores[index_nodule][2] = np.unique(boxes[:, 4]).shape[0] # number of layers crossed

    # sort nodules by score
    sorted_index = nodule_scores[:, 1].argsort()[::-1]
    nodules = np.array(nodules)[sorted_index]
    nodule_scores = nodule_scores[sorted_index]

    return nodules, nodule_scores

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
                            nms=True,
                            convert=FAST_DETECTION_PARM['convert_model'])
    # fast_detection_model.summary()

    fpr_model = load_model(FPR_MODEL_PATH)
    fpr_model = keras.utils.multi_gpu_model(fpr_model)

    fast_detection_generator = ScanGenerator(volume)
    fast_detection = get_fast_detection(
                            generator=fast_detection_generator,
                            model=fast_detection_model,
                            score_threshold=0.05,
                            max_detections=30,
                            verbose=True,
                            save_path=None,
                            do_draw_annotations=False)

    # false positive reduction
    result = false_positive_reduction(volume, fast_detection, fpr_model)

    with open('result.pl', 'wb') as f:
        pickle.dump(result, f)

    return index_scan, volume, result


if __name__ == '__main__':
    result = pickle.load(open('result.pl', 'rb'))
    nodules, nodule_scores = group_nodule(result)
    print(nodule_scores)
    # predict('test', 85)
