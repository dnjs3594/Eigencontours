import torch
import numpy as np
import tqdm
import math
import os
import pickle
import shutil
import random
import shapely
from shapely.geometry import Polygon, MultiPoint
import numpy.polynomial.chebyshev as chebyshev
import cv2 as cv
# from davisinteractive.metrics.jaccard import batched_jaccard, batched_f_measure
from PIL import Image, ImageDraw
import pycocotools.mask as mask_util

global global_seed

global_seed = 123
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)


def _init_fn(worker_id):

    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

# convertor
def to_tensor(data):
    return torch.from_numpy(data).cuda()

def to_np(data):
    return data.cpu().numpy()

def to_np2(data):
    return data.detach().cpu().numpy()

def to_3D_np(data):
    return np.repeat(np.expand_dims(data, 2), 3, 2)

def logger(text, LOGGER_FILE):  # write log
    with open(LOGGER_FILE, 'a') as f:
        f.write(text),
        f.close()


# directory & file
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


# pickle
def save_pickle(dir_name, file_name, data):

    '''
    :param file_path: ...
    :param data:
    :return:
    '''
    mkdir(dir_name)
    with open(dir_name + file_name + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data

# create dict
def create_test_dict():

    out = {'cls': {},
           'reg': {},
           'pos': {}}  # detected lines

    # pred
    out['cls'] = torch.FloatTensor([]).cuda()
    out['reg'] = torch.FloatTensor([]).cuda()

    return out

def create_forward_step(num, batch_size):

    step = [i for i in range(0, num, batch_size)]
    if step[len(step) - 1] != num:
        step.append(num)

    return step

def record_config(cfg, logfile):
    logger("*******Configuration*******\n", logfile)

    data = {k: getattr(cfg, k) for k in cfg.__dir__() if '__' not in k}
    for key, value in data.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, int):
                    logger("%s : %s %d\n" % (key, k, v), logfile)
                elif isinstance(v, str):
                    logger("%s : %s %s\n" % (key, k, v), logfile)
                elif isinstance(v, float):
                    logger("%s : %s %f\n" % (key, k, v), logfile)

        elif isinstance(value, int):
            logger("%s : %d\n" % (key, value), logfile)
        elif isinstance(value, str):
            logger("%s : %s\n" % (key, value), logfile)
        elif isinstance(value, float):
            logger("%s : %f\n" % (key, value), logfile)

    # copy config file
    if cfg.run_mode == 'train' and cfg.resume == True:
        os.system('cp %s %s' %('./options/config.py', os.path.join(cfg.dir['out'] + 'train/log/config.py')))

def generate_datalist(cfg):
    if cfg.category == 'all':
        with open(os.path.join(cfg.dir['dataset'], 'list/test.txt')) as f:
            imglist = []
            for line in f:
                data = line.strip().split(" ")
                imglist.append(data[0][1:])

    else:
        with open(os.path.join(cfg.dir['dataset'], 'list/test_split/', cfg.category + '.txt')) as f:
            imglist = []
            for line in f:
                data = line.strip().split(" ")
                imglist.append(data[0])

    return imglist


def dict_sum(loss_dict1, loss_dict2, loss_dict3):
    new_dict = {}
    for key in loss_dict1.keys():
        new_dict[key] = loss_dict1[key] + loss_dict2[key] + loss_dict3[key]
    return new_dict

def parse_data_config(path: str):
    """데이터셋 설정 파일을 parse한다."""
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_classes(path: str):
    """클래스 이름을 로드한다."""
    with open(path, "r") as f:
        names = f.readlines()
    for i, name in enumerate(names):
        names[i] = name.strip()
    return names


def init_weights_normal(m):
    """정규분포 형태로 가중치를 초기화한다."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.1)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes_original(prediction, rescaled_size: int, original_size: tuple):
    """Rescale bounding boxes to the original shape."""
    ow, oh = original_size
    resize_ratio = rescaled_size / max(original_size)

    # 적용된 패딩 계산
    if ow > oh:
        resized_w = rescaled_size
        resized_h = round(min(original_size) * resize_ratio)
        pad_x = 0
        pad_y = abs(resized_w - resized_h)
    else:
        resized_w = round(min(original_size) * resize_ratio)
        resized_h = rescaled_size
        pad_x = abs(resized_w - resized_h)
        pad_y = 0

    # Rescale bounding boxes
    prediction[:, 0] = (prediction[:, 0] - pad_x // 2) / resize_ratio
    prediction[:, 1] = (prediction[:, 1] - pad_y // 2) / resize_ratio
    prediction[:, 2] = (prediction[:, 2] - pad_x // 2) / resize_ratio
    prediction[:, 3] = (prediction[:, 3] - pad_y // 2) / resize_ratio

    # 예측 결과가 원본 이미지의 좌표를 넘어가지 못하게 한다.
    for i in range(prediction.shape[0]):
        for k in range(0, 3, 2):
            if prediction[i][k] < 0:
                prediction[i][k] = 0
            elif prediction[i][k] > ow:
                prediction[i][k] = ow

        for k in range(1, 4, 2):
            if prediction[i][k] < 0:
                prediction[i][k] = 0
            elif prediction[i][k] > oh:
                prediction[i][k] = oh

    return prediction


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute the average precision, given the Precision-Recall curve.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Compute AP", leave=False):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_batch_statistics_box(outputs, targets, iou_threshold):
    """Compute true positives, predicted scores and predicted labels per batch."""
    batch_metrics = []
    polygon_pred = []
    polygon_gt = []

    for i, output in enumerate(outputs):

        if output is None:
            continue

        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, 6]

        pred_coeffs = output[:, 9:].numpy()
        pred_centers = output[:, 7:9].numpy()


        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:5]
            target_polygons_x = annotations[:, 8:-360].numpy()
            target_polygons_y = annotations[:, -360:].numpy()
            for pred_i, (pred_box, pred_label, pred_coeff, pred_center) in enumerate(zip(pred_boxes, pred_labels, pred_coeffs, pred_centers)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                # iou_= coef_polygon_iou(pred_coeff[np.newaxis], pred_center[np.newaxis], pred_box[np.newaxis], target_polygons_x, target_polygons_y)
                # iou, box_index = iou_.max(1), iou_.argmax(1)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def get_batch_statistics(outputs, targets, iou_threshold, scale):
    """Compute true positives, predicted scores and predicted labels per batch."""
    batch_metrics = []
    polygon_pred = []
    polygon_gt = []

    for i, output in enumerate(outputs):

        if output is None:
            continue

        pred_boxes = output[:, :4].numpy()
        pred_scores = output[:, 4]
        pred_labels = output[:, 6]

        pred_coeffs = output[:, 9:].numpy()
        pred_centers = output[:, 7:9].numpy()


        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:5]
            target_polygons_x = annotations[:, 8:-360].numpy()
            target_polygons_y = annotations[:, -360:].numpy()
            for pred_i, (pred_box, pred_label, pred_coeff, pred_center) in enumerate(zip(pred_boxes, pred_labels, pred_coeffs, pred_centers)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                iou_= coef_polygon_iou(pred_coeff, pred_center, pred_box, target_polygons_x, target_polygons_y, scale)
                iou, box_index = iou_.max(1), iou_.argmax(1)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def get_batch_statistics_contour(out_b, out_s, iou_threshold, targets):
    """Compute true positives, predicted scores and predicted labels per batch."""
    batch_metrics = {}
    for i in iou_threshold:
        batch_metrics["{:.2f}".format(i)] = []
    detected_boxes = {}
    for i in iou_threshold:
        detected_boxes["{:.2f}".format(i)] = []

    if out_b.shape[0] == 0:
        return batch_metrics

    pred_boxes = out_b[:, :4]
    pred_scores = out_b[:, 4]
    pred_labels = out_b[:, 5]
    pred_masks = np.array(out_s)

    annotations = targets['gt_masks'].data[0][0]
    target_labels = targets['gt_labels'].data[0][0]

    for i in iou_threshold:

        if len(annotations):
            true_positives = np.zeros(pred_boxes.shape[0])
            detected_boxes = []
            for pred_i in range(pred_boxes.shape[0]):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_labels[pred_i] not in target_labels:
                    continue

                # iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                iou_= coef_polygon_contour(pred_masks[pred_i:pred_i+1], annotations.numpy())
                iou, box_index = iou_.max(1), iou_.argmax(1)
                if iou >= i and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
            batch_metrics["{:.2f}".format(i)].append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """Returns the IoU of two bounding boxes."""
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def coef_polygon_iou(pred_coef_l, pred_center_l, pred_bbox_l, gt_points_xs_l, gt_points_ys_l, scale):
    """Calculate Intersection-Over-Union(IOU) of pred coefs(Reconstructed) and gt polygon points
    Parameters
    ----------
    pred_coef_l : numpy.ndarray
         An ndarray with shape :math'(N,2*deg+2)
    pred_bbox_l : numpy.ndarray
         An ndarray with shape :math'(N,4)  x1y1x2y2
    pred_center_l : numpy.ndarray
         An ndarray with shape :math'(N,2)  xy
    gt_points_xs_l : numpy.ndarray
         An ndarray with shape :math'(M,360)  x1, x2, x3,..., x360
    gt_points_ys_l : numpy.ndarray
         An ndarray with shape :math'(M,360)  y1, y2, y3,..., y360

    Returns
    ------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        polygons in `pred coef` and gt polygon points`.
    """

    gt_points_xs_l = gt_points_xs_l.reshape(-1, 360, 1)  # M, 360 ,1
    gt_points_ys_l = gt_points_ys_l.reshape(-1, 360, 1)  # M, 360 ,1
    polygon_bs = np.concatenate((gt_points_xs_l, gt_points_ys_l), axis=-1)  # M, 360 ,2
    polygon_as = coef_trans_polygon_ESR(pred_coef_l, pred_bbox_l, pred_center_l, scale)
    iou = polygon_iou(polygon_as, polygon_bs)

    return iou


def coef_polygon_contour(pred, gt):
    """Calculate Intersection-Over-Union(IOU) of pred coefs(Reconstructed) and gt polygon points
    Parameters
    ----------
    pred_coef_l : numpy.ndarray
         An ndarray with shape :math'(N,2*deg+2)
    pred_bbox_l : numpy.ndarray
         An ndarray with shape :math'(N,4)  x1y1x2y2
    pred_center_l : numpy.ndarray
         An ndarray with shape :math'(N,2)  xy
    gt_points_xs_l : numpy.ndarray
         An ndarray with shape :math'(M,360)  x1, x2, x3,..., x360
    gt_points_ys_l : numpy.ndarray
         An ndarray with shape :math'(M,360)  y1, y2, y3,..., y360

    Returns
    ------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        polygons in `pred coef` and gt polygon points`.
    """
    iou = mask_iou(pred, gt)

    return iou

def make_polygon(gt_polygon, pred_polygon):
    gt_num = gt_polygon.shape[0]
    pred_num = pred_polygon.shape[0]

    gt = np.array([], dtype=np.uint8).reshape(0, 416, 416)
    pred = np.array([], dtype=np.uint8).reshape(0, 416, 416)

    for i in range(gt_num):
        im = Image.new("L", (416, 416))
        ImageDraw.Draw(im).polygon(gt_polygon[i], fill=1, outline=True)
        mask = np.array(im)
        gt = np.concatenate((gt, mask[np.newaxis, ...]), axis=0)

    for j in range(pred_num):
        im = Image.new("L", (416, 416))
        ImageDraw.Draw(im).polygon(pred_polygon[j], fill=1, outline=True)
        mask = np.array(im)
        pred = np.concatenate((pred, mask[np.newaxis, ...]), axis=0)

    return gt, pred

def coef_trans_polygon(coefs, bboxs, centers):
    """Reconstruct the Objects Shape Polygons by Coefs and Centers
    Parameters
    ----------
    coefs : numpy.ndarray
         An ndarray with shape :math'(N,2*deg+2)
    bboxs : numpy.ndarray
         An ndarray with shape :math'(N,4)  x1y1x2y2
    centers : numpy.ndarray
         An ndarray with shape :math'(N,2)  xy
         It is the object predicted center.

    Return
    polygons : numpy.ndarray
         An ndarray with shape :math'(N,360,2)
    """

    bboxs_x1 = bboxs[:, 0].reshape(-1, 1)  # N,1
    bboxs_x2 = bboxs[:, 2].reshape(-1, 1)  # N,1
    bboxs_y1 = bboxs[:, 1].reshape(-1, 1)  # N,1
    bboxs_y2 = bboxs[:, 3].reshape(-1, 1)  # N,1
    bboxsw = np.abs(bboxs_x2 - bboxs_x1)  # N,1
    bboxsh = np.abs(bboxs_y2 - bboxs_y1)  # N,1
    relative_lens = np.sqrt(bboxsw * bboxsw + bboxsh * bboxsh)  # N,1
    center_xs = centers[:, 0].reshape(-1, 1)  # N,1
    center_ys = centers[:, 1].reshape(-1, 1)  # N,1
    rs = cheby(coefs) * relative_lens  # N, 360
    rs = rs.astype(np.float32)  # N, 360
    theta_list = np.arange(359, -1, -1).reshape(1, 360)  # 1, 360
    theta_list = theta_list.repeat(int(rs.shape[0]), axis=0).astype(np.float32)  # N,360
    x, y = cv.polarToCart(rs, theta_list, angleInDegrees=True)  # N,360    N,360
    x = x + center_xs.astype(np.float32)  # N.360
    y = y + center_ys.astype(np.float32)  # N,360

    x = np.clip(x, bboxs_x1, bboxs_x2).reshape(-1, 360, 1)  # N,360,1
    y = np.clip(y, bboxs_y1, bboxs_y2).reshape(-1, 360, 1)  # N,360,1
    polygons = np.concatenate((x, y), axis=-1)  # N,360,2

    return polygons


def coef_trans_polygon_ESR(coefs, bboxs, centers, scale):
    """Reconstruct the Objects Shape Polygons by Coefs and Centers
    Parameters
    ----------
    coefs : numpy.ndarray
         An ndarray with shape :math'(N,2*deg+2)
    bboxs : numpy.ndarray
         An ndarray with shape :math'(N,4)  x1y1x2y2
    centers : numpy.ndarray
         An ndarray with shape :math'(N,2)  xy
         It is the object predicted center.

    Return
    polygons : numpy.ndarray
         An ndarray with shape :math'(N,360,2)
    """
    top_k = 18
    U = to_np(load_pickle("../../data/vocsbdche/pickle_180_20/U"))[:, :top_k]
    # pickle_path = "../../data/vocsbdche/pickle_SBD"
    # norm = load_pickle(os.path.join(pickle_path, "offset_distribution"))
    # mean_c = norm['mean']
    # std_c = np.sqrt(norm['var'])
    max_dist = 500 * math.sqrt(2) / 2
    bboxs_x1 = bboxs[0].reshape(-1, 1)  # N,1
    bboxs_x2 = bboxs[2].reshape(-1, 1)  # N,1
    bboxs_y1 = bboxs[1].reshape(-1, 1)  # N,1
    bboxs_y2 = bboxs[3].reshape(-1, 1)  # N,1
    center_xs = centers[0].reshape(-1, 1)  # N,1
    center_ys = centers[1].reshape(-1, 1)
    # coefs = coefs * std_c + mean_c
    rs = np.matmul(U, coefs) * max_dist * scale[0]
    rs = rs.astype(np.float32)[np.newaxis, :]  # N, 360
    theta_list = np.arange(358, -1, -2).reshape(1, 180)  # 1, 360 1, 360
    theta_list = theta_list.repeat(int(rs.shape[0]), axis=0).astype(np.float32)  # N,360
    x, y = cv.polarToCart(rs, theta_list, angleInDegrees=True)  # N,360    N,360
    x = x + center_xs.astype(np.float32)  # N.360
    y = y + center_ys.astype(np.float32)  # N,360

    x = np.clip(x, bboxs_x1, bboxs_x2).reshape(-1, 180, 1)  # N,360,1
    y = np.clip(y, bboxs_y1, bboxs_y2).reshape(-1, 180, 1)  # N,360,1
    polygons = np.concatenate((x, y), axis=2)

    # polygons = np.stack((x, y), axis=2)  # N,360,2
    return polygons


def polygon_iou(polygon_as, polygon_bs, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two polygons

    Parameters
    ----------
    polygon_as : numpy.ndarray
         An ndarray with shape :math'(N, polygon_a_nums, 2)
    polygon_bs : numpy.ndarray
         An ndarray with shape :math'(M, polygon_b_nums, 2)
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

    Returns
    ------
    numpy.ndarray
        An An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        polygons in `polygon_as` and `polygon_bs`.
    This way is not need the points_num is equal
    """
    N = polygon_as.shape[0]
    M = polygon_bs.shape[0]
    polygon_ious = np.zeros((N, M))
    for n in range(N):
        polygon_a = polygon_as[n]
        polya = Polygon(polygon_a).convex_hull
        for m in range(M):
            polygon_b = polygon_bs[m]
            polyb = Polygon(polygon_b).convex_hull
            try:
                inter_area = polya.intersection(polyb).area
                union_poly = np.concatenate((polygon_a, polygon_b))
                union_area = MultiPoint(union_poly).convex_hull.area
                if union_area == 0 or inter_area == 0:
                    iou = 0
                else:
                    iou = float(inter_area) / union_area
                polygon_ious[n][m] = iou
            except shapely.geos.TopologicalError:
                print("shapely.geos.TopologicalError occured, iou set to 0")
                polygon_ious[n][m] = 0
                continue
    return polygon_ious

def mask_iou(polygon_as, polygon_bs, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two polygons
    """
    N = polygon_as.shape[0]
    M = polygon_bs.shape[0]
    polygon_ious = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            inter_area = (polygon_as[n] + polygon_bs[m] == 2).sum()
            union_area = (polygon_as[n] + polygon_bs[m] != 0).sum()
            if union_area == 0 or inter_area == 0:
                iou = 0
            else:
                iou = float(inter_area) / union_area
            polygon_ious[n][m] = iou
    return polygon_ious

def cheby(coef):
    """
    coef numpy.addary with shape (N , 2*deg+2) such as (N,18), (N,26)
    theta nuumpy.addary with shape (360,)    [-1,1]

    Return numpy.array woth shape (N,360)
    """
    theta = np.linspace(-1, 1, 360)
    coef = coef.T
    r = chebyshev.chebval(theta, coef)

    return r

def  non_max_suppression(prediction, conf_thres, nms_thres):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:25].max(1)[0]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:25].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float(), image_pred[:, 25:]), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, 6] == detections[:, 6]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, device):
    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)
    class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float, device=device)

    tcoeffx = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tcoeffy = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tcoeff = torch.zeros(nB, nA, nG, nG, 18, dtype=torch.float, device=device)

    # Convert to position relative to box
    target_boxes = target[:, 2:6].clone() * nG
    gtcoeffxy = target[:, 6:8].clone() * nG
    gtcoeff = target[:, 8:-3].clone()
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    _, best_ious_idx = ious.max(0)

    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    gi[gi < 0] = 0
    gj[gj < 0] = 0
    gi[gi > nG - 1] = nG - 1
    gj[gj > nG - 1] = nG - 1

    gtcoeffx, gtcoeffy = (gtcoeffxy[:, 0] - (gx - gw / 2.0), gtcoeffxy[:, 1] - (gy - gh / 2.0))

    # Set masks
    obj_mask[b, best_ious_idx, gj, gi] = 1
    noobj_mask[b, best_ious_idx, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    tcoeffx[b, best_ious_idx, gj, gi] = torch.log(gtcoeffx / anchors[best_ious_idx][:, 0] + 1e-16)
    tcoeffy[b, best_ious_idx, gj, gi] = torch.log(gtcoeffy / anchors[best_ious_idx][:, 1] + 1e-16)
    tcoeff[b, best_ious_idx, gj, gi] = gtcoeff

    # One-hot encoding of label
    tcls[b, best_ious_idx, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, tcoeffx, tcoeffy, tcoeff

def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute the average precision, given the Precision-Recall curve.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Compute AP", leave=False):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")