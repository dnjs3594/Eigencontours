import cv2
import numpy as np

def decoding_theta(data, dim):
    out = dict()

    centers = np.array(list(data['center'])).astype(np.float32)
    bboxs = np.array(list(data['bbox'])).astype(np.float32)

    x, y, w, h = bboxs

    bboxs_x1 = x - w / 2  # 1
    bboxs_x2 = x + w / 2  # 1
    bboxs_y1 = y - h / 2  # 1
    bboxs_y2 = y + h / 2  # 1
    bboxsw = np.abs(bboxs_x2 - bboxs_x1)  # 1
    bboxsh = np.abs(bboxs_y2 - bboxs_y1)  # 1
    relative_lens = np.sqrt(bboxsw * bboxsw + bboxsh * bboxsh)  # 1
    center_xs = centers[0]  # 1
    center_ys = centers[1]  # 1

    idx = np.linspace(0, 360, dim, endpoint=False).astype(np.int32)
    rs = np.float32(data['r'])[idx]  # N, 360
    num = len(rs)
    rs = rs.astype(np.float32)  # N, 360
    # theta_list = np.arange(num - 1, -1, -1).reshape(num).astype(np.float32) # 360
    theta_list = np.flip(idx, axis=0).astype(np.float32)
    x, y = cv2.polarToCart(rs, theta_list, angleInDegrees=True)  # 360    360
    x = x + center_xs.astype(np.float32)  # 360
    y = y + center_ys.astype(np.float32)  # 360

    x = np.clip(x, bboxs_x1, bboxs_x2).reshape(num, 1)  # 360,1
    y = np.clip(y, bboxs_y1, bboxs_y2).reshape(num, 1)  # 360,1
    polygons = np.concatenate((x, y), axis=-1)  # 360,2

    out['contour_pts'] = polygons

    return out
