import cv2 as cv 
from collections import Counter
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def getTime(noFrame, totalNoFrames, durationInSeconds): 
    return (noFrame / totalNoFrames) * durationInSeconds

def is_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2)


def text_size(box): 
    _, _, _, h = box 
    return h

def quantization(img, n_cluster = 2): 
    h, w = img.shape[:2]
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    clt = MiniBatchKMeans(n_clusters = n_cluster)
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h, w, 3))

    quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
    
    return quant

def fw_preprocess(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3),0)   
    return blur 
     

def text_color(img, n_cluster = 2): 
    q_img = quantization(img, cluster = n_cluster)
    img_arr = np.array(q_img)
    pixels_arr = (img_arr[:, :, 0] + img_arr[:, :, 1]*256 + img_arr[:, :, 2]*(256**2)).flatten()
    cnt = Counter(pixels_arr)

    pixel_value = cnt.most_common(2)[1][0]
    b, g, r = pixel_value%256, (pixel_value//256)%256, pixel_value//(256**2)

    return ('#%02x%02x%02x' % (r, g, b)).upper()


def polygon_region(image, bbox): 

    pt1 = (int(bbox[0][0]), int(bbox[0][1]))
    pt2 = (int(bbox[1][0]), int(bbox[1][1]))
    pt3 = (int(bbox[2][0]), int(bbox[2][1]))
    pt4 = (int(bbox[3][0]), int(bbox[3][1]))
    
    polygon = [pt1, pt2, pt3, pt4]


    mask = np.zeros((image.shape[0], image.shape[1]))
    if not type(polygon) == type(np.array([])):
        polygon = np.array(polygon)

    cv.fillConvexPoly(mask, polygon, 1)


    b_img = image[:,:,0] * mask 
    g_img = image[:,:,1] * mask 
    r_img = image[:,:,2] * mask 

    masked = np.zeros_like(image)
    masked[:,:,0] = b_img
    masked[:,:,1] = g_img
    masked[:,:,2] = r_img


    return masked, mask 


def rotate_image(image, angle, center):
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def get_rotate_angle(mask):
    blob = np.transpose(np.nonzero(mask)).astype(float)

    mean, eigenvectors, eigenvalues = cv.PCACompute2(blob, None)

    vectY = np.array(eigenvectors[0])
    vectX = np.array(eigenvectors[1])
    
    center = np.array([int(mean[0,1]), int(mean[0,0])], dtype=np.float64)

    basisX = np.array([1, 0], dtype=np.float32)
    #basisY = np.array([0, 1], dtype=np.float32)

    angle = np.arccos(np.dot(vectX, basisX))*180/np.pi

    if (vectX[1] < 0):
        return angle, center

    return -angle, center


def get_roi(image, bbox):
    pt1 = (int(bbox[0][0]), int(bbox[0][1]))
    pt2 = (int(bbox[1][0]), int(bbox[1][1]))
    pt3 = (int(bbox[2][0]), int(bbox[2][1]))
    pt4 = (int(bbox[3][0]), int(bbox[3][1]))


    polygon = [pt1, pt2, pt3, pt4]
    copy = np.copy(image)

    img, mask = polygon_region(copy, polygon)

    rotate_angle, center = get_rotate_angle(mask)
    


    if (rotate_angle>=-1.5 and rotate_angle<=1.5):
        return image[pt1[1]:pt3[1], pt1[0]:pt3[0]]


    rotated_mask = rotate_image(mask, rotate_angle, center)
    rotated_img  = rotate_image(img, rotate_angle, center)

    

    region = np.transpose(np.nonzero(rotated_mask))
    top_left = region[0] #+ np.array([1, 1], dtype=np.int64)
    bottom_right = region[len(region)-1] #- np.array([1, 1], dtype=np.int64)

    result = rotated_img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    H,W = result.shape[:2]

    if H > W: 
        return cv.rotate(result, cv.ROTATE_90_COUNTERCLOCKWISE)

    return result


def draw_text(img, text,
          pos=(0, 0),
          font=cv.FONT_HERSHEY_PLAIN,
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def patching_reshape(img):
    H, W = img.shape[:2]

    new_W = W//2 
    new_H = H*2

    half1 = img[:H, :W//2, :]
    half2 = img[:H, W//2:, :]

    new_img = np.zeros((new_H, W//2, 3), dtype=np.uint8)
    new_img[:new_H//2, :new_W, :] = half1
    new_img[new_H//2:, :new_W, :] = half2

    return new_img  

def dup_reshape(img): 
    H, W = img.shape[:2] 
    ratio = int(W/H)
    pass 

def iou(box1, box2): 
    # (x1, y1), (w1, h1) = box1
    # (x2, y2), (w2, h2) = box2

    # x_left = min(x1, x2)
    # x_right = max(x1+w1, x2+w2) 
    # y_top = min(y1, y2) 
    # y_bot = max(y1+w1, y2+w2) 

    # if (x_right < x_left or y_right < y_left): 
    pass 