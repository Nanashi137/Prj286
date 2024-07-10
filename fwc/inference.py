import cv2 as cv 
import numpy as np
from collections import OrderedDict

from .lightning_wrappers import DeepFontWrapper
from .deepfont import DeepFont, DeepFontAutoencoder
import torch 
import torch.nn.functional as F 
from .transformations import IPtrans, inference_input
import time

LABELS2ID = {'unbold': 0, 'bold': 1, 'italic': 2, 'bold_italic': 3}
ID2LABELS = {0: 'unbold', 1: 'bold', 2: 'italic', 3: 'bold_italic'}

def predict(model: DeepFontWrapper, img): 
    #g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY ) 
    img_input = IPtrans(img)
    img_input.unsqueeze_(0)
    logit = model.forward(img_input.cuda())
    pred = F.softmax(logit, dim=1)

    return ID2LABELS[pred.argmax().item()], torch.max(pred).item()

def load_model(checkpoint_path, n_classes, device): 
    df = DeepFont(autoencoder= DeepFontAutoencoder(),  num_classes= n_classes)
    checkpoint = torch.load(checkpoint_path)
    

    df_state_dict = OrderedDict()

    for k, v in checkpoint['state_dict'].items():
        name = k[6:] 
        df_state_dict[name]=v

    df.load_state_dict(df_state_dict)
    return DeepFontWrapper(model= df, num_classes= n_classes).to(device)


def ensemble_predict(model, img):
    all_soft_preds = []
    
    for _ in range(2):
        patches = [inference_input(img, squeezing_ratio=np.random.uniform(low=1.5, high=3.5)) for _ in range(5)]
        # return patches
        inputs = torch.tensor(np.asarray(patches))

        preds = model(inputs.cuda())
        soft_preds = F.softmax(preds, dim=1)
        all_soft_preds.append(soft_preds)

    probs = torch.cat(all_soft_preds).mean(0)
    return ID2LABELS[probs.argmax().item()], probs.tolist()




# Debug 
if __name__=="__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = cv.imread("v5.png", cv.IMREAD_GRAYSCALE)
    img1 = cv.GaussianBlur(img1, (3,3),0)
    #Loading model  

    model = load_model("model/r.ckpt", n_classes=4, device=device)

    infer_b = time.time()
    pred1, score = predict(model, img=img1)
    print(f"Inference time: {time.time() - infer_b}")
    cv.imshow(pred1, img1)
    print(score)
    cv.waitKey(0)
    