import cv2 as cv 
import numpy as np
from utils import getTime, is_overlap, draw_text, fw_preprocess
import time 
from  paddleocr  import  PaddleOCR
from tqdm import tqdm
import torch 
import os 

from fwc.inference import load_model, ensemble_predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
green = (0, 255, 0)
black = (0, 0, 0)


if __name__ == "__main__":
    

    vid = "vid7"

    # Output files path 
    frame_information = f"system_output/{vid}/"
    os.mkdir(frame_information)

    text_information = f"results/{vid}.txt"

    cap = cv.VideoCapture(f"./video/{vid}.mp4")
    fps = cap.get(cv.CAP_PROP_FPS) 
    totalNoFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames // fps
    Wo  = int(cap.get(3))
    Ho = int(cap.get(4)) 



    text_tracking = {}
    delay = 10 #delay between frames

    output_video = cv.VideoWriter( f"./output_videos/output_{vid}.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (Wo, Ho)) 


    st = time.time()
    noFrame = 0


    # Loading ultility models
    print("Loading models")
    ocr  =  PaddleOCR (lang = 'en', show_log = False)
    fwc = load_model("fwc/model/r.ckpt", n_classes= 4, device=device)
    print("Models loaded")

    progress_bar = tqdm(range(int(totalNoFrames)))
    while(cap.isOpened()): 
        ret, frame = cap.read() 
        if (noFrame >= totalNoFrames):
            break
        if ret == True:   
        #if (noFrame %30 == 0): 

            results = ocr.ocr(frame ,cls = False)
            result = results[0]
            if result is None: 
                frame = cv.resize(frame, (Wo, Ho))
                output_video.write(frame)
                noFrame +=1 
                progress_bar.update(1)
                continue 

            BoundingBoxes = [box[0] for box in result]
            Texts = [text[1][0] for text in result]
            Score = [score[1][1] for score in result]
            n_box = 0
            for box, text, score in zip(BoundingBoxes, Texts, Score): 
                box = np.reshape(np.array(box), (-1, 1, 2)).astype(np.int64)
                top_left = tuple(box[0][0])
                bottom_right = tuple(box[2][0])
                w = bottom_right[0] - top_left[0] 
                h = bottom_right[1] - top_left[1]
                bbox = (top_left[0], top_left[1], w, h)
                if score > 0.95:
                    offset = 10
                    roi = frame[top_left[1]-offset: bottom_right[1]+offset, top_left[0]-offset:bottom_right[0]+offset]

                    try:
                        input_img = fw_preprocess(roi)
                        fw, probs = ensemble_predict(model=fwc, img=input_img)
                        fscore = max(probs)
                        img_name = f"{str(text)}_{probs[0]}_{probs[1]}_{probs[2]}_{probs[3]}_{fw}.jpg"
                        img_path = frame_information + img_name
                        cv.imwrite(img_path, input_img)
                        
                    except: 
                        img_path = frame_information + "Frame_" + str(noFrame) + "_box_" + str(n_box) + "_None.jpg"
                        fw = "None"
                        fscore = "None"
                        #cv.imwrite(img_path, roi)

                

                    found = False 

                    for tracked_text, info in text_tracking.items(): 
                        tracked_box, _, start_frame, _, _ = info 

                        if tracked_text == text and is_overlap(tracked_box, bbox): 
                            text_tracking[tracked_text] = (bbox, fw, start_frame, noFrame, fscore)
                            found = True 

                    if not found: 
                        text_tracking[text] = (bbox, fw, noFrame, noFrame, fscore)

                    cv.polylines(frame, [box], isClosed=True, color=green, thickness=2)
                    draw_text(frame, text, pos= (top_left[0], top_left[1] - 15), font = cv.FONT_HERSHEY_SIMPLEX, font_scale = 1, font_thickness= 2, text_color= green, text_color_bg= black)
                n_box+=1
            frame = cv.resize(frame, (Wo, Ho))
            output_video.write(frame)

        noFrame+=1 
        progress_bar.update(1)
    
    with open(text_information, "w") as f: 
        pass 

    with open(text_information, "w") as f: 
        f.writelines("-------Result-------\n")
        for text, info in text_tracking.items(): 
            _, fw, start_frame, end_frame, fscore = info 
            start_ts = getTime(noFrame= start_frame, totalNoFrames= totalNoFrames, durationInSeconds= durationInSeconds)
            end_ts = getTime(noFrame= end_frame, totalNoFrames= totalNoFrames, durationInSeconds= durationInSeconds)    
            f.writelines(f"Text: {text}\n") 
            f.writelines(f"Font-weight: {fw}, Score: {fscore}\n")
            f.writelines(f"Start timestamp: {start_ts} s\n")
            f.writelines(f"End timestamp: {end_ts} s\n")
            f.writelines("--------------\n")



    cap.release() 
    output_video.release()
    cv.destroyAllWindows() 















































