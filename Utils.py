import cv2
import os
import torch
import numpy as np
from collections import Counter
import pickle
from torch.autograd import Variable

class Buffers:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory =[]
        self.position=0
    def push(self, num):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = num
        self.position = (self.position + 1) % self.capacity
    def result(self):
        res = [(torch.argmax(i)).item() for i in self.memory]
        c = Counter(res)
        if c.most_common(1)[0][0] ==0:
            return "Next"
        elif c.most_common(1)[0][0] ==1:
            return "Prev"
        elif c.most_common(1)[0][0] ==2:
            return "Stop"
        else:
            return "Background : Make a gesture"

def process(img):
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_mask = cv2.inRange(img_hls, np.array([0, 40, 20]), np.array([20, 200, 500]))
    img_blur = cv2.GaussianBlur(img_mask, (15, 15), 0)
    _, img_res = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_res = cv2.resize(img_res,(50,50))
    return img_res

def getVal(device):
    picklefiles = [n for n in os.listdir('./dataset/') if n[-4:]=='.pkl']
    if len(picklefiles)==0:
        data_gen()
    file = open('./dataset/dataV.pkl', 'rb')
    dataV= pickle.load(file)
    file = open('./dataset/labelV.pkl', 'rb')
    labelV= pickle.load(file)
    input_size_val = len(dataV)

    n,h,w = dataV.shape
    dataV = dataV.reshape(n,1,h,w)
    dataV = torch.from_numpy(dataV)
    dataV = dataV.type(torch.FloatTensor)
    labelV = labelV.astype(int)
    labelV = torch.from_numpy(labelV)
    dataV , labelV = dataV.to(device), labelV.to(device)
    dataV, labelV = Variable(dataV), Variable(labelV)
    return dataV, labelV, input_size_val

def data_gen():
    videofiles = [n for n in os.listdir('./') if n[-4:]=='.avi']
    videofiles.sort()
    video_index = 0
    cap = cv2.VideoCapture(videofiles[0])
    dataT =[]
    dataV =[]
    labelT=[]
    labelV=[]

    count =0
    while cap.isOpened():
        ret , frame = cap.read()
        if frame is None:
            count =0
            print("end of video " + str(video_index) + " .. next one now")
            video_index +=1
            if video_index>=len(videofiles):
                break
            cap = cv2.VideoCapture(videofiles[video_index])
            ret, frame = cap.read()
        frame = process(frame)
        if(count<3000):
            dataT.append(frame)
            labelT.append(video_index)
            count+=1
        else:
            dataV.append(frame)
            labelV.append(video_index)
        
    dataT = np.array(dataT)
    dataV = np.array(dataV)
    labelT= np.array(labelT)
    labelV= np.array(labelV)
    idx1 = np.random.permutation(len(dataT))
    idx2 = np.random.permutation(len(dataV))
    dataT, labelT = dataT[idx1], labelT[idx1]
    dataV, labelV = dataV[idx2], labelV[idx2]

    size_per_file = len(dataT)//5
    for i in range(0, 5):
        with open('./dataset/data'+str(i)+'.pkl','wb') as f:
            lis = dataT[i*size_per_file:(i+1)*size_per_file]
            pickle.dump(lis, f)
        f.close()
        with open('./dataset/label'+str(i)+'.pkl','wb') as f:
            lis= labelT[i*size_per_file:(i+1)*size_per_file]
            pickle.dump(lis, f)    

    with open('./dataset/dataV.pkl','wb') as f:
        pickle.dump(dataV, f)
    with open('./dataset/labelV.pkl','wb') as f:
        pickle.dump(labelV, f)    


