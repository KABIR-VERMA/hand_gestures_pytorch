import cv2
from Utils import process, Buffers
from brain import CNNModel
import torch
from torch.autograd import Variable

    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    model.load_state_dict(torch.load("./weightedmodel.pth"))
    L = Buffers(12)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if frame is not None:
            tframe = process(frame)
            pframe = tframe.copy()
            tframe = tframe.reshape(1,1,50,50)
            tframe = torch.from_numpy(tframe)
            tframe = tframe.type(torch.FloatTensor)
            tframe = (Variable(tframe)).to(device)
            out = model(tframe)
            L.push(out)
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, str(L.result()),(50,50), font, 1, (255,0,0), 2, cv2.LINE_8)
            cv2.imshow("HASIYE AAP CAMERE ME HAI!!", frame)
            cv2.imshow("cool ", pframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("happy smiling @'-'@")