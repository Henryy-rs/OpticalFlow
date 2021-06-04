import cv2 as cv
import numpy as np
import torch
import os


class DenseFlow:
    def __init__(self, video_path, frame_path):
        self.video_path = video_path
        self.frame_path = frame_path
        #self.video_name = self.video_path.split("/")[-1].split(".")[0]

    def dense_optical_flow(self):
        cap = cv.VideoCapture(self.video_path)
        length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        count = cv.cuda.getCudaEnabledDeviceCount()
        print("opencv gpu: ", count)
        print("torch gpu:", torch.cuda.is_available())
        cuda = torch.device('cuda')

        ret, first_frame = cap.read()

        if ret is False:
            print("ERROR: video loading failed")
            return

        first_frame_gpu = cv.cuda_GpuMat()
        # upload on gpu
        first_frame_gpu.upload(first_frame)

        # use gpu
        prev_gray = cv.cuda.cvtColor(first_frame_gpu, cv.COLOR_BGR2GRAY)
        num_frame = 0

        while(1):
            print(str(round((num_frame/length)*100, 2))+"%")
            ret, frame = cap.read()
            if frame is None:
                break
            frame_gpu = cv.cuda_GpuMat()
            frame_gpu.upload(frame)

            # use gpu
            gray = cv.cuda.cvtColor(frame_gpu, cv.COLOR_BGR2GRAY)

            gpu_flow = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0,)
            # Calculates dense optical flow by Farneback method
            # use gpu
            gpu_flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow, prev_gray, gray, None,)
            # 아래 과정에서, GPU-> CPU-> GPU -> CPU로 텐서가 이동한다. GPU에서 바로 넘길 수 있으면 좋겠으나 방법을 찾지 못했다.
            flow = gpu_flow.download()
            torch_flow = torch.tensor(flow, device=cuda)

            torch_flow = torch_flow/24+0.5  #10% of height

            # # 0, 1사이로 고정
            torch_flow = torch.clamp(torch_flow, 0, 1)

            #padding
            m = torch.nn.ZeroPad2d((0, 1, 0, 0))
            torch_flow = m(torch_flow)
            torch_flow = torch_flow.cpu()

            npy_flow = torch_flow.numpy()
            print(npy_flow[100][100])
            optical_path = self.frame_path+"/flow_"+str(num_frame)+".npy"
            np.save(optical_path, npy_flow)

            prev_gray = gray
            num_frame += 1

        cap.release()

    def delete_flow(self):
        os.system('rmdir /s /q ' + self.frame_path)

    def change_video(self, video_path):
        self.video_path = video_path
        #self.video_name = self.video_path.split("/")[-1].split(".")[0]
