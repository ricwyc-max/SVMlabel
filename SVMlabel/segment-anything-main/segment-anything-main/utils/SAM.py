__author__ = 'Eric'
import sys
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 获取项目根目录（segment-anything-main文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))  # utils目录
project_root = os.path.dirname(current_dir)  # segment-anything-main目录

sys.path.append(project_root)

#导包
import cv2,matplotlib
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt
import numpy as np
import random
import torch




class SAM():
    def __init__(self,imgPath,input_points=None,input_labels = None,input_box =None,device = 'cpu'):

        self.imgPath = imgPath
        self.input_points = input_points
        self.input_labels = input_labels
        self.input_box = input_box

        self.device = device#CUDA运行(如果有的话)
        self.sam = sam_model_registry["vit_h"](checkpoint="../../sam_vit_h_4b8939.pth")#指定模型位置和模型名称

        self.sam.to(device=self.device)#将模型导到CUDA中

        self.img = cv2.imread(imgPath)
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(self.img)#只需要使用一次
        #print(img.shape)

        self.masks = []
        self.scores=[]
        self.logits=[]





    def getSAMView(self):
        #1、
        '''
        # 定义提示
        # 点提示：格式为[x, y]和标签（1=正样本，0=负样本）
        input_points = np.array([[500, 375], [300, 200]])  # 两个正样本点
        input_labels = np.array([1, 1])  # 都是正样本
        # 框提示：格式为[x1, y1, x2, y2]
        input_box = np.array([100, 100, 600, 500])  # [左上x, 左上y, 右下x, 右下y]
        '''

        # 进行预测（可以单独使用点或框，也可以组合使用）
        masks, scores, logits = self.predictor.predict(
            #point_coords=input_points,
            #point_labels=input_labels,
            #box=input_box[None, :],  # 增加一个batch维度
            #multimask_output=True,   # 输出多个可能的掩码(默认为True，输出3个)
        )

        '''
        masks: 你要的最终分割结果
        scores: 判断哪个mask更好的依据
        logits: 用于高级功能如迭代优化
        '''

        #print(masks,scores,logits)

        # 创建子图网格
        #fig, axes = plt.subplots(nrows=1, ncols=len(masks)+2, figsize=(20, 16))

        #axes[0].imshow(cat)
        #axes[0].set_title('ori image')

        #segAllImgMask = np.ones((self.img.shape[0],self.img.shape[1],3))

        #i=0

        #final_seg = np.zeros((self.img.shape[0], self.img.shape[1], 3))

        #创建图像
        figure = Figure(figsize=(5, 20), dpi=100)
        canvas = FigureCanvas(figure)
        axes = figure.add_axes([0,0,1,1])
        #撤销坐标轴
        axes.axis('off')

        for seg in masks:
            rand = random.randint(0,255)
            rand1 = random.randint(0,255)
            rand2 = random.randint(0,255)
            #rand = np.random.randn(0,255)
            #print(rand)
            #print(seg['crop_box'])
            #print(seg['segmentation'])

            # 将掩码中的0部分设置为0，保留掩码中的1部分
            seg_mask = seg[:, :, np.newaxis] # 加一个维度
            seg_mask = np.repeat(seg_mask, 3, axis=2)  # 在通道维度重复3次
            #segimg = cat*seg_mask
            segimg = seg_mask*[rand,rand1,rand2]
            segimg[segimg == 0] = 255

            #axes[i+1].imshow(segimg)
            #axes[i+1].set_title('seg image{}'.format(i))

            #final_seg +=segimg#叠加结果
            #final_seg = final_seg/255
            axes.imshow(segimg,alpha = 0.5)#叠加并淡化

            #i+=1

        #axes[len(masks)+1].set_title('final_seg')

        #final_seg = final_seg/255
        #axes[i+1].imshow(final_seg)
        #axes[i+1].set_title('final_seg')



        #plt.tight_layout()
        #plt.savefig('result.png')
        #plt.show()
        return canvas,axes

#if __name__ == '__main__':
    #SAM()