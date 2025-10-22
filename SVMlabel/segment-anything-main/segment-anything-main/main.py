__author__ = 'Eric'

#导包
import cv2,matplotlib
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt
import numpy as np
import random

#cat = cv2.imread('cat.jpg')
cat = cv2.imread(r"D:\2025College Student Innovation and Entrepreneurship Project\project\WorkerSaftyDetector\datasets\my_dataset\images\train\image003.png")
cat = cv2.cvtColor(cat,cv2.COLOR_BGR2RGB)
#print(cat.shape)

device = "cuda"#CUDA运行



sam = sam_model_registry["vit_h"](checkpoint="../../sam_vit_h_4b8939.pth")#指定模型位置和模型名称

sam.to(device=device)#将模型导到CUDA中

#1、
predictor = SamPredictor(sam)

predictor.set_image(cat)#只需要使用一次


# 定义提示
# 点提示：格式为[x, y]和标签（1=正样本，0=负样本）
input_points = np.array([[86, 76], [154, 63]])  # 两个正样本点
input_labels = np.array([1, 1])  # 都是正样本

'''
# 框提示：格式为[x1, y1, x2, y2]
input_box = np.array([100, 100, 600, 500])  # [左上x, 左上y, 右下x, 右下y]
'''

# 进行预测（可以单独使用点或框，也可以组合使用）
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
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
fig, axes = plt.subplots(nrows=1, ncols=len(masks)+2, figsize=(20, 16))

axes[0].imshow(cat)
axes[0].set_title('ori image')

segAllImgMask = np.ones((cat.shape[0],cat.shape[1],3))

i=0

final_seg = np.zeros((cat.shape[0], cat.shape[1], 3))

for seg in masks:
    rand = random.randint(0,255)
    rand1 = random.randint(0,255)
    rand2 = random.randint(0,255)
    #rand = np.random.randn(0,255)
    #print(rand)
    #print(seg['crop_box'])
    #print(seg['segmentation'])

    # 将掩码中的0部分设置为0，保留掩码中的1部分
    seg_mask = seg[:, :, np.newaxis] # 形状变为 (797,797,1)
    seg_mask = np.repeat(seg_mask, 3, axis=2)  # 在通道维度重复3次
    #segimg = cat*seg_mask
    segimg = seg_mask*[rand,rand1,rand2]
    segimg[segimg == 0] = 255

    axes[i+1].imshow(segimg)
    axes[i+1].set_title('seg image{}'.format(i))

    #final_seg +=segimg#叠加结果
    #final_seg = final_seg/255
    axes[len(masks)+1].imshow(segimg,alpha = 0.3)#叠加并淡化

    i+=1
axes[len(masks)+1].set_title('final_seg')
#final_seg = final_seg/255
#axes[i+1].imshow(final_seg)
#axes[i+1].set_title('final_seg')



plt.tight_layout()

plt.savefig('result.png')

plt.show()



'''
2、
mask_generator = SamAutomaticMaskGenerator(sam)#导入到mask创建器中

masks = mask_generator.generate(cat)#根据输入图像创建mask

# 创建子图网格
fig, axes = plt.subplots(nrows=1, ncols=len(masks)+1, figsize=(20, 16))

axes[0].imshow(cat)
axes[0].set_title('ori image')

segAllImgMask = np.ones((cat.shape[0],cat.shape[1],3))
'''

'''
i=0
for seg in masks:
    rand = random.randint(0,255)
    #rand = np.random.randn(0,255)
    #print(rand)
    #print(seg['crop_box'])
    #print(seg['segmentation'])

    # 将掩码中的0部分设置为0，保留掩码中的1部分
    seg_mask = seg['segmentation'][:, :, np.newaxis] # 形状变为 (797,797,1)
    segimg = cat*seg_mask

    axes[i+1].imshow(segimg)
    axes[i+1].set_title('seg image{}'.format(i))

    i+=1



plt.tight_layout()

plt.savefig('result.png')

plt.show()
'''


'''
#检查坐标位置方法
def checkImgPoints(imgPath,x_point,y_point):
    img =cv2.imread(imgPath)
    x,y,_ = img.shape
    print(x,y)

checkImgPoints('./ico/last.png',1,1)
'''

