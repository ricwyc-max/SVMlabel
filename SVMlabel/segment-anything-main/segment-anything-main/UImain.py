__author__ = 'Eric'
import sys,os
from PySide2.QtWidgets import *
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import cv2
from utils import SAM

# 设置 Matplotlib 使用 Qt5Agg 后端
matplotlib.use("Qt5Agg")

def read_files_with_extension(folder_path, extension):
    file_path_list = []
    # 遍历文件夹中的所有文件
    if folder_path:
        for filename in os.listdir(folder_path):
            # 检查文件后缀名
            if filename.endswith(extension):
                file_path = os.path.join(folder_path, filename)
                # 打开并读取文件内容
                #print(file_path)
                file_path_list.append(file_path)

    return file_path_list

#创建图像方法基于matplotlib
def createMatImg(imgPath):
    img =cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # 创建一个 Matplotlib 图表
    figure = Figure(figsize=(5, 20), dpi=100)
    canvas = FigureCanvas(figure)
    axes = figure.add_axes([0,0,1,1])
    axes.axis('off')
    axes.imshow(img)


    return canvas,axes

#检查坐标位置方法
def checkImgPoints(imgPath,x_point,y_point):
    img =cv2.imread(imgPath)
    x,y,_ = img.shape
    #print(x,y)
    if x_point > x or x_point < 0 or y_point > y or y_point < 0:
        return False
    else:
        return True
    return False



#mainWindow界面
class MainWindow():
    def __init__(self):
        #从文件中加载UI定义
        qfile_mainWindow = QFile('./UI/UI.ui')
        qfile_mainWindow.open(QFile.ReadOnly)
        #从UI定义中动态创建一个相应的窗口对象
        self.ui = QUiLoader().load(qfile_mainWindow)
        qfile_mainWindow.close()

        self.folder_path = ''
        self.checkBoxes=[]
        self.nowImageNum = 0
        self.canvas = None
        self.axes = None
        self.canvas1 = None
        self.axes1 = None
        self.label = []#标签列表
        self.file_path_list=[]#文件路径列表

        #手动设置image栏和label栏的界面布局
        self.ui.ImageLayout.setAlignment(Qt.AlignTop)
        self.ui.label_layout.setAlignment(Qt.AlignTop)

        #绑定信号到槽
        self.ui.openDoc.triggered.connect(self.openDocButton_click)#工具栏openDoc（打开文件）工具被点击时触发
        self.ui.lastFile.triggered.connect(self.lastFileButton_click)#工具栏lastFile（上一文件）工具被点击时触发
        self.ui.nextFile.triggered.connect(self.nextFileButton_click)#工具栏openDoc（下一文件）工具被点击时触发


    def remove_all_widgets(self,layout):
            # 从布局中移除所有控件
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()  # 完全删除控件

    #展示窗口方法
    def show(self):
        self.ui.show()#展示窗口

    #打开文件工具调用时候触发
    def openDocButton_click(self):
        #print("按钮被点击")
        # 调出文件管理器，让用户选择文件夹
        self.folder_path = QFileDialog.getExistingDirectory(self.ui, "选择文件夹",'')
        '''
        if self.folder_path:
            print(f"选择的文件夹：{self.folder_path}")
        '''

        extension = ".png"  # 替换为你需要的文件后缀名
        self.file_path_list = read_files_with_extension(self.folder_path, extension)


        #循环创建checkbox在image栏上
        for i in range(len(self.file_path_list)):
            # 创建一个 QCheckBox
            self.checkBoxes.append( QCheckBox(self.file_path_list[i], self.ui) )
            #self.checkBoxes[i].setChecked(True)  # 设置初始状态为选中
            self.checkBoxes[i].setEnabled(False)#禁用checkBox，使其不能够手动打勾
            #将BOX加入到image栏中
            self.ui.ImageLayout.addWidget(self.checkBoxes[i])


        if self.file_path_list!=[]:
            #先删除视图内原有的图片
            self.remove_all_widgets(self.ui.oriViewLayout)
            self.remove_all_widgets(self.ui.SAMLayout)
            self.remove_all_widgets(self.ui.segViewLayout)
            self.remove_all_widgets(self.ui.promptViewLayout)
            #将第一张图片进行显示
            #获取左右的原始视图
            self.canvas,self.axes = createMatImg(self.file_path_list[self.nowImageNum])
            self.canvas1,self.axes1 = createMatImg(self.file_path_list[self.nowImageNum])
            #实例化SAM对象
            sam = SAM.SAM(self.file_path_list[self.nowImageNum])
            self.canvasSAM,self.axesSAM = sam.getSAMView()#获取SAM视图


            # 连接鼠标滚轮事件
            self.canvas.mpl_connect('scroll_event', lambda event: self.on_mouse_wheel(event,self.axes))
            self.canvas1.mpl_connect('scroll_event',lambda event:  self.on_mouse_wheel(event,self.axes1))
            self.canvasSAM.mpl_connect('scroll_event',lambda event:  self.on_mouse_wheel(event,self.axesSAM))

        if self.canvas:
            # 将图表添加到布局中
            #1、左侧原图
            self.ui.leftAreaImg.setWidget(self.canvas)
            #2、右侧，roiView视图
            # 创建一个 oriViewLayout 并将其设置为 self.ui.oriView 的布局
            self.ui.oriViewLayout.addWidget(self.canvas1)
            #3、右侧，SAM视图
            # 创建一个 SAMLayout 并将其设置为 self.ui.SAM 的布局
            self.ui.SAMLayout.addWidget(self.canvasSAM)


            # 连接鼠标移动事件
            self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
            self.canvas1.mpl_connect("motion_notify_event", self.on_mouse_move1)
            self.canvasSAM.mpl_connect("motion_notify_event", self.on_mouse_moveSAM)


    #显示鼠标在图表中的位置（左边移动的时候）
    def on_mouse_move(self, event):
        if event.inaxes == self.axes:
            x, y = event.xdata, event.ydata
            print(f"鼠标位置: x={x:.2f}, y={y:.2f}")
            in_img=checkImgPoints(self.file_path_list[self.nowImageNum],x,y)
            print(in_img)

    #显示鼠标在图表中的位置（右边roiView视图移动的时候）
    def on_mouse_move1(self, event):
        if event.inaxes == self.axes1:
            x, y = event.xdata, event.ydata
            print(f"鼠标位置: x={x:.2f}, y={y:.2f}")
            in_img=checkImgPoints(self.file_path_list[self.nowImageNum],x,y)
            print(in_img)

    #显示鼠标在图表中的位置（右边SAM视图移动的时候）
    def on_mouse_moveSAM(self, event):
        if event.inaxes == self.axesSAM:
            x, y = event.xdata, event.ydata
            print(f"鼠标位置: x={x:.2f}, y={y:.2f}")
            in_img=checkImgPoints(self.file_path_list[self.nowImageNum],x,y)
            print(in_img)


    #上一文件工具调用时触发
    def lastFileButton_click(self):
        #判断是否为第一张图
        if self.nowImageNum>0:#不为第一张图
            self.nowImageNum = self.nowImageNum-1
            if self.file_path_list!=[]:
                #将相应图片进行显示
                #创建图片
                self.canvas,self.axes = createMatImg(self.file_path_list[self.nowImageNum])#左边图片
                self.canvas1,self.axes1 = createMatImg(self.file_path_list[self.nowImageNum])#右边roiView视图图片
                #实例化SAM对象
                sam = SAM.SAM(self.file_path_list[self.nowImageNum])
                self.canvasSAM,self.axesSAM = sam.getSAMView()#获取SAM视图

                # 连接鼠标滚轮事件
                self.canvas.mpl_connect('scroll_event', lambda event:  self.on_mouse_wheel(event,self.axes))
                self.canvas1.mpl_connect('scroll_event', lambda event:  self.on_mouse_wheel(event,self.axes1))
                self.canvasSAM.mpl_connect('scroll_event',lambda event:  self.on_mouse_wheel(event,self.axesSAM))
            if self.canvas:
                # 将图表添加到布局中
                #1、左侧原图
                self.ui.leftAreaImg.setWidget(self.canvas)
                #2、右侧，roiView视图
                # oriViewLayout为 self.ui.oriView 的布局
                #先删除视图内原有的图片
                self.remove_all_widgets(self.ui.oriViewLayout)
                self.remove_all_widgets(self.ui.SAMLayout)
                self.remove_all_widgets(self.ui.segViewLayout)
                self.remove_all_widgets(self.ui.promptViewLayout)

                #再添加
                self.ui.oriViewLayout.addWidget(self.canvas1)
                self.ui.SAMLayout.addWidget(self.canvasSAM)

                # 连接鼠标移动事件
                self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
                self.canvas1.mpl_connect("motion_notify_event", self.on_mouse_move1)
                self.canvasSAM.mpl_connect("motion_notify_event", self.on_mouse_moveSAM)
                #print(self.nowImageNum)
        else:#为第一张图
            pass

    #下一文件工具调用时候触发
    def nextFileButton_click(self):
        #判断是否为最后一张图
        if self.nowImageNum!=len(self.file_path_list)-1:#不为最后一张图
            self.nowImageNum = self.nowImageNum+1
            if self.file_path_list!=[]:
                #将相应图片进行显示
                #创建图片
                self.canvas,self.axes = createMatImg(self.file_path_list[self.nowImageNum])#左边图片
                self.canvas1,self.axes1 = createMatImg(self.file_path_list[self.nowImageNum])#右边roiView视图图片
                #实例化SAM对象
                sam = SAM.SAM(self.file_path_list[self.nowImageNum])
                self.canvasSAM,self.axesSAM = sam.getSAMView()#获取SAM视图

                # 连接鼠标滚轮事件
                self.canvas.mpl_connect('scroll_event', lambda event:  self.on_mouse_wheel(event,self.axes))
                self.canvas1.mpl_connect('scroll_event', lambda event:  self.on_mouse_wheel(event,self.axes1))
                self.canvasSAM.mpl_connect('scroll_event',lambda event:  self.on_mouse_wheel(event,self.axesSAM))
            if self.canvas:
                # 将图表添加到布局中
                #1、左侧原图
                self.ui.leftAreaImg.setWidget(self.canvas)
                #2、右侧，roiView视图
                # oriViewLayout为 self.ui.oriView 的布局
                #先删除视图内原有的图片
                self.remove_all_widgets(self.ui.oriViewLayout)
                self.remove_all_widgets(self.ui.SAMLayout)
                self.remove_all_widgets(self.ui.segViewLayout)
                self.remove_all_widgets(self.ui.promptViewLayout)

                #再添加
                self.ui.oriViewLayout.addWidget(self.canvas1)
                self.ui.SAMLayout.addWidget(self.canvasSAM)

                # 连接鼠标移动事件
                self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
                self.canvas1.mpl_connect("motion_notify_event", self.on_mouse_move1)
                self.canvasSAM.mpl_connect("motion_notify_event", self.on_mouse_moveSAM)
                #print(self.nowImageNum)
        else:#为最后一张图
            pass


    #滚轮同步缩放
    def on_mouse_wheel(self, event, axes_now):
        if event.inaxes == axes_now:
            # 获取当前鼠标位置
            x, y = event.xdata, event.ydata

            # 获取当前视图范围
            xlim = axes_now.get_xlim()
            ylim = axes_now.get_ylim()

            # 计算鼠标位置在视图中的相对位置
            x_rel = (x - xlim[0]) / (xlim[1] - xlim[0])
            y_rel = (y - ylim[0]) / (ylim[1] - ylim[0])

            # 计算新的视图范围
            if event.button == 'up':  # 放大
                scale_factor = 1.1
            elif event.button == 'down':  # 缩小
                scale_factor = 0.9

            new_xlim = [x - (x - xlim[0]) / scale_factor, x + (xlim[1] - x) / scale_factor]
            new_ylim = [y - (y - ylim[0]) / scale_factor, y + (ylim[1] - y) / scale_factor]

            # 调整视图范围
            self.axes.set_xlim(new_xlim)
            self.axes.set_ylim(new_ylim)
            self.axes1.set_xlim(new_xlim)
            self.axes1.set_ylim(new_ylim)
            self.axesSAM.set_xlim(new_xlim)
            self.axesSAM.set_ylim(new_ylim)

            # 重新绘制图像
            self.canvas1.draw()
            self.canvas.draw()
            self.canvasSAM.draw()













if __name__ == '__main__':


    app = QApplication([])  # 创建 QApplication 实例

    mainWindow = MainWindow()  # 实例化界面
    mainWindow.show()  # 展示界面
    sys.exit(app.exec_())  # 启动事件循环



