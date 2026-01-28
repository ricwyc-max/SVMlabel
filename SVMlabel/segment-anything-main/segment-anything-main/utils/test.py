# -*- coding: utf-8 -*-
"""
SAM 半自动标注器（支持滚轮缩放）
prompt：手画绿框 → SAM 生成蓝框 → Y 确认 → 输入类别 → 记录红框
A/D 切换图（立即保存）；N 撤销当前图（立即保存）；Q 退出
滚轮：向上放大，向下缩小，以鼠标位置为中心
输出：YOLO 格式 txt + classes.txt
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
import torch
import sys
import os

# ========= 用户配置 =========
IMG_DIR   = r'D:\2025College Student Innovation and Entrepreneurship Project\images\val'#文件夹位置
IMG_SUFFIX = ['.jpg', '.jpeg', '.png', '.bmp']#图片类型
SAM_CKPT  = r'../../../sam_vit_h_4b8939.pth'#权重文件位置
SAM_MODEL = 'vit_h'#模式（跟随权重）
# ============================

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
sys.path.append(str(project_root))
from segment_anything import sam_model_registry, SamPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = sam_model_registry[SAM_MODEL](checkpoint=SAM_CKPT)
sam.to(device=device)
predictor = SamPredictor(sam)

img_paths = []
curr_idx  = 0
img       = None          # 原始图像（始终不变）
img_disp_base = None      # 带已确认标注的基础图层（原始坐标）
sam_box   = None          # SAM 结果（原始图像坐标）
draw_box  = []            # 手画 prompt（原始图像坐标）[x1,y1,x2,y2]
mouse_down = False
cls_map   = {}            # name -> id

# 缩放相关全局变量
view_scale = 1.0          # 当前缩放比例
view_topleft = [0, 0]     # 视口左上角在原始图像中的坐标 [x, y]
min_scale = 0.1           # 最小缩放
max_scale = 10.0          # 最大缩放
zoom_factor = 1.2         # 每次缩放因子

cls_path = Path(IMG_DIR) / 'classes.txt'
if cls_path.exists():
    with open(cls_path, encoding='utf-8') as f:
        cls_map = {name.strip(): idx for idx, name in enumerate(f) if name.strip()}

# ----------- 工具 -----------
def load_image_list():
    global img_paths
    img_paths = sorted([p for p in Path(IMG_DIR).iterdir()
                        if p.suffix.lower() in {s.lower() for s in IMG_SUFFIX}])
    if not img_paths:
        raise RuntimeError('文件夹里没有符合条件的图片')
    print(f'共 {len(img_paths)} 张图')

# ---------- 读 ----------
def load_or_init_labels():
    labels = {}
    for p in img_paths:
        txt = p.with_suffix('.txt')
        if txt.exists():
            parts_list = []
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        parts_list.append([float(v) for v in parts])
            labels[p.stem] = parts_list
        else:
            labels[p.stem] = []
    return labels

# ---------- 写 ----------
def save_one_label(labels, p):
    txt = p.with_suffix('.txt')
    with open(txt, 'w') as f:
        for parts in labels[p.stem]:
            # 修改：确保第一列是整数格式，避免保存为 0.0
            f.write(f"{int(parts[0])} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")

# ---------- 生成 ----------
def yolo_line(cls_id, xyxy, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    x_c = ((x1 + x2) * 0.5) / img_w
    y_c = ((y1 + y2) * 0.5) / img_h
    w   = abs(x2 - x1) / img_w
    h   = abs(y2 - y1) / img_h
    return [int(cls_id), x_c, y_c, w, h]

def ask_class_name():
    root = tk.Tk()
    root.withdraw()
    cls = simpledialog.askstring('类别', '请输入英文类别名：')
    root.destroy()
    return cls.strip() if cls else None

def save_classes():
    if cls_map:
        id_name = sorted(cls_map.items(), key=lambda x: x[1])
        with open(Path(IMG_DIR) / 'classes.txt', 'w', encoding='utf-8') as f:
            for name, _id in id_name:
                f.write(f'{name}\n')

def id2name(cls_id: int) -> str:
    return next((k for k, v in cls_map.items() if v == cls_id), f'class_{cls_id}')

# ---------- 缩放/视口 工具 ----------
def screen_to_img(x_screen, y_screen):
    """将屏幕坐标转换为原始图像坐标"""
    x_img = x_screen / view_scale + view_topleft[0]
    y_img = y_screen / view_scale + view_topleft[1]
    return x_img, y_img

def img_to_screen(x_img, y_img):
    """将原始图像坐标转换为屏幕坐标"""
    x_screen = (x_img - view_topleft[0]) * view_scale
    y_screen = (y_img - view_topleft[1]) * view_scale
    return int(x_screen), int(y_screen)

def get_viewport_display(base_img, win_width, win_height):
    """
    根据 view_scale 和 view_topleft 从 base_img 生成视口图像
    base_img: 包含已确认标注的基础图像（原始分辨率）
    """
    h, w = base_img.shape[:2]
    
    # 计算视口在原始图像中的尺寸
    view_w = int(win_width / view_scale)
    view_h = int(win_height / view_scale)
    
    # 视口范围
    x1 = int(view_topleft[0])
    y1 = int(view_topleft[1])
    x2 = min(x1 + view_w, w)
    y2 = min(y1 + view_h, h)
    
    # 边界检查（防止完全越界）
    if x1 >= w: x1, x2 = w-1, w
    if y1 >= h: y1, y2 = h-1, h
    if x2 <= 0: x1, x2 = 0, 1
    if y2 <= 0: y1, y2 = 0, 1
    
    # 确保坐标有效
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1+1, x2)), min(h, max(y1+1, y2))
    
    # 裁剪ROI
    roi = base_img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return np.zeros((win_height, win_width, 3), dtype=np.uint8)
    
    # 缩放到窗口大小
    display = cv2.resize(roi, (win_width, win_height), interpolation=cv2.INTER_LINEAR)
    return display

def clamp_viewport(img_w, img_h, win_w, win_h):
    """限制视口不超出图像边界太多"""
    global view_topleft
    max_x = max(0, img_w - int(win_w / view_scale))
    max_y = max(0, img_h - int(win_h / view_scale))
    view_topleft[0] = max(0, min(view_topleft[0], max_x))
    view_topleft[1] = max(0, min(view_topleft[1], max_y))

# ----------- 鼠标 -----------
def mouse_callback(event, x, y, flags, param):
    global draw_box, mouse_down, sam_box, view_scale, view_topleft
    
    # 滚轮缩放事件 (OpenCV 10 是 MOUSEWHEEL)
    if event == 10 or event == cv2.EVENT_MOUSEWHEEL:
        # 获取滚轮方向 (flags > 0 向上/放大，< 0 向下/缩小)
        zoom_in = flags > 0
        
        # 鼠标在原始图像中的位置（缩放前）
        orig_x, orig_y = screen_to_img(x, y)
        
        # 更新缩放比例
        old_scale = view_scale
        if zoom_in:
            view_scale = min(view_scale * zoom_factor, max_scale)
        else:
            view_scale = max(view_scale / zoom_factor, min_scale)
        
        # 调整视口位置，使鼠标指向的图像点保持在屏幕相同位置
        # 新视口左上角 = 鼠标指向的图像点 - 鼠标屏幕位置 / 新缩放比例
        new_topleft_x = orig_x - x / view_scale
        new_topleft_y = orig_y - y / view_scale
        view_topleft = [new_topleft_x, new_topleft_y]
        
        return
    
    # 将屏幕坐标转换为原始图像坐标进行存储
    x_img, y_img = screen_to_img(x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
        draw_box = [x_img, y_img, x_img, y_img]
        sam_box = None
    elif event == cv2.EVENT_MOUSEMOVE and mouse_down:
        draw_box[2:] = [x_img, y_img]
        # 注意：在 process_one 的循环中会处理显示，这里不直接 imshow
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        draw_box[2:] = [x_img, y_img]
        print(f'prompt 框(原始坐标)：({draw_box[0]:.1f}, {draw_box[1]:.1f}) -> ({draw_box[2]:.1f}, {draw_box[3]:.1f})')
        
        # SAM 预测使用原始图像坐标
        if len(draw_box) == 4:
            # 确保坐标顺序正确（左上角，右下角）
            x1, y1, x2, y2 = draw_box
            input_box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
            
            try:
                masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
                if masks is not None and len(masks) > 0:
                    y_idx, x_idx = np.where(masks[0] > 0)
                    if x_idx.size and y_idx.size:
                        sam_box = [float(x_idx.min()), float(y_idx.min()), 
                                  float(x_idx.max()), float(y_idx.max())]
                        print(f'SAM 优化框(原始坐标)：({sam_box[0]:.1f}, {sam_box[1]:.1f}) -> ({sam_box[2]:.1f}, {sam_box[3]:.1f})')
                    else:
                        sam_box = None
                        print('SAM 未生成有效 mask')
            except Exception as e:
                print(f'SAM 预测出错: {e}')
                sam_box = None

# ----------- 单张图主循环 -----------
def process_one(labels):
    global img, img_disp_base, sam_box, draw_box, view_scale, view_topleft, cls_map
    
    path = img_paths[curr_idx]
    print('=== ', path.name, f'  {curr_idx+1}/{len(img_paths)}  ===')
    
    img = cv2.imread(str(path))
    if img is None:
        print('读图失败，跳过')
        return 1
    
    h, w = img.shape[:2]
    
    # 重置视口
    view_scale = 1.0
    view_topleft = [0, 0]
    
    # 准备基础图层（原图 + 已确认的标注）
    img_disp_base = img.copy()
    for line in labels[path.stem]:
        cls_id, xc, yc, bw, bh = line
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        cv2.rectangle(img_disp_base, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cls_name = id2name(int(cls_id))
        cv2.putText(img_disp_base, cls_name, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # 为SAM设置图像（使用原始图像）
    predictor.set_image(img)
    
    sam_box = None
    draw_box = []
    
    # 获取窗口初始大小（默认使用图像大小）
    win_w, win_h = w, h
    
    while True:
        # 尝试获取实际窗口大小
        try:
            rect = cv2.getWindowImageRect('annotator')
            if rect[2] > 0 and rect[3] > 0:  # 宽 > 0, 高 > 0
                win_w, win_h = rect[2], rect[3]
        except:
            pass
        
        # 限制视口范围
        clamp_viewport(w, h, win_w, win_h)
        
        # 生成视口图像
        display = get_viewport_display(img_disp_base, win_w, win_h)
        
        # 转换坐标并绘制当前框（绿prompt框）
        if len(draw_box) == 4:
            x1_s, y1_s = img_to_screen(draw_box[0], draw_box[1])
            x2_s, y2_s = img_to_screen(draw_box[2], draw_box[3])
            # 确保在显示范围内
            if all(0 <= v < max(win_w, win_h) * 2 for v in [x1_s, y1_s, x2_s, y2_s]):
                cv2.rectangle(display, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
        
        # 转换坐标并绘制SAM框（蓝框）
        if sam_box is not None:
            x1_s, y1_s = img_to_screen(sam_box[0], sam_box[1])
            x2_s, y2_s = img_to_screen(sam_box[2], sam_box[3])
            cv2.rectangle(display, (x1_s, y1_s), (x2_s, y2_s), (255, 0, 0), 2)
        
        # 显示缩放比例提示
        info_text = f'Zoom: {view_scale:.2f}x | Use Mouse Wheel'
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('annotator', display)
        key = cv2.waitKey(30) & 0xFF  # 稍微增加延迟以减轻CPU负担

        if key == ord('y'):
            # 使用原始坐标保存
            box = sam_box if sam_box is not None else (draw_box if len(draw_box) == 4 else None)
            if box is None:
                print('无有效框，跳过')
                continue
            
            cls = ask_class_name()
            if cls is None:
                continue
            
            if cls not in cls_map:
                cls_map[cls] = len(cls_map)
                save_classes()
            
            cls_id = cls_map[cls]
            # 确保是标准格式 [x1, y1, x2, y2]
            box_std = [min(box[0], box[2]), min(box[1], box[3]), 
                      max(box[0], box[2]), max(box[1], box[3])]
            line = yolo_line(cls_id, box_std, w, h)
            labels[path.stem].append(line)
            
            # 更新基础图层（在原始坐标上画红框）
            x1, y1, x2, y2 = map(int, box_std)
            cv2.rectangle(img_disp_base, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_disp_base, cls, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            sam_box = None
            draw_box = []
            print('已记录：', cls, box_std)

        elif key == ord('n'):
            if labels[path.stem]:
                removed = labels[path.stem].pop()
                print('撤销：', removed)
                save_one_label(labels, path)
                save_classes()
                
                # 重绘基础图层
                img_disp_base = img.copy()
                for line in labels[path.stem]:
                    cls_id, xc, yc, bw, bh = line
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    cv2.rectangle(img_disp_base, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cls_name = id2name(int(cls_id))
                    cv2.putText(img_disp_base, cls_name, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                print('当前图片无标注可撤销')

        elif key == ord('a'):
            save_one_label(labels, path)
            save_classes()
            return -1
            
        elif key == ord('d'):
            save_one_label(labels, path)
            save_classes()
            return 1
            
        elif key == ord('q') or key == 27:
            save_one_label(labels, path)
            save_classes()
            return 0
        
        elif key == ord('r'):
            # 新增：R键重置视图
            view_scale = 1.0
            view_topleft = [0, 0]
            print('视图已重置')

# ----------- 主入口 -----------
def main():
    global curr_idx, cls_map
    load_image_list()
    labels = load_or_init_labels()
    cv2.namedWindow('annotator', cv2.WINDOW_NORMAL)  # 可调整大小
    cv2.setMouseCallback('annotator', mouse_callback)

    while 0 <= curr_idx < len(img_paths):
        code = process_one(labels)
        if code == 0:
            break
        curr_idx += code
        curr_idx = max(0, min(curr_idx, len(img_paths)-1))

    # 最后再整体保存一次（含 classes.txt）
    for p in img_paths:
        save_one_label(labels, p)
    if cls_map:
        id_name = sorted(cls_map.items(), key=lambda x: x[1])
        with open(Path(IMG_DIR) / 'classes.txt', 'w', encoding='utf-8') as f:
            for name, _id in id_name:
                f.write(f'{name}\n')
        print('classes.txt 已保存：', [n for n, i in id_name])

    cv2.destroyAllWindows()
    print('全部完成！')

if __name__ == '__main__':
    main()