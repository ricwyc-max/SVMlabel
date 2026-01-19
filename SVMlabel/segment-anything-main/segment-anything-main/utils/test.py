# -*- coding: utf-8 -*-
"""
SAM 半自动标注器
prompt：手画绿框 → SAM 生成蓝框 → Y 确认 → 输入类别 → 记录红框
A/D 切换图（立即保存）；N 撤销当前图（立即保存）；Q 退出
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
IMG_DIR   = r'E:\1'#文件夹位置
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
predictor = SamPredictor(sam)

img_paths = []
curr_idx  = 0
img       = None
img_disp  = None
sam_box   = None          # SAM 结果
draw_box  = []            # 手画 prompt
mouse_down = False
cls_map   = {}            # name -> id

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
    # 2. 再读标注
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
            f.write(' '.join(str(v) for v in parts) + '\n')

# ---------- 生成 ----------
def yolo_line(cls_id, xyxy, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    x_c = ((x1 + x2) * 0.5) / img_w
    y_c = ((y1 + y2) * 0.5) / img_h
    w   = abs(x2 - x1) / img_w
    h   = abs(y2 - y1) / img_h
    return [float(cls_id), x_c, y_c, w, h]   # 统一 float

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
    """根据 ID 返回类别名称，找不到就生成占位符"""
    return next((k for k, v in cls_map.items() if v == cls_id), f'class_{cls_id}')

# ----------- 鼠标 -----------
def mouse_callback(event, x, y, flags, param):
    global draw_box, mouse_down, img_disp, sam_box
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
        draw_box = [x, y, x, y]
        sam_box = None   # 新画时清除旧 SAM 框
    elif event == cv2.EVENT_MOUSEMOVE and mouse_down:
        draw_box[2:] = [x, y]
        tmp = img.copy()
        cv2.rectangle(tmp, (draw_box[0], draw_box[1]), (draw_box[2], draw_box[3]), (0, 255, 0), 2)
        cv2.imshow('annotator', tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        draw_box[2:] = [x, y]
        print('prompt 框：', draw_box)
        # 立即用 SAM 优化
        if len(draw_box) == 4:
            input_box = np.array(draw_box, dtype=np.float32)
            masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
            if masks is not None:
                # 从 mask 取外接矩形作为新框
                y_idx, x_idx = np.where(masks[0] > 0)
                if x_idx.size and y_idx.size:
                    sam_box = [x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()]
                    print('SAM 优化框：', sam_box)
                else:
                    sam_box = None
                    print('SAM 未生成有效 mask')

# ----------- 单张图主循环 -----------
def process_one(labels):
    global img, img_disp, sam_box, draw_box, cls_map
    path = img_paths[curr_idx]
    print('=== ', path.name, f'  {curr_idx+1}/{len(img_paths)}  ===')
    img = cv2.imread(str(path))
    if img is None:
        print('读图失败，跳过')
        return
    h, w = img.shape[:2]

    # 画已有标注
    img_disp = img.copy()
    for line in labels[path.stem]:
        cls_id, xc, yc, bw, bh = line
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cls_name = id2name(int(cls_id))   # ← 用函数
        cv2.putText(img_disp, cls_name, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    predictor.set_image(img)
    sam_box = None
    draw_box = []

    while True:
        tmp = img_disp.copy()
        if draw_box:
            cv2.rectangle(tmp, (draw_box[0], draw_box[1]), (draw_box[2], draw_box[3]), (0, 255, 0), 2)
        if sam_box is not None:
            cv2.rectangle(tmp, (int(sam_box[0]), int(sam_box[1])), (int(sam_box[2]), int(sam_box[3])), (255, 0, 0), 2)
        cv2.imshow('annotator', tmp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('y'):
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
            line = yolo_line(cls_id, box, w, h)
            labels[path.stem].append(line)
            # 实时画到背景
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_disp, cls, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            sam_box = None
            draw_box = []
            print('已记录：', cls, box)

        elif key == ord('n'):
            if labels[path.stem]:
                removed = labels[path.stem].pop()
                print('撤销：', removed)
                save_one_label(labels, path)
                save_classes()          # ← 新增
                # 重绘背景
                img_disp = img.copy()
                for line in labels[path.stem]:
                    cls_id, xc, yc, bw, bh = line
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # 先拿名称，拿不到再用兜底
                    cls_name = id2name(int(cls_id))   # ← 用函数
                    cv2.putText(img_disp, cls_name, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                print('当前图片无标注可撤销')

        elif key == ord('a'):
            save_one_label(labels, path)
            save_classes()          # ← 新增

            return -1
        elif key == ord('d'):
            save_one_label(labels, path)
            save_classes()          # ← 新增

            return 1
        elif key == ord('q') or key == 27:
            save_one_label(labels, path)
            save_classes()          # ← 新增

            return 0

# ----------- 主入口 -----------
def main():
    global curr_idx, cls_map
    load_image_list()
    labels = load_or_init_labels()
    cv2.namedWindow('annotator', cv2.WINDOW_NORMAL)
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