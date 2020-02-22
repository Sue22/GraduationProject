# -*- coding:utf-8 -*-

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

import numpy as np
import os
import config
import math

from yolo import YOLO

import xml.dom.minidom
import xml.dom

# VOC数据集变量
_POSE = 'Unspecified'
_TRUNCATED = '0'
_DIFFICULT = '0'
_SEGMENTED = '0'


class LabelTool:
    def __init__(self, master):
        self.parent = master
        self.parent.title("自动图像标注工具")
        self.W, self.H = self.parent.maxsize()
        self.parent.state('zoomed')
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)

        # 鼠标状态初始化
        self.STATE = {'x': 0, 'y': 0}
        self.STATE_COCO = {'click': 0}

        # 全局变量初始化
        self.imageDir = ''
        self.imageDirPathBuffer = ''  # 当前路径名
        self.img = None
        self.tkimg = None  # 加载到画布上的图片
        self.imageList = []  # 文件路径
        self.imageTotal = 1
        self.imageCur = 0  # 图片序号
        self.cur = 0  # imageList的图片索引
        self.bboxIdList = []
        self.bboxList = []  # 边界框列表
        self.bboxPointList = []
        self.bboxId = None
        self.currLabel = None
        self.currBboxColor = None
        self.editbboxId = None
        self.wpercent = 1  # 放大倍数
        self.width = 0  # 图片原始宽度
        self.height = 0  # 图片原始高度
        self.depth = 0  # 图片原始深度
        self.o1 = None
        self.o2 = None
        self.o3 = None
        self.o4 = None
        self.zoomImgId = None
        self.zoomImg = None
        self.zoomImgCrop = None
        self.tkZoomImg = None
        self.hl = None  # 水平线
        self.vl = None  # 垂线
        self.editPointId = None
        self.imagename = None
        self.filename = None  # 当前文件路径
        self.label_filename = None  # xml格式标注
        self.label_filename1 = None  # txt格式标注
        self.objectLabelList = []
        self.EDIT = False

        # ------------------ GUI ---------------------

        # 控制面板
        self.openLabel = Label(self.frame, text="图片路径:")
        self.openLabel.place(x=10, y=10, width=71, height=16)
        self.openEntry = Label(self.frame, bg='white')
        self.openEntry.place(x=90, y=10, width=171, height=20)
        self.openBtn = Button(self.frame, text="打开图片", command=self.open_image)
        self.openBtn.place(x=270, y=10, width=75, height=20)

        self.imgSizeLabel = Label(self.frame, text="图像大小:")
        self.imgSizeLabel.place(x=10, y=40, width=71, height=16)
        self.imgSize = Label(self.frame, bg='white')
        self.imgSize.place(x=90, y=40, width=171, height=16)
        self.openDirBtn = Button(self.frame, text="打开目录", command=self.open_image_dir)
        self.openDirBtn.place(x=270, y=40, width=75, height=20)

        # 物体栏
        self.listBoxNameLabel = Label(self.frame, text="物体栏:")
        self.listBoxNameLabel.place(x=10, y=70, width=71, height=16)
        self.objectListBox = Listbox(self.frame)
        self.objectListBox.place(x=10, y=90, width=250, height=200)

        self.xLabel = Label(self.frame, bg='white', fg='red', anchor=W, text="x: ")
        self.xLabel.place(x=270, y=120, width=75, height=20)
        self.yLabel = Label(self.frame, bg='white', fg='red', anchor=W, text="y: ")
        self.yLabel.place(x=270, y=150, width=75, height=20)

        self.previousBtn = Button(self.frame, text="< 上一张", command=self.open_previous)
        self.previousBtn.place(x=10, y=300, width=65, height=20)
        self.previousBtn.bind_all('w', self.open_previous)
        self.previousBtn.bind_all('a', self.open_previous)
        self.parent.bind("<Key-Left>", self.open_previous)
        self.nextBtn = Button(self.frame, text="下一张 >", command=self.open_next)
        self.nextBtn.place(x=90, y=300, width=65, height=20)
        self.nextBtn.bind_all("s", self.open_next)
        self.nextBtn.bind_all("d", self.open_next)
        self.parent.bind("<Key-Right>", self.open_next)

        self.saveBtn = Button(self.frame, text="保存", command=self.save)
        self.saveBtn.place(x=170, y=300, width=65, height=20)
        self.helpBtn = Button(self.frame, text="帮助")
        self.helpBtn.place(x=275, y=300, width=65, height=20)

        self.delObjectBtn = Button(self.frame, text="删除", command=self.del_bbox)
        self.delObjectBtn.place(x=270, y=200, width=75, height=20)
        self.clearAllBtn = Button(self.frame, text="清除所有", command=self.clear_bbox)
        self.clearAllBtn.place(x=270, y=230, width=75, height=20)

        # 类栏
        self.classesNameLabel = Label(self.frame, text="类栏:")
        self.classesNameLabel.place(x=10, y=330, width=71, height=20)
        self.labelListBox = Listbox(self.frame)
        self.labelListBox.place(x=10, y=350, width=250, height=120)
        self.customClassesNameLabel = Label(self.frame, text="自定义类:")
        self.customClassesNameLabel.place(x=270, y=350, width=75, height=20)
        self.textBox = Entry(self.frame, text="Enter label")
        self.textBox.place(x=270, y=370, width=75, height=20)

        self.addLabelBtn = Button(self.frame, text="添加", command=self.add_label)
        self.addLabelBtn.place(x=270, y=420, width=75, height=20)
        self.delLabelBtn = Button(self.frame, text="删除", command=self.del_label)
        self.delLabelBtn.place(x=270, y=450, width=75, height=20)

        self.mb = Menubutton(self.frame, text="选择标注类", relief=RAISED)
        self.mb.menu = Menu(self.mb, tearoff=0)
        self.mb["menu"] = self.mb.menu
        self.mb.place(x=10, y=480, width=80, height=20)

        self.addCocoBtn = Button(self.frame, text="添加标注类", command=self.add_labels_coco)
        self.addCocoBtn.place(x=120, y=480, width=80, height=20)
        self.semiAutoBtn = Button(self.frame, text="自动标注", command=self.automate)
        self.semiAutoBtn.place(x=230, y=480, width=80, height=20)
        self.cocoLabels = config.labels_to_names.values()

        self.cocoIntVars = []

        for idxcoco, label_coco in enumerate(self.cocoLabels):
            self.cocoIntVars.append(IntVar())
            self.mb.menu.add_checkbutton(label=label_coco, variable=self.cocoIntVars[idxcoco])

        # 放大区
        self.zoomPanelLabel = Label(self.frame, text="放大区:")
        self.zoomPanelLabel.place(x=10, y=500, width=71, height=20)
        self.zoomcanvas = Canvas(self.frame, bg='white')
        self.zoomcanvas.place(x=10, y=520, width=130, height=130)

        # 主画布
        self.canvas_w = self.W - 370
        self.canvas_h = self.H - 80
        self.canvas = Canvas(self.frame, bg='white')
        self.canvas.bind("<Button-1>", self.mouse_click)
        self.canvas.bind("<Motion>", self.mouse_move, "+")
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)
        self.canvas.place(x=360, y=10, width=self.canvas_w, height=self.canvas_h)
        self.parent.bind("Escape", self.cancel_bbox)

        # 状态栏

        self.processingLabel = Label(self.frame, text="状态栏")
        self.processingLabel.place(x=360, y=self.canvas_h + 10, width=self.canvas_w / 2,
                                   height=20)
        self.imageIdxLabel = Label(self.frame, text="图片序号")
        self.imageIdxLabel.place(x=360 + self.canvas_w / 2, y=self.canvas_h + 10,
                                 width=self.canvas_w / 2, height=20)

    def open_image(self):
        self.filename = filedialog.askopenfilename(title="选择图片", filetypes=(("jpeg files", "*.jpg"),
                                                                            ("all files", "*.*")))
        if not self.filename:
            return None
        self.load_image(self.filename)

    def open_image_dir(self):
        self.imageDir = filedialog.askdirectory(title="选择文件夹")
        if not self.imageDir:
            return None
        self.imageList = os.listdir(self.imageDir)
        self.imageList = sorted(self.imageList)
        self.imageTotal = len(self.imageList)
        self.filename = None
        self.imageDirPathBuffer = self.imageDir
        self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])

    def load_image(self, file):
        # 加载图片
        self.img = Image.open(file)
        self.imageCur = self.cur + 1
        self.imageIdxLabel.config(text='     图片序号: %d / %d' % (self.imageCur, self.imageTotal))
        size_img = np.shape(self.img)
        self.height = size_img[0]
        self.width = size_img[1]
        self.depth = size_img[2]
        self.openEntry.config(text='%s' % file)
        self.imgSize.config(text='(%d, %d, %d)' % (size_img[1], size_img[0], size_img[2]))

        # 调整图片适应画布大小
        w, h = self.img.size
        if w >= h:
            baseW = self.canvas_w
            self.wpercent = (baseW / float(w))
            hsize = int((float(h) * float(self.wpercent)))
            self.img = self.img.resize((baseW, hsize), Image.ANTIALIAS)
        else:
            baseH = self.canvas_h
            self.wpercent = (baseH / float(h))
            wsize = int((float(w) * float(self.wpercent)))
            self.img = self.img.resize((wsize, baseH), Image.ANTIALIAS)

        self.tkimg = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=NW)

        # 加载标注
        self.clear_bbox()
        self.imagename = os.path.split(file)[-1].split('.')[0]

        if not os.path.exists('./xmlLabels'):
            os.mkdir('./xmlLabels')
        if not os.path.exists('./txtLabels'):
            os.mkdir('./txtLabels')

        label_name = self.imagename + '.xml'
        self.label_filename = os.path.join('./xmlLabels', label_name)
        label_name1 = self.imagename + '.txt'
        self.label_filename1 = os.path.join('./txtLabels', label_name1)

        if os.path.exists(self.label_filename1):
            with open(self.label_filename1) as f:
                for (ind, line) in enumerate(f):
                    if ind == 0:
                        continue
                    tmp = [t.strip() for t in line.split()]

                    x1 = float(tmp[0]) * self.wpercent
                    y1 = float(tmp[1]) * self.wpercent
                    x2 = float(tmp[2]) * self.wpercent
                    y2 = float(tmp[3]) * self.wpercent
                    self.bboxId = self.canvas.create_rectangle(x1, y1,
                                                               x2, y2,
                                                               width=2,
                                                               outline=config.COLORS[
                                                                   len(self.bboxList) % len(config.COLORS)])
                    o1 = self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="red")
                    o2 = self.canvas.create_oval(x2 - 3, y1 - 3, x2 + 3, y1 + 3, fill="red")
                    o3 = self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="red")
                    o4 = self.canvas.create_oval(x1 - 3, y2 - 3, x1 + 3, y2 + 3, fill="red")

                    self.bboxList.append((int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])))
                    self.objectLabelList.append(tmp[4])
                    self.bboxPointList.append(o1)
                    self.bboxPointList.append(o2)
                    self.bboxPointList.append(o3)
                    self.bboxPointList.append(o4)

                    self.bboxIdList.append(self.bboxId)
                    self.bboxId = None

                    self.objectListBox.insert(END,
                                              '(%d, %d) -> (%d, %d)' % (int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])) + ': ' + tmp[4])
                    self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                                  fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])



    def del_bbox(self):
        sel = self.objectListBox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.canvas.delete(self.bboxIdList[idx])
        self.canvas.delete(self.bboxPointList[idx * 4])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 1])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 2])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 3])
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectLabelList.pop(idx)
        self.objectListBox.delete(idx)

    def clear_bbox(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
        for idx in range(len(self.bboxPointList)):
            self.canvas.delete(self.bboxPointList[idx])
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.objectLabelList = []
        self.bboxPointList = []

    def open_previous(self, event=None):
        self.save()
        if self.cur > 0:
            self.cur -= 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def open_next(self, event=None):
        self.save()
        if self.cur < len(self.imageList) - 1:
            self.cur += 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def save(self):
        saveimgname = self.imagename + '.jpg'
        shape = [self.width, self.height, self.depth]
        doc = createXML(saveimgname, shape, self.objectLabelList, self.bboxList)
        writeXMLFile(doc, self.label_filename)
        with open(self.label_filename1, 'w') as f:
            f.write('%d\n' % len(self.bboxList))
            for ind in range(len(self.bboxList)):
                f.write(' '.join(map(str, self.bboxList[ind])) + ' ' + self.objectLabelList[ind] + '\n')


    def mouse_click(self, event):
        if self.canvas.find_enclosed(event.x - 5, event.y - 5, event.x + 5, event.y + 5):
            self.EDIT = True
            self.editPointId = int(self.canvas.find_enclosed(event.x - 5, event.y - 5, event.x + 5, event.y + 5)[0])
        else:
            self.EDIT = False

        # Set the initial point
        if self.EDIT:
            idx = self.bboxPointList.index(self.editPointId)
            self.editbboxId = self.bboxIdList[math.floor(idx / 4.0)]
            self.bboxId = self.editbboxId
            pidx = self.bboxIdList.index(self.editbboxId)
            pidx = pidx * 4
            self.o1 = self.bboxPointList[pidx]
            self.o2 = self.bboxPointList[pidx + 1]
            self.o3 = self.bboxPointList[pidx + 2]
            self.o4 = self.bboxPointList[pidx + 3]
            if self.editPointId == self.o1:
                a, b, c, d = self.canvas.coords(self.o3)
            elif self.editPointId == self.o2:
                a, b, c, d = self.canvas.coords(self.o4)
            elif self.editPointId == self.o3:
                a, b, c, d = self.canvas.coords(self.o1)
            elif self.editPointId == self.o4:
                a, b, c, d = self.canvas.coords(self.o2)
            self.STATE['x'], self.STATE['y'] = int((a + c) / 2), int((b + d) / 2)
        else:
            self.STATE['x'], self.STATE['y'] = event.x, event.y

    def mouse_move(self, event):
        self.xLabel.config(text='x: %d ' % event.x)
        self.yLabel.config(text='y: %d ' % event.y)
        self.zoom_view(event)
        if self.tkimg:
            # Horizontal and Vertical Line for precision
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)

    def mouse_drag(self, event):
        self.mouse_move(event)
        if self.bboxId:
            self.currBboxColor = self.canvas.itemcget(self.bboxId, "outline")
            self.canvas.delete(self.bboxId)
            self.canvas.delete(self.o1)
            self.canvas.delete(self.o2)
            self.canvas.delete(self.o3)
            self.canvas.delete(self.o4)
        if self.EDIT:
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=self.currBboxColor)
        else:
            self.currBboxColor = config.COLORS[len(self.bboxList) % len(config.COLORS)]
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=self.currBboxColor)

    def mouse_release(self, event):
        try:
            labelidx = self.labelListBox.curselection()
            self.currLabel = self.labelListBox.get(labelidx)
        except:
            pass
        if self.EDIT:
            self.update_bbox()
            self.EDIT = False
        x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
        y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)

        o1 = self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="red")
        o2 = self.canvas.create_oval(x2 - 3, y1 - 3, x2 + 3, y1 + 3, fill="red")
        o3 = self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="red")
        o4 = self.canvas.create_oval(x1 - 3, y2 - 3, x1 + 3, y2 + 3, fill="red")

        x1 = x1 * self.wpercent
        x2 = x2 * self.wpercent
        y1 = y1 * self.wpercent
        y2 = y2 * self.wpercent

        self.bboxList.append((x1, y1, x2, y2))

        self.bboxPointList.append(o1)
        self.bboxPointList.append(o2)
        self.bboxPointList.append(o3)
        self.bboxPointList.append(o4)
        self.bboxIdList.append(self.bboxId)
        self.bboxId = None
        self.objectLabelList.append(str(self.currLabel))
        self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' + str(self.currLabel))
        self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                      fg=self.currBboxColor)
        self.currLabel = None

    def zoom_view(self, event):
        try:
            if self.zoomImgId:
                self.zoomcanvas.delete(self.zoomImgId)
            self.zoomImg = self.img.copy()
            self.zoomImgCrop = self.zoomImg.crop(((event.x - 25), (event.y - 25), (event.x + 25), (event.y + 25)))
            self.zoomImgCrop = self.zoomImgCrop.resize((150, 150))
            self.tkZoomImg = ImageTk.PhotoImage(self.zoomImgCrop)
            self.zoomImgId = self.zoomcanvas.create_image(0, 0, image=self.tkZoomImg, anchor=NW)
            hl = self.zoomcanvas.create_line(0, 75, 150, 75, width=2)
            vl = self.zoomcanvas.create_line(75, 0, 75, 150, width=2)
        except:
            pass

    def update_bbox(self):
        idx = self.bboxIdList.index(self.editbboxId)
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectListBox.delete(idx)
        self.currLabel = self.objectLabelList[idx]
        self.objectLabelList.pop(idx)
        idx = idx * 4
        self.canvas.delete(self.bboxPointList[idx])
        self.canvas.delete(self.bboxPointList[idx + 1])
        self.canvas.delete(self.bboxPointList[idx + 2])
        self.canvas.delete(self.bboxPointList[idx + 3])
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)

    def cancel_bbox(self):
        if self.STATE['click'] == 1:
            if self.bboxId:
                self.canvas.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def add_label(self):
        if self.textBox.get() is not '':
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if self.textBox.get() not in curr_label_list:
                self.labelListBox.insert(END, str(self.textBox.get()))
            self.textBox.delete(0, 'end')

    def del_label(self):
        labelidx = self.labelListBox.curselection()
        self.labelListBox.delete(labelidx)

    def add_labels_coco(self):
        for listidxcoco, list_label_coco in enumerate(self.cocoLabels):
            if self.cocoIntVars[listidxcoco].get():
                curr_label_list = self.labelListBox.get(0, END)
                curr_label_list = list(curr_label_list)
                if list_label_coco not in curr_label_list:
                    self.labelListBox.insert(END, str(list_label_coco))

    def automate(self):
        self.processingLabel.config(text="处理中    ")
        self.processingLabel.update_idletasks()
        self.clear_bbox()

        boxes, labels, scores = yolo.detect_image(self.img)
        for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if score < 0.5:
                continue

            if config.labels_to_names[label] not in curr_label_list:
                continue

            b = box.astype(int)

            self.bboxId = self.canvas.create_rectangle(b[1], b[0],
                                                       b[3], b[2],
                                                       width=2,
                                                       outline=config.COLORS[len(self.bboxList) % len(config.COLORS)])

            o1 = self.canvas.create_oval(b[1] - 3, b[0] - 3, b[1] + 3, b[0] + 3, fill="red")
            o2 = self.canvas.create_oval(b[3] - 3, b[0] - 3, b[3] + 3, b[0] + 3, fill="red")
            o3 = self.canvas.create_oval(b[3] - 3, b[2] - 3, b[3] + 3, b[2] + 3, fill="red")
            o4 = self.canvas.create_oval(b[1] - 3, b[2] - 3, b[1] + 3, b[2] + 3, fill="red")

            b[0] = int(b[0] / self.wpercent)
            b[1] = int(b[1] / self.wpercent)
            b[2] = int(b[2] / self.wpercent)
            b[3] = int(b[3] / self.wpercent)

            self.bboxList.append((b[1], b[0], b[3], b[2]))
            self.bboxPointList.append(o1)
            self.bboxPointList.append(o2)
            self.bboxPointList.append(o3)
            self.bboxPointList.append(o4)
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.objectLabelList.append(str(config.labels_to_names[label]))
            self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (b[1], b[0], b[3], b[2]) + ': ' +
                                      str(config.labels_to_names[label]))
            self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                          fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])
        self.processingLabel.config(text="处理完成             ")


def createXML(saveimgname, shape,objectLabelList, bbox):
    my_dom = xml.dom.getDOMImplementation()
    doc = my_dom.createDocument(None, 'annotation', None)

    root_node = doc.documentElement
    createChildNode(doc, 'folder', 'VOC2007', root_node)
    createChildNode(doc, 'filename', saveimgname, root_node)

    size_node = doc.createElement('size')
    createChildNode(doc, 'width', str(shape[0]), size_node)
    createChildNode(doc, 'height', str(shape[1]), size_node)
    createChildNode(doc, 'depth', str(shape[2]), size_node)
    root_node.appendChild(size_node)

    createChildNode(doc, 'segmented', _SEGMENTED, root_node)

    object_node = createObjectNode(doc, objectLabelList, bbox)
    root_node.appendChild(object_node)
    return doc


def createElementNode(doc, tag, attr):
    element_node = doc.createElement(tag)
    text_node = doc.createTextNode(attr)
    element_node.appendChild(text_node)
    return element_node


def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)
    parent_node.appendChild(child_node)


def createObjectNode(doc, classes_name, bbox):
    object_node = doc.createElement('object')
    for i in range(len(classes_name)):
        class_name = classes_name[i]
        boxes = bbox[i]
        createChildNode(doc, 'name', class_name, object_node)
        createChildNode(doc, 'pose', _POSE, object_node)
        createChildNode(doc, 'truncated', _TRUNCATED, object_node)
        createChildNode(doc, 'difficult', _DIFFICULT, object_node)

        bndbox_node = doc.createElement('bndbox')
        createChildNode(doc, 'xmin', str(boxes[0]), bndbox_node)
        createChildNode(doc, 'ymin', str(boxes[1]), bndbox_node)
        createChildNode(doc, 'xmax', str(boxes[2]), bndbox_node)
        createChildNode(doc, 'ymax', str(boxes[3]), bndbox_node)
        object_node.appendChild(bndbox_node)
    return object_node


def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')
    doc.writexml(tmpfile, addindent=' ' * 4, newl='\n', encoding='utf-8')
    tmpfile.close()

    fin = open('tmp.xml')
    fout = open(filename, 'w')
    lines = fin.readlines()

    for line in lines[1:]:
        if line.split():
            fout.writelines(line)
    fin.close()
    fout.close()
    os.remove('tmp.xml')


if __name__ == '__main__':
    yolo = YOLO()
    root = Tk()
    root.iconbitmap('favicon.ico')
    tool = LabelTool(root)
    root.mainloop()
