"""极简手部框标注工具：拖框 -> 直接写 YOLO 格式到 labels/。

只做一件事：单类别 hand 的矩形框。不用装任何标注软件，
输出位置和格式就是 prepare_dataset.py / verify_pose.py 要读的那个。

用法：
    python annotate.py --dir to_label

操作：
    左键拖拽      画一个框
    u            撤销当前图最后一个框
    c            清空当前图所有框
    d / → / 空格  下一张（自动保存）
    a / ←        上一张（自动保存）
    g            跳到第一张未标注的
    q / ESC      退出（自动保存）

标注要点：框住**整只手**（指尖到手腕，含手套袖口），框比手大 10~30% 没关系，
实测偏 25% 都扛得住，但别松到手只占框的 1/3。
"""

import argparse
import glob
import os
import sys

import cv2

WIN = "annotate hand boxes  [drag=box  u=undo  c=clear  d/a=next/prev  g=first unlabeled  q=quit]"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


class Annotator:
    def __init__(self, img_dir, lab_dir, max_width):
        # ** 递归：直接开 dataset/ 时 images/ 下是 train/val 分层结构；
        # to_label 这类平铺目录同样兼容
        self.img_dir = img_dir
        self.files = sorted(f for f in glob.glob(os.path.join(img_dir, "**", "*"),
                                                 recursive=True)
                            if f.lower().endswith(IMG_EXT))
        if not self.files:
            sys.exit(f"{img_dir} 下没有图片")
        self.lab_dir = lab_dir
        os.makedirs(lab_dir, exist_ok=True)
        self.max_width = max_width
        self.i = 0
        self.boxes = []          # 原图坐标系下的 [x1,y1,x2,y2]
        self.img = None
        self.scale = 1.0
        self.drawing = False
        self.p0 = None
        self.p1 = None
        self.load(0)

    # ---------- 读写 ----------
    def lab_path(self, i=None):
        # labels/ 下保持与 images/ 相同的子目录结构
        # （images/train/x.jpg → labels/train/x.txt；平铺目录退化为同名）
        rel = os.path.relpath(self.files[self.i if i is None else i], self.img_dir)
        stem = os.path.splitext(rel)[0]
        return os.path.join(self.lab_dir, stem + ".txt")

    def load(self, i):
        self.i = i % len(self.files)
        self.img = cv2.imread(self.files[self.i])
        h, w = self.img.shape[:2]
        self.scale = min(1.0, self.max_width / w)
        self.boxes = []
        p = self.lab_path()
        if os.path.exists(p) and os.path.getsize(p) > 0:
            for line in open(p):
                t = line.split()
                if len(t) >= 5:
                    _c, cx, cy, bw, bh = (float(v) for v in t[:5])
                    self.boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                       (cx + bw / 2) * w, (cy + bh / 2) * h])

    def save(self):
        """无框也要写空文件——它标记「这张我看过了」，与「还没标」区分开。

        注意 prepare_dataset.py --merge 会跳过空文件，所以确实没有手的图
        不会被当成负样本混进训练集。
        """
        h, w = self.img.shape[:2]
        os.makedirs(os.path.dirname(self.lab_path()), exist_ok=True)
        with open(self.lab_path(), "w") as f:
            for x1, y1, x2, y2 in self.boxes:
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # ---------- 交互 ----------
    def on_mouse(self, event, x, y, flags, _param):
        ox, oy = x / self.scale, y / self.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing, self.p0, self.p1 = True, (ox, oy), (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.p1 = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.p1 = (ox, oy)
            x1, x2 = sorted((self.p0[0], self.p1[0]))
            y1, y2 = sorted((self.p0[1], self.p1[1]))
            if x2 - x1 > 8 and y2 - y1 > 8:      # 忽略误点产生的小框
                self.boxes.append([x1, y1, x2, y2])
            self.p0 = self.p1 = None

    def render(self):
        c = self.img.copy()
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            cv2.rectangle(c, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
            cv2.putText(c, f"hand{i+1}", (int(x1) + 4, max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        if self.drawing and self.p0 and self.p1:
            cv2.rectangle(c, (int(self.p0[0]), int(self.p0[1])),
                          (int(self.p1[0]), int(self.p1[1])), (0, 200, 255), 2)
        done = sum(1 for k in range(len(self.files)) if os.path.exists(self.lab_path(k)))
        tag = (f"[{self.i+1}/{len(self.files)}] {os.path.basename(self.files[self.i])}"
               f"   boxes={len(self.boxes)}   labeled={done}/{len(self.files)}")
        cv2.putText(c, tag, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(c, tag, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)
        if self.scale < 1.0:
            c = cv2.resize(c, None, fx=self.scale, fy=self.scale)
        return c

    def first_unlabeled(self):
        for k in range(len(self.files)):
            if not os.path.exists(self.lab_path(k)):
                return k
        return self.i

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                self.save()
                break
            elif k == ord("u") and self.boxes:
                self.boxes.pop()
            elif k == ord("c"):
                self.boxes = []
            elif k in (ord("d"), ord(" "), 83):
                self.save()
                self.load(self.i + 1)
            elif k in (ord("a"), 81):
                self.save()
                self.load(self.i - 1)
            elif k == ord("g"):
                self.save()
                self.load(self.first_unlabeled())
        cv2.destroyAllWindows()
        done = sum(1 for k in range(len(self.files)) if os.path.exists(self.lab_path(k)))
        print(f"已标注 {done}/{len(self.files)} 张 -> {self.lab_dir}")
        print("下一步:  python verify_pose.py --dir " + os.path.dirname(self.lab_dir))


def main():
    ap = argparse.ArgumentParser(description="手部框标注（单类别 hand，输出 YOLO 格式）")
    ap.add_argument("--dir", default="to_label",
                    help="含 images/ 的目录（支持子目录结构），标注写到同级 labels/ 相同结构")
    ap.add_argument("--max-width", type=int, default=1280, help="显示宽度上限")
    args = ap.parse_args()
    Annotator(os.path.join(args.dir, "images"),
              os.path.join(args.dir, "labels"), args.max_width).run()


if __name__ == "__main__":
    main()
