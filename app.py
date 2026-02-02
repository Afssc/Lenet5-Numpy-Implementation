from re import S
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter
from model import *
import model



class DigitDrawUI:
    def __init__(self, canvas_size=28, brush=2, model_path ="./full_trained/lenet5_full_acc94.pkl"):
        self.canvas_size = canvas_size
        self.brush = brush
        self.canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        self._is_down = False

        self.fig = plt.figure(figsize=(8, 4))
        self.ax_draw = self.fig.add_subplot(1, 2, 1)
        self.ax_prob = self.fig.add_subplot(1, 2, 2)
        self.ax_draw.set_title("Draw (28x28)")
        self.ax_draw.set_xticks([])
        self.ax_draw.set_yticks([])
        self.im = self.ax_draw.imshow(self.canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

        self.ax_prob.set_title("RBF prob")
        self.bars = self.ax_prob.bar(range(10), np.zeros(10))
        self.ax_prob.set_ylim(0, 1)
        self.ax_prob.set_xticks(range(10))

        ax_btn = self.fig.add_axes((0.20, 0.02, 0.20, 0.08))
        self.btn = Button(ax_btn, "Clear")
        self.btn.on_clicked(self.clear)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        # model = Lenet5.load_model("./lenet5_5000_11epoch_testacc=84_best.pkl")
        model = Lenet5.load_model(model_path)
        self.layer_lists = model.model

        # self.update_probs()
        plt.show()

    def _softmax_neg_dist(self, d: np.ndarray) -> np.ndarray:
        # d: (10,)  距离越小概率越大
        x = -d.astype(np.float64)
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

    def _predict_probs(self, img28: np.ndarray) -> np.ndarray:
        # img28: (28,28) in [0,1]
        img32 = np.pad(img28, ((2, 2), (2, 2)), 'constant', constant_values=0)
        x = img32.astype(np.float32) * 2.0 - 1.0
        x = x.reshape(1, 32, 32)
        out = x
        for layer in self.layer_lists:
            out = layer.forward(out)
        d = np.array(out).reshape(-1)
        return self._softmax_neg_dist(d)
    def clear(self, _=None):
        self.canvas.fill(0)
        self.im.set_data(self.canvas)
        self.update_probs()
        self.fig.canvas.draw_idle()

    def draw_at(self, x, y):
        if x is None or y is None:
            return
        xi = int(round(x))
        yi = int(round(y))
        r = self.brush
        x0 = max(0, xi - r)
        x1 = min(self.canvas_size, xi + r )
        y0 = max(0, yi - r)
        y1 = min(self.canvas_size, yi + r )
        self.canvas[y0:y1, x0:x1] = 1.0
        self.im.set_data(self.canvas)
        self.fig.canvas.draw_idle()
        # self.update_probs()

    def update_probs(self):

        img28 = gaussian_filter(self.canvas.astype(np.float32), sigma=0.6)
        # print(self.canvas)
        # print(images[2]+1)
        self.fig.canvas.draw_idle()
        # img28 = self.canvas.astype(np.float32)
        self.im.set_data(img28)
        probs = self._predict_probs(img28)
        for i, b in enumerate(self.bars):
            b.set_height(float(probs[i]))
        self.ax_prob.set_ylim(0, max(1e-3, float(np.max(probs)) * 1.2))

    def on_press(self, event):
        if event.inaxes != self.ax_draw:
            return
        self._is_down = True
        self.draw_at(event.xdata, event.ydata)

    def on_release(self, event):
        self.update_probs()
        self._is_down = False

    def on_move(self, event):
        if not self._is_down or event.inaxes != self.ax_draw:
            return
        self.draw_at(event.xdata, event.ydata)

# 启动演示窗口
ui = DigitDrawUI(canvas_size=28, brush=1,model_path ="./full_trained/lenet5_full_acc94.pkl")
# plt.figure()
# plt.imshow(ui.canvas)
# plt.show()
# plt.imshow(images[5])
# plt.show()