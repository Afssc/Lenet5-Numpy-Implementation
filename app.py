from matplotlib import gridspec
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter
from model import *
# matplotlib.use("TkAgg")



class DigitDrawUI:
    def __init__(self, canvas_size=28, brush=2, model_path ="./full_trained/lenet5_full_acc94.pkl"):
        self.canvas_size = canvas_size
        self.brush = brush
        self.canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        self._is_down = False

        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(1, 3, hspace=0, wspace=0.3,width_ratios=[2,4,1.5])
        
        # Col 0: Input
        self.ax_draw = self.fig.add_subplot(gs[0, 0])
        self.ax_draw.set_title("Draw (28x28)")
        self.ax_draw.set_xticks([])
        self.ax_draw.set_yticks([])
        self.im = self.ax_draw.imshow(self.canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        
        # Col 2: internal features grid
        # subgrid col 1 = kernels 6rows, col 2 = C1 features 6rows, col 3 =s2 pooling features 6rows,
        #  col 4 = C3 features 16 rows, col 5 = s4 pooling features 16 rows
        # col 6 = C5 features 120x1 scalar col 7 = f6 output 12*7 graph 
        self.gs1 = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[0, 1], hspace=0, wspace=0,width_ratios=[1,4,2,2,1,0.8,0.8,])
        self.ax_c1_kernels = []
        
        def create_gs1_subplots(rows,gs1_col,title):
            ret = []
            for i in range(rows):
                gss = gridspec.GridSpecFromSubplotSpec(rows, 1, subplot_spec=self.gs1[0, gs1_col], hspace=0.1, wspace=0)
                ax = self.fig.add_subplot(gss[i, 0])
                if i == 0:
                    ax.set_title(title)                
                ax.set_xticks([])
                ax.set_yticks([])
                ret.append(ax)
            return ret
        
        self.ax_c1_kernels = create_gs1_subplots(6,0,"C1_k")
        self.ax_c1_features = create_gs1_subplots(6,1,"C1")
        self.ax_s2_pool = create_gs1_subplots(6,2,"S2")
        self.ax_c3_features = create_gs1_subplots(16,3,"C3")
        self.ax_s4_pool = create_gs1_subplots(16,4,"S4")
        self.ax_c5_features = create_gs1_subplots(1,5,"C5")
        self.ax_f6_flat = create_gs1_subplots(1,6,"F6")

        
        # Row 3: f6 output (84 -> 12x7)
        
        #Col 3: rbf prob bars and rbf templates
        self.gs_rbf = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,2], hspace=0.3, wspace=0)
        self.ax_prob = self.fig.add_subplot(self.gs_rbf[0, 0])
        self.ax_prob.set_title("RBF prob")
        self.bars = self.ax_prob.bar(range(10), np.zeros(10))
        self.ax_prob.set_ylim(0, 1)
        self.ax_prob.set_xticks(range(10))
        
        self.ax_f6_out = self.fig.add_subplot(self.gs_rbf[1,0])
        self.ax_f6_out.set_title("f6 output (12x7)")
        self.ax_f6_out.set_xticks([])
        self.ax_f6_out.set_yticks([])
        
        
        # Clear button
        ax_btn = self.fig.add_axes((0.20, 0.02, 0.05, 0.03))
        self.btn = Button(ax_btn, "Clear")
        self.btn.on_clicked(self.clear)
        self.refresh_counter = 0
 
        self.forward_outs = []

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        # model = Lenet5.load_model("./lenet5_5000_11epoch_testacc=84_best.pkl")
        model = Lenet5.load_model(model_path)
        self.layer_list = model.model
        self.update_probs()
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
        self.forward_outs = []
        x = img32.astype(np.float32) * 2.0 - 1.0
        x = x.reshape(1, 32, 32)
        out = x
        for layer in self.layer_list:
            out = layer.forward(out)
            self.forward_outs.append(out)
        d = np.array(out).reshape(-1)
        return self._softmax_neg_dist(d)

    def display_feature_maps(self, feature_maps, axes,title):
        num_maps = feature_maps.shape[0]
        for i in range(num_maps):
            feature = feature_maps[i]
            axes[i].clear()
            axes[i].imshow(feature, cmap="gray")
            if i==0:
                axes[i].set_title(f"{title}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])

    
    def _draw_internal_feature_subplots(self):
        """Extract C1 feature maps and f6 output."""
        
        # C1: Conv_layer
        c1_kernels = self.layer_list[0].m_kernel  # (6, 5, 5)
        # c1_features = self.layer_list[0].forward(out)  # (6, 28, 28)
        c1_features,s2_result,c3_features,s4_result,c5_result,f6_out,rbf_out = self.forward_outs
        c5_result_flat = c5_result.reshape((120,1))  # (120,)
        f6_out_flat = f6_out.reshape((84,1))  # (84,)
                
        self.display_feature_maps(c1_kernels, self.ax_c1_kernels, "C1_k")
        self.display_feature_maps(c1_features, self.ax_c1_features, "C1")
        self.display_feature_maps(s2_result, self.ax_s2_pool, "S2")
        self.display_feature_maps(c3_features, self.ax_c3_features, "C3")
        self.display_feature_maps(s4_result, self.ax_s4_pool, "S4")

        self.ax_c5_features[0].clear()
        self.ax_c5_features[0].imshow(c5_result_flat, cmap="gray")
        self.ax_c5_features[0].set_title("C5")
        self.ax_c5_features[0].set_xticks([])
        self.ax_c5_features[0].set_yticks([])
        
        self.ax_f6_flat[0].clear()
        self.ax_f6_flat[0].imshow(f6_out_flat, cmap="gray")
        self.ax_f6_flat[0].set_title("F6")
        self.ax_f6_flat[0].set_xticks([])
        self.ax_f6_flat[0].set_yticks([])

        f6_img = f6_out.reshape(12, 7)
        self.ax_f6_out.clear()
        self.ax_f6_out.imshow(f6_img, cmap="gray")
        self.ax_f6_out.set_title("f6 graph")
        self.ax_f6_out.set_xticks([])
        self.ax_f6_out.set_yticks([])


    def clear(self, _=None):
        self.canvas.fill(0)
        self.im.set_data(self.canvas)
        self.update_probs()
        self.fig.canvas.draw_idle()
        self.refresh_counter =0

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
        if self.refresh_counter % 10 ==0:
            self.im.set_data(self.canvas)
            self.ax_draw.draw_artist(self.im)
            self.fig.canvas.blit(self.ax_draw.bbox)
            # self.fig.canvas.flush_events()
            # self.fig.canvas.draw_idle()
            # self.update_probs()

    def update_probs(self):
        img28 = gaussian_filter(self.canvas.astype(np.float32), sigma=0.6)
        self.fig.canvas.draw_idle()
        self.im.set_data(img28)
        probs = self._predict_probs(img28)
        for i, b in enumerate(self.bars):
            b.set_height(float(probs[i]))
        self.ax_prob.set_ylim(0, max(1e-3, float(np.max(probs)) * 1.2))
        
        # Update internal features
        self._draw_internal_feature_subplots() 
        # Display f6 output (84,) -> (12, 7)

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
        self.refresh_counter+=1
        if self.refresh_counter%2 == 0:
            self.draw_at(event.xdata, event.ydata)

# 启动演示窗口
ui = DigitDrawUI(canvas_size=28, brush=1,model_path ="./full_trained/lenet5_full_acc94.pkl")