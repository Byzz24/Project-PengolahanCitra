import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QListWidget, QListWidgetItem, QSlider, QGroupBox, QFormLayout, 
    QSpinBox, QFrame, QDoubleSpinBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

METHODS = [
    "Image Negative",  # <-- DITAMBAHKAN
    "Grayscale",
    "Histogram Equalization",
    # "Adaptive Histogram Equalization (CLAHE)",  <-- DIHAPUS
    "Threshold (Binary)",
    "Gaussian Blur",
    "Median Blur",
    "Bilateral Filter",
    "Canny Edge Detection",
    "Sobel Edge Detection",
    "Laplacian Edge Detection",
    "Morphology (Open)",
    "Morphology (Close)",
    "Dilation",
    "Erosion",
    "Brightness/Contrast Adjustment",
    "Sharpen / Contrast"
]

METHOD_DESCRIPTIONS = {
    "Image Negative": "Membalik nilai piksel (Invert)",  # <-- DITAMBAHKAN
    "Grayscale": "Konversi gambar ke skala abu-abu",
    "Histogram Equalization": "Pemerataan histogram standar",
    # "Adaptive Histogram Equalization (CLAHE)": "Pemerataan histogram adaptif",  <-- DIHAPUS
    "Threshold (Binary)": "Konversi ke gambar biner hitam-putih",
    "Gaussian Blur": "Blur dengan kernel Gaussian",
    "Median Blur": "Blur median untuk noise reduction",
    "Bilateral Filter": "Filter yang menjaga edge sambil blur",
    "Canny Edge Detection": "Deteksi tepi Canny",
    "Sobel Edge Detection": "Deteksi tepi Sobel",
    "Laplacian Edge Detection": "Deteksi tepi Laplacian",
    "Morphology (Open)": "Operasi morfologi opening",
    "Morphology (Close)": "Operasi morfologi closing",
    "Dilation": "Operasi dilasi (memperbesar objek)",
    "Erosion": "Operasi erosi (memperkecil objek)",
    "Brightness/Contrast Adjustment": "Penyesuaian brightness dan contrast",
    "Sharpen / Contrast": "Penajaman dan peningkatan kontras"
}

def qimg_from_cv(img):
    """Convert an OpenCV image (BGR or gray) to QImage"""
    if img is None:
        return None
    if len(img.shape) == 2:  # grayscale
        h, w = img.shape
        bytes_per_line = w
        return QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
    else:
        h, w, ch = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        fig.tight_layout()

    def plot_hist(self, img, per_channel=True):
        self.ax.clear()
        if img is None:
            self.ax.text(0.5, 0.5, 'No Image', horizontalalignment='center', 
                         verticalalignment='center', transform=self.ax.transAxes,
                         fontsize=12, color='gray')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.draw()
            return
        
        if len(img.shape) == 2 or not per_channel:
            # grayscale histogram
            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.ax.hist(gray.ravel(), bins=256, range=(0,255), color='#495057', alpha=0.7)
            self.ax.set_title("Grayscale Histogram", fontsize=10, fontweight='bold', color='#343a40')
        else:
            # color histogram per channel
            chans = cv2.split(img)
            colors = ('#4285f4', '#34a853', '#ea4335')  # Blue, Green, Red
            labels = ('Blue', 'Green', 'Red')
            for ch, col, lbl in zip(chans, colors, labels):
                self.ax.hist(ch.ravel(), bins=256, range=(0,255), alpha=0.7, color=col, label=lbl)
            self.ax.legend(fontsize=8, loc='upper right')
            self.ax.set_title("Color Histogram", fontsize=10, fontweight='bold', color='#343a40')
        
        self.ax.set_xlim(0, 255)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#f8f9fa')
        self.ax.tick_params(axis='both', which='major', labelsize=8, colors='#343a40')
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing App - Enhanced (16 Fitur) - PySide6 + OpenCV")
        self.orig = None  # original cv image (BGR)
        self.result = None  # result cv image (BGR or gray)
        self._setup_ui()

    def _setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setStyleSheet("background-color: #f8f9fa;")
        
        # Left Panel - Controls (DIPERLEBAR)
        left_panel = QFrame()
        left_panel.setFixedWidth(420)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }
        """)
        
        # Scroll area untuk left panel agar semua bisa di-scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f8f9fa;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #ced4da;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #adb5bd;
            }
        """)
        
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # App Title
        title = QLabel("Image Processor Pro")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #343a40;
                padding: 10px;
                border-bottom: 2px solid #e9ecef;
            }
        """)
        left_layout.addWidget(title)
        
        # File Operations Group
        file_group = QGroupBox("📁 File Operations")
        file_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #495057;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #343a40;
            }
        """)
        file_layout = QVBoxLayout()
        
        btn_load = QPushButton("📁 Load Image")
        btn_save = QPushButton("💾 Save Result")
        btn_reset = QPushButton("🔄 Reset")
        
        for btn in [btn_load, btn_save, btn_reset]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffffff;
                    border: 1px solid #ced4da;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #495057;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
                QPushButton:pressed {
                    background-color: #dee2e6;
                }
            """)
        
        btn_load.clicked.connect(self.load_image)
        btn_save.clicked.connect(self.save_result)
        btn_reset.clicked.connect(self.reset)
        
        file_layout.addWidget(btn_load)
        file_layout.addWidget(btn_save)
        file_layout.addWidget(btn_reset)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # Processing Methods Group - DIPERBAIKI
        method_group = QGroupBox("⚙️ Processing Methods (16 Fitur)")
        method_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #495057;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #343a40;
            }
        """)
        method_layout = QVBoxLayout()
        method_layout.setSpacing(10)  # ✅ Method yang benar (bukan setVerticalSpacing)
        method_layout.setContentsMargins(10, 15, 10, 10)
        
        # Label untuk deskripsi
        self.desc_label = QLabel("Pilih metode processing")
        self.desc_label.setStyleSheet("""
            QLabel {
                background-color: #e7f3ff;
                border: 1px solid #b3d9ff;
                border-radius: 5px;
                padding: 8px;
                font-size: 11px;
                color: #004085;
                font-weight: bold;
            }
        """)
        self.desc_label.setWordWrap(True)
        method_layout.addWidget(self.desc_label)
        
        # ListWidget untuk methods (GANTI DARI COMBOBOX)
        self.method_list = QListWidget()
        self.method_list.setMaximumHeight(250)
        self.method_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ced4da;
                border-radius: 5px;
                background-color: #ffffff;
                outline: none;
                color: #343a40;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e9ecef;
                color: #343a40;
            }
            QListWidget::item:selected {
                background-color: #007bff;
                color: white;
                font-weight: bold;
            }
            QListWidget::item:hover {
                background-color: #e9ecef;
                color: #343a40;
            }
            QScrollBar:vertical {
                border: none;
                background: #f8f9fa;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #ced4da;
                border-radius: 5px;
            }
        """)
        
        for method in METHODS:
            item = QListWidgetItem(method)
            self.method_list.addItem(item)
        
        self.method_list.itemSelectionChanged.connect(self.method_changed)
        self.method_list.setCurrentRow(0)
        method_layout.addWidget(self.method_list)
        
        # Slider / param
        self.param_slider = QSlider(Qt.Horizontal)
        self.param_slider.setRange(1, 100)
        self.param_slider.setValue(3)
        self.param_slider.valueChanged.connect(lambda v: self.param_label.setText(str(v)))
        self.param_label = QLabel(str(self.param_slider.value()))
        self.param_label.setStyleSheet("font-weight: bold; color: #343a40;")
        
        param_layout = QHBoxLayout()
        param_label_text = QLabel("Parameter:")
        param_label_text.setStyleSheet("color: #343a40; font-weight: bold;")
        param_layout.addWidget(param_label_text)
        param_layout.addWidget(self.param_slider)
        param_layout.addWidget(self.param_label)
        method_layout.addLayout(param_layout)

        # Extra spin for threshold
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(0, 255)
        self.spin_thresh.setValue(127)
        self.spin_thresh.setStyleSheet("""
            QSpinBox {
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
                color: #343a40;
                font-weight: bold;
            }
        """)
        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("Threshold:")
        thresh_label.setStyleSheet("color: #343a40; font-weight: bold;")
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.spin_thresh)
        method_layout.addLayout(thresh_layout)

        # Double spin for brightness/contrast
        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.setRange(-100, 100)
        self.spin_brightness.setValue(0)
        self.spin_brightness.setSingleStep(5)
        self.spin_brightness.setStyleSheet("""
            QDoubleSpinBox {
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
                color: #343a40;
                font-weight: bold;
            }
        """)
        bright_layout = QHBoxLayout()
        bright_label = QLabel("Brightness:")
        bright_label.setStyleSheet("color: #343a40; font-weight: bold;")
        bright_layout.addWidget(bright_label)
        bright_layout.addWidget(self.spin_brightness)
        method_layout.addLayout(bright_layout)

        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.5, 3.0)
        self.spin_contrast.setValue(1.0)
        self.spin_contrast.setSingleStep(0.1)
        self.spin_contrast.setStyleSheet(self.spin_brightness.styleSheet())
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("Contrast:")
        contrast_label.setStyleSheet("color: #343a40; font-weight: bold;")
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.spin_contrast)
        method_layout.addLayout(contrast_layout)

        apply_btn = QPushButton("🚀 Apply Processing")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        apply_btn.clicked.connect(self.apply_method)
        method_layout.addWidget(apply_btn)
        
        method_group.setLayout(method_layout)
        left_layout.addWidget(method_group)
        
        # Set scroll area content
        scroll_widget = QWidget()
        scroll_widget.setLayout(left_layout)
        scroll_area.setWidget(scroll_widget)
        
        # Right Panel - Images and Histograms
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(20, 0, 20, 20)
        
        # Original Image Section
        orig_section = QFrame()
        orig_section.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }
        """)
        orig_layout = QVBoxLayout()
        orig_layout.setContentsMargins(15, 15, 15, 15)
        
        orig_label = QLabel("Original Image")
        orig_label.setAlignment(Qt.AlignCenter)
        orig_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #343a40;
                padding: 5px;
            }
        """)
        orig_layout.addWidget(orig_label)
        
        orig_content = QHBoxLayout()
        self.lbl_orig = QLabel("No image loaded")
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setFixedSize(400, 300)
        self.lbl_orig.setStyleSheet("""
            QLabel {
                border: 2px dashed #ced4da;
                border-radius: 8px;
                background-color: #f8f9fa;
                color: #6c757d;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        self.orig_hist_canvas = HistogramCanvas(self, width=4, height=2)
        self.orig_hist_canvas.setFixedSize(400, 200)
        
        orig_content.addWidget(self.lbl_orig)
        orig_content.addWidget(self.orig_hist_canvas)
        orig_layout.addLayout(orig_content)
        
        orig_section.setLayout(orig_layout)
        right_layout.addWidget(orig_section)
        
        # Result Image Section
        result_section = QFrame()
        result_section.setStyleSheet(orig_section.styleSheet())
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(15, 15, 15, 15)
        
        result_label = QLabel("Processed Image")
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet(orig_label.styleSheet())
        result_layout.addWidget(result_label)
        
        result_content = QHBoxLayout()
        self.lbl_result = QLabel("Processing result will appear here")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setFixedSize(400, 300)
        self.lbl_result.setStyleSheet("""
            QLabel {
                border: 2px dashed #ced4da;
                border-radius: 8px;
                background-color: #f8f9fa;
                color: #6c757d;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        self.result_hist_canvas = HistogramCanvas(self, width=4, height=2)
        self.result_hist_canvas.setFixedSize(400, 200)
        
        result_content.addWidget(self.lbl_result)
        result_content.addWidget(self.result_hist_canvas)
        result_layout.addLayout(result_content)
        
        result_section.setLayout(result_layout)
        right_layout.addWidget(result_section)
        
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(right_panel)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.resize(1400, 950)

        # initial control visibility
        self.method_changed()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.bmp)")
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        self.orig = img
        self.result = img.copy()
        self.update_previews()

    def save_result(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result", "", "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        ext = path.split('.')[-1]
        _, buf = cv2.imencode(f'.{ext}', self.result)
        buf.tofile(path)

    def reset(self):
        if self.orig is None:
            return
        self.result = self.orig.copy()
        self.update_previews()

    def update_previews(self):
        # Original image preview and histogram
        if self.orig is not None:
            qimg_o = qimg_from_cv(self.orig)
            pix_o = QPixmap.fromImage(qimg_o).scaled(
                self.lbl_orig.width(), self.lbl_orig.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_orig.setPixmap(pix_o)
            self.orig_hist_canvas.plot_hist(self.orig)
        else:
            self.lbl_orig.clear()
            self.lbl_orig.setText("No image loaded")
            self.orig_hist_canvas.plot_hist(None)

        # Result image preview and histogram
        if self.result is not None:
            qimg_r = qimg_from_cv(self.result)
            pix_r = QPixmap.fromImage(qimg_r).scaled(
                self.lbl_result.width(), self.lbl_result.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_result.setPixmap(pix_r)
            self.result_hist_canvas.plot_hist(self.result)
        else:
            self.lbl_result.clear()
            self.lbl_result.setText("Processing result will appear here")
            self.result_hist_canvas.plot_hist(None)

    def method_changed(self):
        method = self.method_list.currentItem().text() if self.method_list.currentItem() else ""
        
        # Update deskripsi
        self.desc_label.setText(METHOD_DESCRIPTIONS.get(method, ""))
        
        # Hide all controls by default
        self.param_slider.setEnabled(False)
        self.spin_thresh.setEnabled(False)
        self.spin_brightness.setEnabled(False)
        self.spin_contrast.setEnabled(False)
        
        if method in ("Gaussian Blur", "Median Blur", "Morphology (Open)", "Morphology (Close)"):
            self.param_slider.setEnabled(True)
            self.param_slider.setRange(1, 31)
            self.param_slider.setValue(3)
            self.param_label.setText(str(self.param_slider.value()))
        elif method == "Threshold (Binary)":
            self.spin_thresh.setEnabled(True)
        elif method in ("Canny Edge Detection", "Laplacian Edge Detection"):
            self.param_slider.setEnabled(True)
            self.param_slider.setRange(10, 300)
            self.param_slider.setValue(100)
        elif method == "Sobel Edge Detection":
            self.param_slider.setEnabled(True)
            self.param_slider.setRange(1, 10)
            self.param_slider.setValue(1)
        elif method == "Bilateral Filter":
            self.param_slider.setEnabled(True)
            self.param_slider.setRange(5, 50)
            self.param_slider.setValue(9)
        elif method in ("Dilation", "Erosion"):
            self.param_slider.setEnabled(True)
            self.param_slider.setRange(1, 20)
            self.param_slider.setValue(3)
        # elif method == "Adaptive Histogram Equalization (CLAHE)":  <-- DIHAPUS
        #     self.param_slider.setEnabled(True)
        #     self.param_slider.setRange(1, 40)
        #     self.param_slider.setValue(2)
        elif method == "Sharpen / Contrast":
            self.param_slider.setEnabled(True)
            self.param_slider.setRange(50, 300)
            self.param_slider.setValue(100)
        elif method == "Brightness/Contrast Adjustment":
            self.spin_brightness.setEnabled(True)
            self.spin_contrast.setEnabled(True)
        
        # "Image Negative" tidak ada di sini, jadi otomatis tidak ada parameter (sesuai)

    def apply_method(self):
        if self.orig is None:
            return
        method = self.method_list.currentItem().text() if self.method_list.currentItem() else ""
        img = self.orig.copy()
        
        if method == "Image Negative":  # <-- DITAMBAHKAN
            inverted = cv2.bitwise_not(img)
            self.result = inverted
            
        elif method == "Grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.result = gray
        
        elif method == "Histogram Equalization":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            self.result = eq
        
        # elif method == "Adaptive Histogram Equalization (CLAHE)":  <-- DIHAPUS
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     clip_limit = float(self.param_slider.value())
        #     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        #     eq = clahe.apply(gray)
        #     self.result = eq
        
        elif method == "Threshold (Binary)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t = int(self.spin_thresh.value())
            _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            self.result = th
        
        elif method == "Gaussian Blur":
            k = int(self.param_slider.value())
            if k % 2 == 0:
                k += 1
            blur = cv2.GaussianBlur(img, (k, k), 0)
            self.result = blur
        
        elif method == "Median Blur":
            k = int(self.param_slider.value())
            if k % 2 == 0:
                k += 1
            med = cv2.medianBlur(img, k)
            self.result = med
        
        elif method == "Bilateral Filter":
            d = int(self.param_slider.value())
            bilateral = cv2.bilateralFilter(img, d, 75, 75)
            self.result = bilateral
        
        elif method == "Canny Edge Detection":
            v = int(self.param_slider.value())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, v//2, v)
            self.result = edges
        
        elif method == "Sobel Edge Detection":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            k = int(self.param_slider.value())
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = np.uint8(255 * sobel / np.max(sobel))
            self.result = sobel
        
        elif method == "Laplacian Edge Detection":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            self.result = laplacian
        
        elif method == "Morphology (Open)":
            k = int(self.param_slider.value())
            if k < 1: k = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            self.result = opened
        
        elif method == "Morphology (Close)":
            k = int(self.param_slider.value())
            if k < 1: k = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            self.result = closed
        
        elif method == "Dilation":
            k = int(self.param_slider.value())
            if k < 1: k = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(th, kernel, iterations=1)
            self.result = dilated
        
        elif method == "Erosion":
            k = int(self.param_slider.value())
            if k < 1: k = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            eroded = cv2.erode(th, kernel, iterations=1)
            self.result = eroded
        
        elif method == "Brightness/Contrast Adjustment":
            brightness = float(self.spin_brightness.value())
            contrast = float(self.spin_contrast.value())
            adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
            self.result = adjusted
        
        elif method == "Sharpen / Contrast":
            alpha = self.param_slider.value() / 100.0
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp = cv2.filter2D(img, -1, kernel)
            out = cv2.convertScaleAbs(sharp, alpha=alpha, beta=0)
            self.result = out
        
        else:
            self.result = img

        self.update_previews()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())