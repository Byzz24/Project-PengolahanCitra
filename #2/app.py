import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QListWidget, QListWidgetItem, QSlider, QGroupBox, QFormLayout, 
    QSpinBox, QFrame, QDoubleSpinBox, QScrollArea, QComboBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

METHODS = [
    "Image Negative",
    "Grayscale",
    "Histogram Equalization",
    "Threshold (Binary)",
    "Blurring/Smoothing",
    "Edge Detection",  # Menggabungkan berbagai jenis edge detection
    "Morphology (Open)",
    "Morphology (Close)",
    "Dilation",
    "Erosion",
    "Brightness/Contrast Adjustment",
    "Sharpen / Contrast"
]

METHOD_DESCRIPTIONS = {
    "Image Negative": "Membalik nilai piksel (Invert)",
    "Grayscale": "Konversi gambar ke skala abu-abu",
    "Histogram Equalization": "Pemerataan histogram standar",
    "Threshold (Binary)": "Konversi ke gambar biner hitam-putih",
    "Blurring/Smoothing": "Berbagai teknik blur dan smoothing gambar",
    "Edge Detection": "Berbagai teknik deteksi tepi pada gambar",
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
        self.setWindowTitle("Image Processing App - Enhanced (12 Fitur) - PySide6 + OpenCV")
        self.orig = None  # original cv image (BGR)
        self.result = None  # result cv image (BGR or gray)
        self._setup_ui()

    def _setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setStyleSheet("background-color: #f8f9fa;")
        
        # Left Panel - Controls
        left_panel = QFrame()
        left_panel.setFixedWidth(450)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }
        """)
        
        # Scroll area untuk left panel
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
        
        # Processing Methods Group
        method_group = QGroupBox("⚙️ Processing Methods (12 Fitur)")
        method_group.setStyleSheet(file_group.styleSheet())
        method_layout = QVBoxLayout()
        method_layout.setSpacing(10)
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
        
        # ListWidget untuk methods
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
        
        # Parameter Container - untuk menampung semua kontrol parameter
        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)
        self.param_layout.setSpacing(8)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        
        # Blur Type ComboBox
        self.blur_type_widget = QWidget()
        blur_type_layout = QHBoxLayout(self.blur_type_widget)
        blur_type_label = QLabel("Blur Type:")
        blur_type_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.blur_type_combo = QComboBox()
        self.blur_type_combo.addItems(["Gaussian Blur", "Median Blur", "Mean Blur", "Bilateral Filter"])
        self.blur_type_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
                color: #343a40;
                font-weight: bold;
            }
        """)
        blur_type_layout.addWidget(blur_type_label)
        blur_type_layout.addWidget(self.blur_type_combo)
        self.param_layout.addWidget(self.blur_type_widget)
        
        # Edge Detection Type ComboBox
        self.edge_type_widget = QWidget()
        edge_type_layout = QHBoxLayout(self.edge_type_widget)
        edge_type_label = QLabel("Edge Type:")
        edge_type_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.edge_type_combo = QComboBox()
        self.edge_type_combo.addItems(["Canny", "Sobel", "Laplacian"])
        self.edge_type_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 5px;
                background-color: #ffffff;
                color: #343a40;
                font-weight: bold;
            }
        """)
        edge_type_layout.addWidget(edge_type_label)
        edge_type_layout.addWidget(self.edge_type_combo)
        self.param_layout.addWidget(self.edge_type_widget)
        
        # Kernel Size Parameter
        self.kernel_widget = QWidget()
        kernel_layout = QHBoxLayout(self.kernel_widget)
        kernel_label = QLabel("Kernel Size:")
        kernel_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 31)
        self.kernel_slider.setValue(3)
        self.kernel_value = QLabel("3")
        self.kernel_value.setStyleSheet("font-weight: bold; color: #343a40;")
        self.kernel_slider.valueChanged.connect(lambda v: self.kernel_value.setText(str(v)))
        kernel_layout.addWidget(kernel_label)
        kernel_layout.addWidget(self.kernel_slider)
        kernel_layout.addWidget(self.kernel_value)
        self.param_layout.addWidget(self.kernel_widget)
        
        # Bilateral Filter Parameters
        self.bilateral_widget = QWidget()
        bilateral_layout = QHBoxLayout(self.bilateral_widget)
        bilateral_label = QLabel("Diameter:")
        bilateral_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.bilateral_slider = QSlider(Qt.Horizontal)
        self.bilateral_slider.setRange(5, 50)
        self.bilateral_slider.setValue(9)
        self.bilateral_value = QLabel("9")
        self.bilateral_value.setStyleSheet("font-weight: bold; color: #343a40;")
        self.bilateral_slider.valueChanged.connect(lambda v: self.bilateral_value.setText(str(v)))
        bilateral_layout.addWidget(bilateral_label)
        bilateral_layout.addWidget(self.bilateral_slider)
        bilateral_layout.addWidget(self.bilateral_value)
        self.param_layout.addWidget(self.bilateral_widget)
        
        # Sigma Parameters untuk Bilateral
        self.sigma_widget = QWidget()
        sigma_layout = QHBoxLayout(self.sigma_widget)
        sigma_label = QLabel("Sigma:")
        sigma_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setRange(10, 200)
        self.sigma_slider.setValue(75)
        self.sigma_value = QLabel("75")
        self.sigma_value.setStyleSheet("font-weight: bold; color: #343a40;")
        self.sigma_slider.valueChanged.connect(lambda v: self.sigma_value.setText(str(v)))
        sigma_layout.addWidget(sigma_label)
        sigma_layout.addWidget(self.sigma_slider)
        sigma_layout.addWidget(self.sigma_value)
        self.param_layout.addWidget(self.sigma_widget)
        
        # Canny Parameters
        self.canny_widget = QWidget()
        canny_layout = QVBoxLayout(self.canny_widget)
        
        # Threshold 1
        canny_thresh1_widget = QWidget()
        canny_thresh1_layout = QHBoxLayout(canny_thresh1_widget)
        canny_thresh1_label = QLabel("Threshold 1:")
        canny_thresh1_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.canny_thresh1_slider = QSlider(Qt.Horizontal)
        self.canny_thresh1_slider.setRange(10, 300)
        self.canny_thresh1_slider.setValue(100)
        self.canny_thresh1_value = QLabel("100")
        self.canny_thresh1_value.setStyleSheet("font-weight: bold; color: #343a40;")
        self.canny_thresh1_slider.valueChanged.connect(lambda v: self.canny_thresh1_value.setText(str(v)))
        canny_thresh1_layout.addWidget(canny_thresh1_label)
        canny_thresh1_layout.addWidget(self.canny_thresh1_slider)
        canny_thresh1_layout.addWidget(self.canny_thresh1_value)
        canny_layout.addWidget(canny_thresh1_widget)
        
        # Threshold 2
        canny_thresh2_widget = QWidget()
        canny_thresh2_layout = QHBoxLayout(canny_thresh2_widget)
        canny_thresh2_label = QLabel("Threshold 2:")
        canny_thresh2_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.canny_thresh2_slider = QSlider(Qt.Horizontal)
        self.canny_thresh2_slider.setRange(10, 300)
        self.canny_thresh2_slider.setValue(200)
        self.canny_thresh2_value = QLabel("200")
        self.canny_thresh2_value.setStyleSheet("font-weight: bold; color: #343a40;")
        self.canny_thresh2_slider.valueChanged.connect(lambda v: self.canny_thresh2_value.setText(str(v)))
        canny_thresh2_layout.addWidget(canny_thresh2_label)
        canny_thresh2_layout.addWidget(self.canny_thresh2_slider)
        canny_thresh2_layout.addWidget(self.canny_thresh2_value)
        canny_layout.addWidget(canny_thresh2_widget)
        
        self.param_layout.addWidget(self.canny_widget)
        
        # Sobel Parameters
        self.sobel_widget = QWidget()
        sobel_layout = QHBoxLayout(self.sobel_widget)
        sobel_label = QLabel("Kernel Size:")
        sobel_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.sobel_slider = QSlider(Qt.Horizontal)
        self.sobel_slider.setRange(1, 7)
        self.sobel_slider.setValue(3)
        self.sobel_value = QLabel("3")
        self.sobel_value.setStyleSheet("font-weight: bold; color: #343a40;")
        self.sobel_slider.valueChanged.connect(lambda v: self.sobel_value.setText(str(v)))
        sobel_layout.addWidget(sobel_label)
        sobel_layout.addWidget(self.sobel_slider)
        sobel_layout.addWidget(self.sobel_value)
        self.param_layout.addWidget(self.sobel_widget)
        
        # Threshold Parameter
        self.thresh_widget = QWidget()
        thresh_layout = QHBoxLayout(self.thresh_widget)
        thresh_label = QLabel("Threshold:")
        thresh_label.setStyleSheet("color: #343a40; font-weight: bold;")
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
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.spin_thresh)
        self.param_layout.addWidget(self.thresh_widget)
        
        # Brightness/Contrast Parameters
        self.brightness_widget = QWidget()
        brightness_layout = QHBoxLayout(self.brightness_widget)
        bright_label = QLabel("Brightness:")
        bright_label.setStyleSheet("color: #343a40; font-weight: bold;")
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
        brightness_layout.addWidget(bright_label)
        brightness_layout.addWidget(self.spin_brightness)
        self.param_layout.addWidget(self.brightness_widget)
        
        self.contrast_widget = QWidget()
        contrast_layout = QHBoxLayout(self.contrast_widget)
        contrast_label = QLabel("Contrast:")
        contrast_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.5, 3.0)
        self.spin_contrast.setValue(1.0)
        self.spin_contrast.setSingleStep(0.1)
        self.spin_contrast.setStyleSheet(self.spin_brightness.styleSheet())
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.spin_contrast)
        self.param_layout.addWidget(self.contrast_widget)
        
        # Sharpen Parameter
        self.sharpen_widget = QWidget()
        sharpen_layout = QHBoxLayout(self.sharpen_widget)
        sharpen_label = QLabel("Sharpness Factor:")
        sharpen_label.setStyleSheet("color: #343a40; font-weight: bold;")
        self.sharpen_slider = QSlider(Qt.Horizontal)
        self.sharpen_slider.setRange(50, 300)
        self.sharpen_slider.setValue(100)
        self.sharpen_value = QLabel("100")
        self.sharpen_value.setStyleSheet("font-weight: bold; color: #343a40;")
        self.sharpen_slider.valueChanged.connect(lambda v: self.sharpen_value.setText(str(v)))
        sharpen_layout.addWidget(sharpen_label)
        sharpen_layout.addWidget(self.sharpen_slider)
        sharpen_layout.addWidget(self.sharpen_value)
        self.param_layout.addWidget(self.sharpen_widget)

        # Tambahkan container parameter ke layout utama
        method_layout.addWidget(self.param_container)
        
        # Apply Button
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
        self.lbl_result.setStyleSheet(self.lbl_orig.styleSheet())
        
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

        # Initial control visibility
        self.method_changed()
        
        # Connect combo boxes to update parameters
        self.blur_type_combo.currentTextChanged.connect(self.update_blur_parameters)
        self.edge_type_combo.currentTextChanged.connect(self.update_edge_parameters)

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

    def update_blur_parameters(self):
        """Update parameter visibility based on selected blur type"""
        blur_type = self.blur_type_combo.currentText()
        
        # Hide all blur parameters first
        self.kernel_widget.setVisible(False)
        self.bilateral_widget.setVisible(False)
        self.sigma_widget.setVisible(False)
        
        # Show relevant parameters based on blur type
        if blur_type == "Gaussian Blur":
            self.kernel_widget.setVisible(True)
            self.kernel_slider.setRange(1, 31)
            self.kernel_slider.setValue(3)
            self.kernel_value.setText("3")
        elif blur_type == "Median Blur":
            self.kernel_widget.setVisible(True)
            self.kernel_slider.setRange(1, 31)
            self.kernel_slider.setValue(3)
            self.kernel_value.setText("3")
        elif blur_type == "Mean Blur":
            self.kernel_widget.setVisible(True)
            self.kernel_slider.setRange(1, 31)
            self.kernel_slider.setValue(3)
            self.kernel_value.setText("3")
        elif blur_type == "Bilateral Filter":
            self.bilateral_widget.setVisible(True)
            self.sigma_widget.setVisible(True)
            self.bilateral_slider.setRange(5, 50)
            self.bilateral_slider.setValue(9)
            self.bilateral_value.setText("9")
            self.sigma_slider.setRange(10, 200)
            self.sigma_slider.setValue(75)
            self.sigma_value.setText("75")

    def update_edge_parameters(self):
        """Update parameter visibility based on selected edge detection type"""
        edge_type = self.edge_type_combo.currentText()
        
        # Hide all edge detection parameters first
        self.canny_widget.setVisible(False)
        self.sobel_widget.setVisible(False)
        
        # Show relevant parameters based on edge type
        if edge_type == "Canny":
            self.canny_widget.setVisible(True)
            self.canny_thresh1_slider.setRange(10, 300)
            self.canny_thresh1_slider.setValue(100)
            self.canny_thresh1_value.setText("100")
            self.canny_thresh2_slider.setRange(10, 300)
            self.canny_thresh2_slider.setValue(200)
            self.canny_thresh2_value.setText("200")
        elif edge_type == "Sobel":
            self.sobel_widget.setVisible(True)
            self.sobel_slider.setRange(1, 7)
            self.sobel_slider.setValue(3)
            self.sobel_value.setText("3")
        elif edge_type == "Laplacian":
            # Laplacian tidak memerlukan parameter tambahan
            pass

    def method_changed(self):
        method = self.method_list.currentItem().text() if self.method_list.currentItem() else ""
        
        # Update deskripsi
        self.desc_label.setText(METHOD_DESCRIPTIONS.get(method, ""))
        
        # Sembunyikan semua parameter terlebih dahulu
        self.blur_type_widget.setVisible(False)
        self.edge_type_widget.setVisible(False)
        self.kernel_widget.setVisible(False)
        self.thresh_widget.setVisible(False)
        self.canny_widget.setVisible(False)
        self.sobel_widget.setVisible(False)
        self.bilateral_widget.setVisible(False)
        self.sigma_widget.setVisible(False)
        self.brightness_widget.setVisible(False)
        self.contrast_widget.setVisible(False)
        self.sharpen_widget.setVisible(False)
        
        # Tampilkan hanya parameter yang relevan
        if method == "Blurring/Smoothing":
            self.blur_type_widget.setVisible(True)
            self.update_blur_parameters()
            
        elif method == "Edge Detection":
            self.edge_type_widget.setVisible(True)
            self.update_edge_parameters()
            
        elif method == "Threshold (Binary)":
            self.thresh_widget.setVisible(True)
            self.spin_thresh.setValue(127)
            
        elif method == "Brightness/Contrast Adjustment":
            self.brightness_widget.setVisible(True)
            self.contrast_widget.setVisible(True)
            self.spin_brightness.setValue(0)
            self.spin_contrast.setValue(1.0)
            
        elif method == "Sharpen / Contrast":
            self.sharpen_widget.setVisible(True)
            self.sharpen_slider.setRange(50, 300)
            self.sharpen_slider.setValue(100)
            self.sharpen_value.setText("100")
        
        # Untuk metode tanpa parameter (Image Negative, Grayscale, Histogram Equalization, 
        # Morphology operations) - tidak menampilkan parameter apapun

    def apply_method(self):
        if self.orig is None:
            return
        method = self.method_list.currentItem().text() if self.method_list.currentItem() else ""
        img = self.orig.copy()
        
        try:
            if method == "Image Negative":
                inverted = cv2.bitwise_not(img)
                self.result = inverted
                
            elif method == "Grayscale":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.result = gray
            
            elif method == "Histogram Equalization":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                self.result = eq
            
            elif method == "Threshold (Binary)":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                t = int(self.spin_thresh.value())
                _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
                self.result = th
            
            elif method == "Blurring/Smoothing":
                blur_type = self.blur_type_combo.currentText()
                
                if blur_type == "Gaussian Blur":
                    k = int(self.kernel_slider.value())
                    if k % 2 == 0:  # Kernel harus ganjil
                        k += 1
                    if k < 1: k = 1
                    blur = cv2.GaussianBlur(img, (k, k), 0)
                    self.result = blur
                    
                elif blur_type == "Median Blur":
                    k = int(self.kernel_slider.value())
                    if k % 2 == 0:  # Kernel harus ganjil
                        k += 1
                    if k < 1: k = 1
                    med = cv2.medianBlur(img, k)
                    self.result = med
                    
                elif blur_type == "Mean Blur":
                    k = int(self.kernel_slider.value())
                    if k < 1: k = 1
                    # Mean blur menggunakan box filter
                    blur = cv2.blur(img, (k, k))
                    self.result = blur
                    
                elif blur_type == "Bilateral Filter":
                    d = int(self.bilateral_slider.value())
                    sigma = int(self.sigma_slider.value())
                    bilateral = cv2.bilateralFilter(img, d, sigma, sigma)
                    self.result = bilateral
            
            elif method == "Edge Detection":
                edge_type = self.edge_type_combo.currentText()
                
                if edge_type == "Canny":
                    threshold1 = int(self.canny_thresh1_slider.value())
                    threshold2 = int(self.canny_thresh2_slider.value())
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, threshold1, threshold2)
                    self.result = edges
                    
                elif edge_type == "Sobel":
                    k = int(self.sobel_slider.value())
                    if k % 2 == 0:  # Kernel harus ganjil
                        k += 1
                    if k < 1: k = 1
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
                    sobel = np.sqrt(sobelx**2 + sobely**2)
                    sobel = np.uint8(255 * sobel / np.max(sobel)) if np.max(sobel) > 0 else np.zeros_like(sobelx, dtype=np.uint8)
                    self.result = sobel
                    
                elif edge_type == "Laplacian":
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    laplacian = np.uint8(np.absolute(laplacian))
                    self.result = laplacian
            
            elif method == "Morphology (Open)":
                # Default kernel size 3x3 untuk morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
                self.result = opened
            
            elif method == "Morphology (Close)":
                # Default kernel size 3x3 untuk morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
                self.result = closed
            
            elif method == "Dilation":
                # Default kernel size 3x3 untuk dilation
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                dilated = cv2.dilate(th, kernel, iterations=1)
                self.result = dilated
            
            elif method == "Erosion":
                # Default kernel size 3x3 untuk erosion
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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
                kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
                sharp = cv2.filter2D(img, -1, kernel)
                out = cv2.convertScaleAbs(sharp, alpha=1, beta=0)
                self.result = out
            
            else:
                self.result = img

            self.update_previews()
            
        except Exception as e:
            print(f"Error applying method {method}: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())