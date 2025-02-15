import sys
import os
import json
import io  # Import io to use BytesIO
import numpy as np
import tifffile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QColorDialog, QLabel, QVBoxLayout, 
    QWidget, QPushButton, QHBoxLayout, QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem, QGraphicsPathItem, QInputDialog, QAction, QGraphicsPolygonItem, QDialog
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QPainterPath, QShowEvent, QIcon, QPolygonF, QBrush
from PyQt5.QtCore import Qt, QPoint, QEvent, QBuffer
from PyQt5.QtSvg import QGraphicsSvgItem
from PIL import Image, ImageDraw, ImageChops
from skimage.filters import gabor, sobel
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage import img_as_ubyte, img_as_float
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import scipy.signal as sig
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu
from typing import Optional, Dict, Tuple, List

class CustomGraphicsView(QGraphicsView):

    """Custom QGraphicsView for handling image display and user interactions.
    
    Parameters
    ----------
    main_window : ImageSegmentationApp
        Reference to the parent main window instance.
    
    Attributes
    ----------
    main_window : ImageSegmentationApp
        Reference to parent main window
    zoom_factor : float
        Zoom multiplier for scroll events
    last_mouse_pos : QPoint
        Last recorded mouse position
    current_path_item : Optional[QGraphicsPathItem]
        Current drawing path item
    current_path : Optional[QPainterPath]
        Current drawing path
    """

    def __init__(self, main_window: 'ImageSegmentationApp'):
        """Initialize view with default parameters and setup scene."""
        super().__init__()
        self.main_window = main_window  # Reference to main window
        self.zoom_factor = 1.05
        self.last_mouse_pos: Optional[QPoint] = None
        self.current_path_item: Optional[QGraphicsPathItem] = None
        self.current_path: Optional[QPainterPath] = None

        # View configuration
        self.viewport().setCursor(Qt.CrossCursor)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Scene setup
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event: QEvent)  -> None:
        """Handle mouse wheel events for zooming in and out."""
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1/self.zoom_factor, 1/self.zoom_factor)

    def mousePressEvent(self, event: QEvent) -> None:

        """Handle mouse press events for both left and right mouse buttons.
        
        If the left mouse button is pressed, it records the current mouse position
        and passes the scene position to the main window's handle_left_press method.
        
        If the right mouse button is pressed, it sets the drag mode to ScrollHandDrag
        and calls the parent class's mousePressEvent method.
        """

        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            scene_pos = self.mapToScene(event.pos())
            self.main_window.handle_left_press(scene_pos)  # Call main window method
        elif event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QEvent) -> None:

        """
        Handle mouse move events for updating the scene position.

        If the last mouse position is recorded, this method maps the current
        mouse position to the scene coordinates and calls the main window's
        handle_left_move method with the updated position. It then calls the
        parent class's mouseMoveEvent method to ensure default behavior is maintained.

        Parameters
        ----------
        event : QEvent
            The mouse event containing information about the current mouse position.
        """

        if self.last_mouse_pos:
            scene_pos = self.mapToScene(event.pos())
            self.main_window.handle_left_move(scene_pos)  # Call main window method
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QEvent) -> None:

        """
        Handle mouse release events for left and right mouse buttons.

        If the left mouse button is released, it calls the main window's
        handle_left_release method with the current scene position and
        resets the last mouse position.

        If the right mouse button is released, it sets the drag mode to
        NoDrag to prevent accidental panning.

        Parameters
        ----------
        event : QEvent
            The mouse event containing information about the current mouse position.
        """

        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None
            scene_pos = self.mapToScene(event.pos())
            self.main_window.handle_left_release(scene_pos)  # Call main window method
        elif event.button() == Qt.RightButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

class ImageSegmentationApp(QMainWindow):

    """Main application window for image segmentation and analysis.
    
    Attributes
    ----------
    original_image : Optional[Image.Image]
        Original loaded image for resetting
    processed_data : Optional[np.ndarray]
        Current processed image data (grayscale)
    class_colors : Dict[Tuple[int, int, int], int]
        RGB to class index mapping
    class_names : Dict[int, str]
        Class index to name mapping
    labels : List[Tuple[List[QPoint], QColor]]
        Stored label polygons with colors
    """

    def __init__(self):
        """Initialize application window and UI components."""
        super().__init__()
        self.setWindowTitle('SeisSeg App')
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.statusBar = self.statusBar()
        self.label_counter = 0
        self.log_file_path = None

        # Add menu bar to the main window (QMainWindow)
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu('File')
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

        # Add close polygon shortcut
        self.close_action = QAction(self)
        self.close_action.setShortcut('C')
        self.close_action.triggered.connect(self.close_current_polygon)
        self.addAction(self.close_action)

        # Add Edit menu with Undo/Redo actions
        edit_menu = menu_bar.addMenu('Edit')
        undo_action = QAction('Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction('Redo', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        # Add "Apply" menu
        apply_menu = menu_bar.addMenu('Apply')
        apply_menu.addAction('Gabor Filter', self.apply_gabor)
        apply_menu.addAction('Local Binary Pattern (LBP)', self.apply_lbp)
        apply_menu.addAction('GLCM', self.apply_glcm)
        apply_menu.addAction('Gradient of Texture (GoT)', self.apply_got)
        apply_menu.addAction('Histogram of Gradients (HOG)', self.apply_hog)

        # New seismic attributes
        apply_menu.addAction('Semblance', self.apply_semblance)
        apply_menu.addAction('Amplitude Envelope', self.apply_amplitude_envelope)
        apply_menu.addAction('Instantaneous Frequency', self.apply_instantaneous_frequency)
        apply_menu.addAction('Butterworth Filter', self.apply_butterworth)
        apply_menu.addAction('Multi-Otsu Threshold', self.apply_multi_otsu)
        apply_menu.addAction('Spectral Decomposition', self.apply_spectral_decomposition)
        apply_menu.addAction('Structural Smoothing', self.apply_structural_smoothing)

        # Advanced Attributes
        apply_menu.addAction('Structural Curvature', self.apply_curvature)
        apply_menu.addAction('Structure-Oriented Filtering', self.apply_dip_steering)

        # Add "Reset" menu
        apply_menu.addAction('Reset to Original', self.reset_to_original)

        # Add Color Palette menu
        palette_menu = menu_bar.addMenu('Color Palettes')
        seismic_action = QAction('Seismic', self)
        seismic_action.triggered.connect(lambda: self.apply_color_palette('seismic'))
        palette_menu.addAction(seismic_action)

        rainbow_action = QAction('Rainbow', self)
        rainbow_action.triggered.connect(lambda: self.apply_color_palette('rainbow'))
        palette_menu.addAction(rainbow_action)

        viridis_action = QAction('Viridis', self)
        viridis_action.triggered.connect(lambda: self.apply_color_palette('viridis'))
        palette_menu.addAction(viridis_action)

        jet_action = QAction('Jet', self)
        jet_action.triggered.connect(lambda: self.apply_color_palette('jet'))
        palette_menu.addAction(jet_action)

        hot_action = QAction('Hot', self)
        hot_action.triggered.connect(lambda: self.apply_color_palette('hot'))
        palette_menu.addAction(hot_action)

        terrain_action = QAction('Terrain', self)
        terrain_action.triggered.connect(lambda: self.apply_color_palette('terrain'))
        palette_menu.addAction(terrain_action)

        gistearth_action = QAction('Gist Earth', self)
        gistearth_action.triggered.connect(lambda: self.apply_color_palette('gist_earth'))
        palette_menu.addAction(gistearth_action)

        flag_action = QAction('Flag', self)
        flag_action.triggered.connect(lambda: self.apply_color_palette('flag'))
        palette_menu.addAction(flag_action)

        prism_action = QAction('Prism', self)
        prism_action.triggered.connect(lambda: self.apply_color_palette('prism'))
        palette_menu.addAction(prism_action)

        ocean_action = QAction('Ocean', self)
        ocean_action.triggered.connect(lambda: self.apply_color_palette('ocean'))
        palette_menu.addAction(ocean_action)

        giststern_action = QAction('Gist Stern', self)
        giststern_action.triggered.connect(lambda: self.apply_color_palette('gist_stern'))
        palette_menu.addAction(giststern_action)

        gnuplot_action = QAction('GNU Plot', self)
        gnuplot_action.triggered.connect(lambda: self.apply_color_palette('gnuplot'))
        palette_menu.addAction(gnuplot_action)

        gnuplotv2_action = QAction('GNU Plot v2', self)
        gnuplotv2_action.triggered.connect(lambda: self.apply_color_palette('gnuplot2'))
        palette_menu.addAction(gnuplotv2_action)

        cmrmap_action = QAction('CMRmap', self)
        cmrmap_action.triggered.connect(lambda: self.apply_color_palette('CMRmap'))
        palette_menu.addAction(cmrmap_action)

        cubehelix_action = QAction('Cube Helix', self)
        cubehelix_action.triggered.connect(lambda: self.apply_color_palette('cubehelix'))
        palette_menu.addAction(cubehelix_action)

        brg_action = QAction('BRG', self)
        brg_action.triggered.connect(lambda: self.apply_color_palette('brg'))
        palette_menu.addAction(brg_action)

        gistrainbow_action = QAction('Gist Rainbow', self)
        gistrainbow_action.triggered.connect(lambda: self.apply_color_palette('gist_rainbow'))
        palette_menu.addAction(gistrainbow_action)

        turbo_action = QAction('Turbo', self)
        turbo_action.triggered.connect(lambda: self.apply_color_palette('turbo'))
        palette_menu.addAction(turbo_action)

        nipyspectral_action = QAction('Nipy Spectral', self)
        nipyspectral_action.triggered.connect(lambda: self.apply_color_palette('nipy_spectral'))
        palette_menu.addAction(nipyspectral_action)

        gistncar_action = QAction('Gist Ncar', self)
        gistncar_action.triggered.connect(lambda: self.apply_color_palette('gist_ncar'))
        palette_menu.addAction(gistncar_action)

        plasma_action = QAction('Plasma', self)
        plasma_action.triggered.connect(lambda: self.apply_color_palette('plasma'))
        palette_menu.addAction(plasma_action)

        inferno_action = QAction('Inferno', self)
        inferno_action.triggered.connect(lambda: self.apply_color_palette('inferno'))
        palette_menu.addAction(inferno_action)

        magma_action = QAction('Magma', self)
        magma_action.triggered.connect(lambda: self.apply_color_palette('magma'))
        palette_menu.addAction(magma_action)

        cividis_action = QAction('Cividis', self)
        cividis_action.triggered.connect(lambda: self.apply_color_palette('cividis'))
        palette_menu.addAction(cividis_action)

        # Add Help menu after other menus
        help_menu = menu_bar.addMenu('Help')
        
        # Add Help action
        help_action = QAction('Documentation', self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        # Add About action
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Add original image storage
        self.original_image = None  # Store original image for resetting
        self.current_image = None   # Store modified image

        # Graphics View
        self.view = CustomGraphicsView(self)
        self.layout.addWidget(self.view)
        self.scene = self.view.scene

        # Add SVG logo to the scene
        self.logo_item = QGraphicsSvgItem('img/seisseg_logo_cc.svg')

        self.label = QLabel()
        self.label.setText('SeisSeg v0.1.7')
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.scene.addItem(self.logo_item)

        # Set window title
        self.setWindowTitle('SeisSeg App')

        # Set window icon
        icon = QIcon('img/seisseg_icon_n.png')
        self.setWindowIcon(icon)

        # Buttons layout
        button_layout = QHBoxLayout()
        self.open_button = QPushButton('Open Image')
        self.open_button.clicked.connect(self.open_image)
        button_layout.addWidget(self.open_button)

        self.color_button = QPushButton('Create Class w/ Color')
        self.color_button.clicked.connect(self.choose_color)
        button_layout.addWidget(self.color_button)

        self.reset_button = QPushButton('Reset Labels')
        self.reset_button.clicked.connect(self.reset_labels)
        button_layout.addWidget(self.reset_button)

        self.save_button = QPushButton('Save Labels')
        self.save_button.clicked.connect(self.save_labels)
        button_layout.addWidget(self.save_button)

        # Add mask button
        self.save_mask_button = QPushButton('Save Masks')
        self.save_mask_button.clicked.connect(self.save_masks)
        button_layout.addWidget(self.save_mask_button)

        # Class management
        self.class_colors = {}  # {(r,g,b): class_index}
        self.class_names = {}   # {class_index: name}
        self.next_class_index = 1
        self.load_class_mapping()

        self.layout.addLayout(button_layout)

        # Variables
        self.image = None
        self.image_item = None
        self.labels = []
        self.current_label = []
        self.current_color = QColor(255, 0, 0)
        self.drawing = False
        self.current_path_item = None

        self.processed_data = None  # Holds grayscale data for processing
        self.color_palette = None   # Tracks current color palette

        # Eraser mode flag and related attributes
        self.eraser_mode = False
        self.eraser_drawing = False
        self.eraser_path: Optional[QPainterPath] = None
        self.eraser_path_item: Optional[QGraphicsPathItem] = None

        # Eraser action with a shortcut (e.g. "E")
        self.eraser_action = QAction(self)
        self.eraser_action.setShortcut('E')
        self.eraser_action.triggered.connect(self.toggle_eraser_mode)
        self.addAction(self.eraser_action)

        # Optionally add an eraser button to your button layout:
        self.eraser_button = QPushButton('Eraser (E)')
        self.eraser_button.clicked.connect(self.toggle_eraser_mode)
        button_layout.addWidget(self.eraser_button)

        # Initialize undo and redo stacks
        self.undo_stack = []
        self.redo_stack = []

        # Add auto-pick mode controls
        self.auto_pick_mode = False
        self.seed_point = None
        self.auto_pick_points = []
        
        # Add auto-pick button
        self.auto_pick_button = QPushButton('Auto-Pick Mode (A)')
        self.auto_pick_button.clicked.connect(self.toggle_auto_pick_mode)
        button_layout.addWidget(self.auto_pick_button)
        
        # Add keyboard shortcut
        self.auto_pick_action = QAction(self)
        self.auto_pick_action.setShortcut('A')
        self.auto_pick_action.triggered.connect(self.toggle_auto_pick_mode)
        self.addAction(self.auto_pick_action)

        # In the button layout section
        self.flood_fill_button = QPushButton('Flood Fill (F)')
        self.flood_fill_button.clicked.connect(self.toggle_flood_fill_mode)
        button_layout.addWidget(self.flood_fill_button)

        # Keyboard shortcut
        self.flood_fill_action = QAction(self)
        self.flood_fill_action.setShortcut('F')
        self.flood_fill_action.triggered.connect(self.toggle_flood_fill_mode)
        self.addAction(self.flood_fill_action)

        # Add a flag for flood fill mode
        self.flood_fill_mode = False

    def showEvent(self, event: QShowEvent) -> None:
        """Handle window show event to fit the logo initially"""
        super().showEvent(event)
        if self.logo_item is not None and self.image is None:
            self.view.fitInView(self.logo_item, Qt.KeepAspectRatio)

    def toggle_flood_fill_mode(self):
        self.flood_fill_mode = not self.flood_fill_mode
        status = "ON" if self.flood_fill_mode else "OFF"
        self.statusBar.showMessage(f"Flood Fill mode {status}", 2000)
        # Update cursor
        if self.flood_fill_mode:
            self.view.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.view.viewport().setCursor(Qt.CrossCursor)

    def toggle_eraser_mode(self) -> None:
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.statusBar.showMessage("Eraser mode ON", 2000)
            # Change cursor to a distinctive eraser style (you may choose a custom cursor)
            self.view.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.statusBar.showMessage("Eraser mode OFF", 2000)
            self.view.viewport().setCursor(Qt.CrossCursor)


    def toggle_auto_pick_mode(self) -> None:

        if self.auto_pick_mode:
            # Cancel any unfinished auto-pick
            if self.current_path_item:
                self.scene.removeItem(self.current_path_item)
                self.current_path_item = None
            self.current_label = []
            
        self.auto_pick_mode = not self.auto_pick_mode
        status = "ON" if self.auto_pick_mode else "OFF"
        self.statusBar.showMessage(f"Auto-pick mode {status}", 2000)
        self.auto_pick_button.setChecked(self.auto_pick_mode)
        
        # Change cursor for auto-pick mode
        if self.auto_pick_mode:
            self.view.viewport().setCursor(Qt.CrossCursor)
        else:
            self.view.viewport().setCursor(Qt.ArrowCursor)

    def show_about(self):
        """Display About dialog with application information"""
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About SeisSeg")
        about_dialog.setFixedSize(500, 600)
        
        layout = QVBoxLayout(about_dialog)
        
        # Application logo
        logo_label = QLabel()
        pixmap = QPixmap('img/seisseg_icon_n.png').scaled(100, 100, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        
        # Text content
        text = QLabel()
        text.setText(
            f"<b>SeisSeg v0.1.7</b><br><br>"
            "Seismic Image Segmentation Tool<br><br>"
            "Developed by:<br>"
            "MarcFreir<br>"
            "Contact: marcfreir@outlook.com<br><br>"
            "Licence: AGPL-3.0 license<br><br>"
            "Proudly developed @ Discovery Lab | Unicamp<br><br>"
            "Citation:<br>Freire, M., & Borin, E. (2025). <br>SeisSeg - Seismic Image Segmentation (0.1.7). <br>Zenodo. https://doi.org/10.5281/zenodo.14812035<br><br>"
            "Software Heritage<br>swh:1:dir:0c30dc7c35347af0f657dee26e6e7c922b2996ea<br><br>"
            "© 2025 SeisSeg Team"

        )
        text.setAlignment(Qt.AlignCenter)

        about_dialog.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
                margin: 10px;
            }
            QPushButton {
                min-width: 80px;
                padding: 5px;
            }
        """)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(about_dialog.close)
        
        layout.addWidget(logo_label)
        layout.addWidget(text)
        layout.addWidget(close_btn)
        
        about_dialog.exec_()

    def show_help(self):
        """Display Help dialog with documentation links"""
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Help")
        help_dialog.setFixedSize(500, 200)
        
        layout = QVBoxLayout(help_dialog)
        
        # Help content
        content = QLabel()
        content.setText(
            "<h3>SeisSeg Documentation</h3>"
            "User Guide and Tutorials:<br>"
            "<a href='https://github.com/marcfreir/seisseg'>https://github.com/marcfreir/seisseg</a><br><br>"
            
            "Keyboard Shortcuts:<br>"
            "- C: Close polygon<br>"
            "- E: Toggle eraser<br>"
            "- F: Flood fill mode<br>"
            "- A: Auto-pick mode"
        )
        content.setOpenExternalLinks(True)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(help_dialog.close)
        
        layout.addWidget(content)
        layout.addWidget(close_btn)
        
        help_dialog.exec_()

    def close_current_polygon(self):
        """Close and finalize both manual and auto-pick polygons"""
        if not self.current_label or len(self.current_label) < 3:
            self.statusBar.showMessage("Need at least 3 points to close", 3000)
            return
            
        # Close the path
        self.current_label.append(self.current_label[0])
        self.current_path.lineTo(self.current_label[0].x(), self.current_label[0].y())
        self.current_path_item.setPath(self.current_path)
        
        # Create and store label
        self.fill_label()
        
        # Cleanup
        self.current_label = []
        self.current_path_item = None
        if self.auto_pick_mode:
            self.auto_pick_mode = False
            self.toggle_auto_pick_mode()


    def auto_pick_horizon(self, seed_x: int, seed_y: int):
        """Automatically track seismic horizon from seed point"""
        if self.processed_data is None or self.image is None:
            return []
        
        data = self.processed_data
        height, width = data.shape
        
        # Configure picking parameters
        search_window = 2  # pixels to search vertically
        correlation_window = 2  # pixels for similarity comparison
        max_traces = 100  # maximum traces to pick in each direction

        # Initialize tracking
        horizon = []
        directions = [1, -1]  # Right then left
        
        for direction in directions:
            current_x = seed_x
            current_y = seed_y
            last_valid = (current_x, current_y)
            
            for _ in range(max_traces):
                next_x = current_x + direction
                if next_x < 0 or next_x >= width:
                    break
                
                # Define search range
                y_min = max(0, current_y - search_window)
                y_max = min(height-1, current_y + search_window)
                best_y = current_y
                best_similarity = -np.inf
                
                for y in range(y_min, y_max+1):
                    # Define correlation windows
                    ref_start = max(0, current_y - correlation_window)
                    ref_end = min(height, current_y + correlation_window + 1)
                    target_start = max(0, y - correlation_window)
                    target_end = min(height, y + correlation_window + 1)
                    
                    # Extract reference and target traces
                    ref = data[ref_start:ref_end, current_x]
                    target = data[target_start:target_end, next_x]
                    
                    # Skip if window sizes don't match
                    if ref.size != target.size:
                        continue
                    
                    # Calculate similarity manually to avoid division by zero
                    ref_mean = np.mean(ref)
                    target_mean = np.mean(target)
                    covariance = np.mean((ref - ref_mean) * (target - target_mean))
                    std_ref = np.std(ref)
                    std_target = np.std(target)
                    
                    if std_ref > 0 and std_target > 0:
                        similarity = covariance / (std_ref * std_target)
                    else:
                        similarity = 0.0  # Handle zero-variance cases
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_y = y
                
                # Validate and update position
                if best_similarity > 0.5:  # Similarity threshold
                    current_x = next_x
                    current_y = best_y
                    last_valid = (current_x, current_y)
                    horizon.append((current_x, current_y))
                else:
                    # Use last valid position and stop
                    horizon.append(last_valid)
                    break
        
        # Sort points by X coordinate
        horizon.sort(key=lambda p: p[0])
        return horizon

    def open_image(self) -> None:

        """
        Open an image file, and display it in the view.

        This method uses a QFileDialog to prompt the user for an image file to open. If the file is a TIFF, it uses tifffile to read the image; otherwise, it uses PIL to read the image. The image is then displayed in the view and stored as self.image. The original image is also stored as self.original_image.

        :return: None
        """

        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)'
        )

        if not file_path:
            return

        self.image_path = file_path

        # Clear the scene (this removes the logo)
        self.scene.clear()
        self.current_label = []
        self.labels = []

        # Clear undo/redo stacks when opening a new image
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.label_counter = 0

        # Handle TIFF separately
        if file_path.lower().endswith(('.tif', '.tiff')):
            try:
                tiff_img = tifffile.imread(file_path)

                # Ensure image is in a displayable format (uint8)
                if tiff_img.dtype not in [np.uint8, np.uint16]:
                    tiff_img = (255 * (tiff_img - np.min(tiff_img)) / (np.max(tiff_img) - np.min(tiff_img))).astype(np.uint8)

                # Handle different TIFF shapes
                if len(tiff_img.shape) == 2:  # Grayscale
                    self.image = Image.fromarray(tiff_img, mode='L')
                elif len(tiff_img.shape) == 3:
                    if tiff_img.shape[0] in [3, 4]:  # Channels-first (C, H, W) → Convert to (H, W, C)
                        tiff_img = np.transpose(tiff_img, (1, 2, 0))
                    
                    if tiff_img.shape[2] == 3:  # RGB
                        self.image = Image.fromarray(tiff_img, mode='RGB')
                    elif tiff_img.shape[2] == 4:  # RGBA
                        self.image = Image.fromarray(tiff_img, mode='RGBA')
                    else:
                        self.statusBar.showMessage(f"Unsupported TIFF format: {tiff_img.shape}", 5000)
                        return
                else:
                    self.statusBar.showMessage(f"Unknown TIFF shape: {tiff_img.shape}", 5000)
                    return

            except Exception as e:
                self.statusBar.showMessage(f"Error opening TIFF: {str(e)}", 5000)
                return

        else:
            # Handle all other image formats using PIL
            try:
                self.image = Image.open(file_path).convert('RGB')
            except Exception as e:
                self.statusBar.showMessage(f"Error opening image: {str(e)}", 5000)
                return

        # Display the image
        self.scene.clear()
        self.current_label = []
        self.labels = []

        qimage_format = QImage.Format_Grayscale8 if self.image.mode == 'L' else QImage.Format_RGB888
        qimage = QImage(self.image.tobytes(), self.image.width, self.image.height,
                        self.image.width * (1 if self.image.mode == 'L' else 3), qimage_format)
        pixmap = QPixmap.fromImage(qimage)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        self.scene.setSceneRect(0, 0, self.image.width, self.image.height)
        self.view.fitInView(self.image_item, Qt.KeepAspectRatio)
        self.statusBar.showMessage(f'Image opened: {file_path}')

        # Store original and processed data
        self.original_image = self.image.copy()
        img_gray = self.original_image.convert('L') if self.original_image.mode != 'L' else self.original_image.copy()
        self.processed_data = np.array(img_gray)
        self.color_palette = None
        self._update_display()


    def apply_gabor(self) -> None:
        """Apply Gabor filter to current image.
        
        Prompts user for:
        - Frequency (0.01-1.0)
        - Theta (0-π radians)
        
        Updates display with filtered result.
        """

        if self.processed_data is not None:
            freq, ok1 = QInputDialog.getDouble(self, 'Gabor Filter', 'Frequency:', 0.1, 0.01, 1.0, 2)
            theta, ok2 = QInputDialog.getDouble(self, 'Gabor Filter', 'Theta (0-π):', 0, 0, np.pi, 2)

            if ok1 and ok2:
                # Convert to float representation
                image_array = img_as_float(self.processed_data)
                filt_real, filt_imag = gabor(image_array, frequency=freq, theta=theta)
                filtered = np.sqrt(filt_real**2 + filt_imag**2)
                
                # # Normalize to [0, 1]
                # filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
                # self.processed_data = (filtered * 255).astype(np.uint8)
                # self._update_display()
                # Convert to uint8 after processing
                filtered = (filtered * 255).astype(np.uint8)
                self.processed_data = filtered
                self._update_display()


    def apply_lbp(self) -> None:
        """Apply Local Binary Pattern (LBP) texture analysis.
        
        Parameters
        ----------
        radius : int (1-5)
            Neighborhood radius
        n_points : int (4-24)
            Number of circular points
            
        Updates display with LBP result.
        """

        if self.processed_data is not None:
            radius, ok1 = QInputDialog.getInt(self, 'LBP', 'Radius:', 1, 1, 5)
            n_points, ok2 = QInputDialog.getInt(self, 'LBP', 'Points:', 8, 4, 24)
            
            if ok1 and ok2:
                # Convert to integer type first
                if np.issubdtype(self.processed_data.dtype, np.floating):
                    # Scale to 0-255 and convert to uint8
                    lbp_input = (self.processed_data * 255).astype(np.uint8)
                else:
                    lbp_input = self.processed_data.astype(np.uint8)
                
                # Calculate LBP on integer image
                lbp = local_binary_pattern(lbp_input, n_points, radius, method='uniform')
                
                # Normalize for display
                lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())  # Now in [0, 1]
                self.processed_data = (lbp * 255).astype(np.uint8)
                self._update_display()


    def apply_glcm(self) -> None:
        """
        Apply Gray-Level Co-Occurrence Matrix (GLCM) texture analysis.
        
        Calculates contrast and homogeneity properties from the GLCM and
        displays them in the status bar.
        """

        if self.processed_data is not None:
            image_array = img_as_ubyte(self.processed_data)  # Convert to uint8
            
            # Calculate GLCM
            glcm = graycomatrix(image_array, 
                            distances=[1], 
                            angles=[0], 
                            levels=256,
                            symmetric=True, 
                            normed=True)
            
            # Calculate properties
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            # Show properties without modifying image data
            self.statusBar.showMessage(
                f"GLCM - Contrast: {contrast:.2f}, Homogeneity: {homogeneity:.2f}", 
                5000
            )

    def apply_got(self) -> None:
        """
        Apply Gradient of Texture (GoT) feature extraction.

        Calculates the gradient of the image using the Sobel operator.
        The gradient is then used to update the display.

        """

        if self.processed_data is not None:

            image_array = self.processed_data
            
            # Calculate gradient using Sobel
            gradient = sobel(image_array)
            # self._update_display(gradient)
            self.processed_data = gradient

            self._update_display()


    def apply_hog(self) -> None:
        if self.processed_data is not None:
            image_array = img_as_ubyte(self.processed_data)  # Convert to uint8
            
            # Add a small noise to uniform regions to avoid division by zero
            if np.var(image_array) == 0:
                image_array = image_array + np.random.uniform(-1, 1, size=image_array.shape).astype(np.uint8)
            
            # Calculate HOG
            try:
                _, hog_image = hog(
                    image_array,
                    orientations=8,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1),
                    visualize=True,
                    channel_axis=None
                )
                
                # Normalize the HOG image
                hog_min = hog_image.min()
                hog_max = hog_image.max()
                if hog_max > hog_min:  # Avoid division by zero
                    hog_image = (hog_image - hog_min) / (hog_max - hog_min)
                else:
                    hog_image = np.zeros_like(hog_image)  # Handle uniform images
                
                self.processed_data = hog_image.astype(np.float32)
                self._update_display()
            
            except Exception as e:
                self.statusBar.showMessage(f"Error applying HOG: {str(e)}", 5000)

    # New attributes - seismic attributes
    def apply_semblance(self) -> None:
        """Calculate seismic coherence/semblance attribute"""
        if self.processed_data is None: return
        
        # Get window size from user
        w, ok = QInputDialog.getInt(self, 'Semblance', 'Window Size (odd):', 3, 3, 15, 2)
        if not ok or w%2 == 0: return
        
        data = self.processed_data.astype(np.float32)
        w2 = w//2
        semblance = np.zeros_like(data)
        
        # Pad array for border processing
        padded = np.pad(data, w2, mode='reflect')
        
        # Calculate semblance in sliding window
        for i in range(w2, data.shape[0]+w2):
            for j in range(w2, data.shape[1]+w2):
                window = padded[i-w2:i+w2+1, j-w2:j+w2+1]
                mean = window.mean()
                semblance[i-w2,j-w2] = window.var() / (mean**2 + 1e-6) if mean != 0 else 0
                
        self.processed_data = semblance
        self._update_display()

    def apply_amplitude_envelope(self) -> None:
        """Calculate Hilbert envelope"""
        if self.processed_data is None: return
        
        analytic = sig.hilbert(self.processed_data)
        envelope = np.abs(analytic)
        self.processed_data = envelope
        self._update_display()

    def apply_butterworth(self) -> None:
        """Bandpass Butterworth filter"""
        if self.processed_data is None: return
        
        # Get filter parameters
        low, ok1 = QInputDialog.getDouble(self, 'Butterworth', 'Low cutoff (0-0.5):', 0.1, 0, 0.5)
        high, ok2 = QInputDialog.getDouble(self, 'Butterworth', 'High cutoff (0-0.5):', 0.4, 0, 0.5)
        order, ok3 = QInputDialog.getInt(self, 'Butterworth', 'Filter order:', 4, 1, 8)
        
        if not (ok1 and ok2 and ok3) or low >= high: return
        
        # Design filter
        b, a = sig.butter(order, [low, high], btype='band')
        filtered = sig.filtfilt(b, a, self.processed_data)
        self.processed_data = filtered
        self._update_display()

    def apply_multi_otsu(self) -> None:
        """Multi-level Otsu thresholding"""
        if self.processed_data is None: return
        
        classes, ok = QInputDialog.getInt(self, 'Otsu', 'Number of classes:', 3, 2, 5)
        if not ok: return
        
        thresholds = threshold_multiotsu(self.processed_data, classes)
        regions = np.digitize(self.processed_data, thresholds)
        self.processed_data = regions/classes  # Normalize for display
        self._update_display()


    def apply_spectral_decomposition(self) -> None:
        """Spectral decomposition using STFT with proper dimension handling"""
        if self.processed_data is None: 
            return
            
        # Get parameters
        freq, ok = QInputDialog.getDouble(
            self, 'Spectral Decomp', 
            'Target Frequency (Hz):', 30, 1, 100
        )
        if not ok: return

        # STFT parameters
        nperseg = 64
        noverlap = 32  # 50% overlap for better time resolution
        fs = 1.0       # Normalized frequency

        # Process each row individually
        height, width = self.processed_data.shape
        spec_mag = np.zeros_like(self.processed_data, dtype=np.float32)

        for y in range(height):
            # Compute STFT for each row
            f, t, Zxx = sig.stft(
                self.processed_data[y], 
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                boundary=None
            )
            
            # Find nearest frequency bin
            freq_idx = np.argmin(np.abs(f - freq))
            
            # Extract magnitude and interpolate to original width
            mag = np.abs(Zxx[freq_idx])
            spec_mag[y] = np.interp(
                np.linspace(0, 1, width),
                np.linspace(0, 1, len(mag)),
                mag
            )

        # Normalize and update
        self.processed_data = (spec_mag - spec_mag.min()) / (spec_mag.max() - spec_mag.min())
        self._update_display()

    def apply_structural_smoothing(self) -> None:
        """Edge-preserving smoothing"""
        if self.processed_data is None: return
        
        sigma, ok = QInputDialog.getDouble(self, 'Smoothing', 'Sigma:', 1.0, 0.1, 5.0)
        if not ok: return
        
        smoothed = gaussian_filter(self.processed_data, sigma=sigma)
        self.processed_data = smoothed
        self._update_display()

    def apply_instantaneous_frequency(self) -> None:
        """Calculate instantaneous frequency with user parameters"""
        
        if self.processed_data is None:
            self.statusBar.showMessage("No image data available!", 5000)
            return

        # Get parameters from user
        axis, ok1 = QInputDialog.getInt(
            self, 'Frequency Axis', 
            'Time axis (0=vertical, 1=horizontal):', 0, 0, 1
        )
        low_perc, ok2 = QInputDialog.getDouble(
            self, 'Normalization Range', 
            'Lower percentile:', 2.0, 0.0, 100.0
        )
        high_perc, ok3 = QInputDialog.getDouble(
            self, 'Normalization Range',
            'Upper percentile:', 98.0, 0.0, 100.0
        )
        
        if not all([ok1, ok2, ok3]) or low_perc >= high_perc:
            self.statusBar.showMessage("Invalid parameters", 5000)
            return

        try:
            data = self.processed_data.astype(np.float32)
            analytic_signal = sig.hilbert(data, axis=axis)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal), axis=axis)
            instantaneous_frequency = np.gradient(instantaneous_phase, axis=axis) / (2 * np.pi)
            
            # User-controlled normalization
            vmin, vmax = np.percentile(instantaneous_frequency, [low_perc, high_perc])
            instantaneous_frequency = np.clip(instantaneous_frequency, vmin, vmax)
            instantaneous_frequency = (instantaneous_frequency - vmin) / (vmax - vmin + 1e-8)
            
            self.processed_data = instantaneous_frequency
            self._update_display()
            self.statusBar.showMessage(
                f"Instantaneous frequency (axis {axis}, {low_perc}-{high_perc}%)", 
                3000
            )
            
        except Exception as e:
            self.statusBar.showMessage(f"Error: {str(e)}", 5000)


    def apply_curvature(self) -> None:
        """Structural curvature with stability parameter"""
        if self.processed_data is None: 
            return
        
        # Get stability factor
        epsilon, ok = QInputDialog.getDouble(
            self, 'Curvature Stability',
            'Stabilization factor (1e-6 recommended):',
            1e-6, 1e-12, 1e-3, 10
        )
        if not ok:
            return

        try:
            dy, dx = np.gradient(self.processed_data)
            dyy, dxy = np.gradient(dy)
            _, dxx = np.gradient(dx)
            
            denominator = (dx**2 + dy**2 + epsilon)**1.5
            curvature = (dxx*dy**2 - 2*dxy*dx*dy + dyy*dx**2) / denominator
            
            self.processed_data = curvature
            self._update_display()
            self.statusBar.showMessage(
                f"Curvature calculated (ε={epsilon:.1e})", 
                3000
            )
            
        except Exception as e:
            self.statusBar.showMessage(f"Curvature error: {str(e)}", 5000)

    def apply_dip_steering(self) -> None:
        """Structure-oriented filtering using dip steering"""
        if self.processed_data is None: 
            return
        
        try:
            data = self.processed_data.astype(np.float32)
            
            # Calculate gradient components
            grad_y, grad_x = np.gradient(data)
            
            # Calculate dip orientation (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                dip = np.arctan2(grad_y, grad_x + 1e-8)  # Add small epsilon to prevent division by zero
            
            # Get kernel size from user
            kernel_size, ok = QInputDialog.getInt(
                self, 'Dip Steering', 'Kernel Size (odd):', 5, 3, 15, 2
            )
            if not ok or kernel_size % 2 == 0:
                return
            
            half_size = kernel_size // 2
            
            # Create coordinate grid for kernel
            y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
            
            # Initialize output
            filtered = np.zeros_like(data)
            
            # Pad input data
            padded = np.pad(data, half_size, mode='reflect')
            
            # Vectorized filtering
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    # Get local dip angle
                    theta = dip[i, j]
                    
                    # Rotate coordinates
                    x_rot = x * np.cos(theta) + y * np.sin(theta)
                    y_rot = -x * np.sin(theta) + y * np.cos(theta)
                    
                    # Create Gaussian kernel oriented along dip
                    kernel = np.exp(-0.5 * (x_rot**2 + y_rot**2))
                    kernel /= kernel.sum()
                    
                    # Apply convolution
                    filtered[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
            
            self.processed_data = filtered
            self._update_display()
            self.statusBar.showMessage("Dip-steered filtering applied", 3000)
            
        except Exception as e:
            self.statusBar.showMessage(f"Error in dip steering: {str(e)}", 5000)

    def reset_to_original(self) -> None:
        """
        Reset the processed image data to the original grayscale image.

        If the original image data is available, it is converted to grayscale
        (if not already) and stored as the new processed data. The color palette
        is also reset to None. The display is then updated with the new data.
        """

        if self.original_image:
            # Reset to original grayscale data
            if self.original_image.mode != 'L':
                img_gray = self.original_image.convert('L')
            else:
                img_gray = self.original_image.copy()
            self.processed_data = np.array(img_gray)
            self.color_palette = None
            self._update_display()

    
    def apply_color_palette(self, palette_name: str) -> None:

        """
        Apply a color palette to the currently displayed image.

        Applies the specified color palette to the internal processed_data and
        then updates the display. The color palette should be a string referring
        to a valid matplotlib color map name.

        :param palette_name: Name of the color palette to apply
        :type palette_name: str
        """

        if self.processed_data is not None:
            self.color_palette = palette_name
            self._update_display()

    def _update_display(self) -> None:
        
        """
        Internal method to update the display with the currently processed data.

        If the processed data is valid, it is normalized to the range 0-1 and then
        either converted to grayscale or colored using the specified color palette.
        The resulting image is then displayed in the main window.

        If any error occurs, an error message is shown in the status bar.
        """

        if self.processed_data is None:
            return

        try:
            # Ensure we're working with a 2D array
            if self.processed_data.ndim != 2:
                raise ValueError("Processed data must be 2-dimensional")

            # Normalize to 0-1
            min_val = self.processed_data.min()
            max_val = self.processed_data.max()
            normalized = (self.processed_data - min_val) / (max_val - min_val + 1e-8)

            if self.color_palette:
                # Apply colormap
                cmap = plt.get_cmap(self.color_palette)
                colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
                processed_image = Image.fromarray(colored, 'RGB')
            else:
                # Convert to grayscale
                processed_image = Image.fromarray((normalized * 255).astype(np.uint8)).convert('RGB')

            # Update display
            qimage = QImage(
                processed_image.tobytes(), 
                processed_image.width, 
                processed_image.height, 
                processed_image.width * 3, 
                QImage.Format_RGB888
            )
            self.image_item.setPixmap(QPixmap.fromImage(qimage))
            
        except Exception as e:
            self.statusBar.showMessage(f"Display error: {str(e)}", 5000)

    def choose_color(self) -> None:
        """
        Opens a color dialog to select a new color for the current class.

        After selecting a color, the corresponding class index is determined and
        the mapping is saved to file. If the color is new, a new class index is
        assigned and the mapping is updated accordingly.

        :return: None
        """

        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color

            # Register new color if needed
            rgb = (color.red(), color.green(), color.blue())
            if rgb not in self.class_colors:
                self.class_colors[rgb] = self.next_class_index
                self.class_names[self.next_class_index] = f'Class {self.next_class_index}'
                self.next_class_index += 1
                self.save_class_mapping()

    def save_class_mapping(self) -> None:
        """
        Saves the current class-to-color mapping to a JSON file.

        The mapping is saved as a dictionary with RGB tuples as keys and
        dictionaries containing the class index and name as values.

        :return: None
        """

        mapping = {
            f'{r},{g},{b}': {
                'index': idx,
                'name': self.class_names[idx]
            }
            for (r,g,b), idx in self.class_colors.items()
        }
        with open('class_mapping.json', 'w') as f:
            json.dump(mapping, f, indent=2)

    def load_class_mapping(self) -> None:
        """
        Loads a class-to-color mapping from a JSON file.

        The mapping is expected to be a dictionary with RGB tuples as keys and
        dictionaries containing the class index and name as values.

        If the file does not exist, the method does nothing.

        :return: None
        """

        try:
            with open('class_mapping.json', 'r') as f:
                mapping = json.load(f)
                for rgb_str, data in mapping.items():
                    r, g, b = map(int, rgb_str.split(','))
                    idx = data['index']
                    self.class_colors[(r,g,b)] = idx
                    self.class_names[idx] = data['name']
                if self.class_colors:
                    self.next_class_index = max(self.class_names.keys()) + 1
        except FileNotFoundError:
            pass

    def save_masks(self) -> None:
        """
        Saves the labeled overlay as a grayscale mask image.

        The mask is a grayscale image where each pixel's value corresponds to the class index of the label.
        If a color is not registered, the corresponding pixels are set to 0 (black).

        The user is prompted to select a file path, and the mask is saved to that path as a PNG image.
        If the selected path does not end with '.png', it is appended.

        :return: None
        """

        if not self.image or (not self.labels and not self.undo_stack):
            return
        
        # Create grayscale mask
        mask = Image.new('L', self.image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw all labels
        for polygon, color in self.labels:
            rgb = (color.red(), color.green(), color.blue())
            class_idx = self.class_colors.get(rgb, 0)
            if class_idx == 0:
                continue  # Skip unregistered colors
                
            points = [(p.x(), p.y()) for p in polygon]
            draw.polygon(points, fill=class_idx)

        # Draw all paths from undo stack (including unclosed traces)
        for entry in self.undo_stack:
            if entry.get('type') == 'draw' and entry.get('path'):
                path = entry['path'].path()
                color = entry['label'][1]
                rgb = (color.red(), color.green(), color.blue())
                class_idx = self.class_colors.get(rgb, 0)
                if class_idx == 0:
                    continue
                
                # Convert path to line segments
                points = []
                for i in range(path.elementCount()):
                    elem = path.elementAt(i)
                    points.append((elem.x, elem.y))
                
                # Draw lines with original pen width
                pen_width = max(1, int(entry['path'].pen().width()))
                if len(points) >= 2:
                    for i in range(len(points)-1):
                        draw.line([points[i], points[i+1]], fill=class_idx, width=pen_width)
                    
                    # Close the path if it's a polygon
                    if points[0] == points[-1] and len(points) >= 3:
                        draw.polygon(points, fill=class_idx)

        # Add flood fill processing
        for entry in self.undo_stack:
            if entry.get('type') == 'flood_fill':
                fill_mask = entry.get('mask')
                color = entry.get('color')
                if not fill_mask or not color:
                    continue
                    
                rgb = (color.red(), color.green(), color.blue())
                class_idx = self.class_colors.get(rgb, 0)
                if class_idx == 0:
                    continue
                    
                # Apply the stored mask
                mask.paste(class_idx, mask=fill_mask)
        
        # Save mask
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        default_path = os.path.join(os.path.dirname(self.image_path), f'{base_name}_mask.png')
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Mask', default_path, 'PNG Files (*.png)'
        )
        
        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            mask.save(file_path)
            self.statusBar.showMessage(f'Mask saved to {file_path}', 3000)


    def handle_left_press(self, scene_pos: QPoint) -> None:

        if self.flood_fill_mode and self.image:
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                # Use the boundary-aware flood fill:
                self.perform_flood_fill(x, y)
            return

        # If eraser mode is active, start an eraser stroke instead of a new label.
        if self.eraser_mode:
            self.start_eraser(scene_pos)
            return
    
        if self.auto_pick_mode and self.image:
            # Auto-pick mode handling
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                # Store seed point and run auto-picker
                self.seed_point = (x, y)
                self.auto_pick_points = self.auto_pick_horizon(x, y)
                
                # Create path from auto-picked points
                self.current_label = [QPoint(p[0], p[1]) for p in self.auto_pick_points]
                self.current_path = QPainterPath()
                if self.current_label:
                    self.current_path.moveTo(self.current_label[0].x(), self.current_label[0].y())
                    for point in self.current_label[1:]:
                        self.current_path.lineTo(point.x(), point.y())
                    
                    self.current_path_item = QGraphicsPathItem(self.current_path)
                    self.current_path_item.setPen(QPen(self.current_color, 2))
                    self.scene.addItem(self.current_path_item)
                    self.drawing = True  # Enable drawing adjustments
                return

        # Original manual drawing code
        if self.image and not self.auto_pick_mode:
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                self.drawing = True
                self.current_label = [QPoint(x, y)]
                self.current_path = QPainterPath()
                self.current_path.moveTo(x, y)
                self.current_path_item = QGraphicsPathItem(self.current_path)
                self.current_path_item.setPen(QPen(self.current_color, 2))
                self.scene.addItem(self.current_path_item)


    def handle_left_move(self, scene_pos: QPoint) -> None:

        if self.eraser_mode and self.eraser_drawing:
            self.update_eraser(scene_pos)
            return
    
        if self.auto_pick_mode and self.drawing and self.current_path_item:
            # Adjust auto-picked path
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                self.current_path.lineTo(x, y)
                self.current_path_item.setPath(self.current_path)
                self.current_label.append(QPoint(x, y))
        elif self.drawing and self.image:  # Original manual drawing code
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                self.current_label.append(QPoint(x, y))
                self.current_path.lineTo(x, y)
                self.current_path_item.setPath(self.current_path)


    def handle_left_release(self, scene_pos: QPoint) -> None:
        if self.eraser_mode and self.eraser_drawing:
            self.finish_eraser()
            return

        if self.drawing:
            self.drawing = False
            self.statusBar.showMessage("Drawing stopped", 3000)

            # Always add to undo stack, even if not closed
            self.undo_stack.append({
                'type': 'draw',
                'label': (self.current_label.copy(), self.current_color),
                'overlay': None,
                'path': self.current_path_item
            })
            self.redo_stack.clear()

    def start_eraser(self, scene_pos: QPoint) -> None:
        self.eraser_drawing = True
        self.eraser_path = QPainterPath()
        self.eraser_path.moveTo(scene_pos)
        self.eraser_path_item = QGraphicsPathItem(self.eraser_path)
        # Set a thick pen to simulate an eraser brush (adjust width and color as needed)
        eraser_pen = QPen(Qt.white, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.eraser_path_item.setPen(eraser_pen)
        self.scene.addItem(self.eraser_path_item)

    def update_eraser(self, scene_pos: QPoint) -> None:
        if self.eraser_drawing and self.eraser_path is not None:
            self.eraser_path.lineTo(scene_pos)
            self.eraser_path_item.setPath(self.eraser_path)


    def finish_eraser(self) -> None:
        if not self.eraser_drawing or self.eraser_path is None:
            return
        
        # Convert eraser path to PIL mask
        eraser_mask = Image.new('L', self.image.size, 0)
        draw = ImageDraw.Draw(eraser_mask)
        path_points = [(self.eraser_path.elementAt(i).x, self.eraser_path.elementAt(i).y)
                    for i in range(self.eraser_path.elementCount())]
        
        if len(path_points) >= 2:
            for i in range(len(path_points)-1):
                # Draw thicker lines to account for anti-aliasing (width)
                draw.line([path_points[i], path_points[i+1]], fill=255, width=5)

        # Process all undo stack entries
        modified_entries = []

        # For each label in the undo stack, try to subtract the eraser stroke.
        for entry in self.undo_stack:
            # Handle flood fill erasure
            if entry.get('type') == 'flood_fill':
                original_mask = entry['mask']

                # Convert eraser mask to binary (if not already)
                eraser_bin = eraser_mask.convert("1")
                # Invert the eraser mask so that erased regions become 0
                inverted_eraser = ImageChops.invert(eraser_bin.convert("L"))
                # Update the original flood fill mask by keeping only areas not erased.
                # (We use a logical AND between the original mask and the inverted eraser mask.)
                
                # # Create inverse of eraser mask
                # inverted_eraser = Image.eval(eraser_mask, lambda x: 255 - x)
                
                # Subtract eraser area using logical AND
                updated_mask = ImageChops.logical_and(
                    original_mask.convert('1'), 
                    inverted_eraser.convert('1')
                ).convert('L')

                # Only update if there is a change
                if list(updated_mask.getdata()) != list(original_mask.getdata()):
                    entry['mask'] = updated_mask
                    # Recreate the overlay using the stored color
                    color_layer = Image.new("RGBA", self.image.size, entry['color'].name())
                    new_overlay_img = Image.composite(color_layer, Image.new("RGBA", self.image.size), updated_mask)
                    qimage = self.pil2qimage(new_overlay_img)
                    new_pixmap = QPixmap.fromImage(qimage)
                    # Replace the overlay item in the scene
                    if entry.get('overlay') and entry['overlay'].scene() == self.scene:
                        self.scene.removeItem(entry['overlay'])
                    entry['overlay'] = QGraphicsPixmapItem(new_pixmap)
                    self.scene.addItem(entry['overlay'])

            label_path_item = entry.get('path')
            if label_path_item is None:
                modified_entries.append(entry)
                continue
            label_path = label_path_item.path()
            # If the eraser stroke intersects this label, subtract the eraser area.
            if label_path.intersects(self.eraser_path):
                new_path = label_path.subtracted(self.eraser_path)
                label_path_item.setPath(new_path)
                # Optionally update the overlay if the label is closed.
                if entry.get('overlay') is not None:
                    # Only remove the overlay if it is in the current scene.
                    if entry.get('overlay').scene() == self.scene:
                        self.scene.removeItem(entry.get('overlay'))
                    # Recompute the overlay from the modified path if possible.
                    if new_path.elementCount() > 2:
                        polygon = []
                        for i in range(new_path.elementCount()):
                            el = new_path.elementAt(i)
                            polygon.append((el.x, el.y))

                        if polygon and polygon[0] == polygon[-1]:
                            mask_img = Image.new('L', self.image.size, 0)
                            draw_polygon = ImageDraw.Draw(mask_img)
                            draw_polygon.polygon(polygon, fill=255)
                            color_layer = Image.new('RGBA', self.image.size, entry['label'][1].name())
                            overlay_img = Image.alpha_composite(self.image.convert('RGBA'),
                                                                color_layer.convert('RGBA'))
                            overlay_img.putalpha(mask_img)
                            qimage = QImage(overlay_img.tobytes(), overlay_img.width, overlay_img.height,
                                            overlay_img.width * 4, QImage.Format_RGBA8888)
                            new_overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
                            self.scene.addItem(new_overlay_item)
                            entry['overlay'] = new_overlay_item
            modified_entries.append(entry)
        self.undo_stack = modified_entries

        # Remove the eraser stroke item if it is in the scene.
        if self.eraser_path_item and self.eraser_path_item.scene() == self.scene:
            self.scene.removeItem(self.eraser_path_item)
        self.eraser_path_item = None
        self.eraser_drawing = False
        self.eraser_path = None
        self.statusBar.showMessage("Erasing applied", 2000)

    def fill_label(self) -> None:
        """
        Fill the current label polygon with the selected color and update the display.

        This method creates a mask for the current label polygon and fills it with the
        selected color on the overlay. It converts the overlay to a QImage and adds it
        to the scene. The label is logged and a status message is shown indicating the
        nth label created.

        Preconditions:
        - `self.image` must be set to a valid image.
        - `self.current_label` must contain the points of the polygon.

        Postconditions:
        - The filled polygon is added to the scene.
        - The label's coordinates are logged.
        - The label counter is incremented and a status message is displayed.

        :return: None
        """

        if len(self.current_label) < 2:  # Changed from 3 to 2 to allow simple lines
            return
        
        # Automatically close if first and last points are close
        if (self.current_label[0] - self.current_label[-1]).manhattanLength() < 5:
            self.current_label.append(self.current_label[0])
      
        if self.image and self.current_label:
            polygon = [(p.x(), p.y()) for p in self.current_label]
            mask = Image.new('L', self.image.size, 0)
            draw = ImageDraw.Draw(mask)

            # Handle both polygons and lines
            if len(polygon) >= 3 and polygon[0] == polygon[-1]:
                draw.polygon(polygon, fill=255)
            else:
                pen_width = self.current_path_item.pen().width()
                if len(polygon) >= 2:
                    for i in range(len(polygon)-1):
                        draw.line([polygon[i], polygon[i+1]], fill=255, width=pen_width)

            color_layer = Image.new('RGBA', self.image.size, self.current_color.name())
            overlay = Image.alpha_composite(self.image.convert('RGBA'), color_layer.convert('RGBA'))
            overlay.putalpha(mask)

            qimage = QImage(overlay.tobytes(), overlay.width, overlay.height, 
                            overlay.width * 4, QImage.Format_RGBA8888)
            overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
            self.scene.addItem(overlay_item)
            self.labels.append((self.current_label, self.current_color))

            # Update undo stack
            self.undo_stack.append({
                'type': 'draw',
                'label': (self.current_label.copy(), self.current_color),
                'overlay': overlay_item,
                'path': self.current_path_item
            })
            self.redo_stack.clear()  # Clear redo stack on new action

    def reset_labels(self) -> None:
        """Reset only labels while preserving image processing state"""
        try:
            # Remove all label-related graphics items from the scene
            for entry in self.undo_stack + self.redo_stack:
                overlay = entry.get('overlay')
                if overlay is not None and overlay.scene() == self.scene:
                    self.scene.removeItem(overlay)
                path = entry.get('path')
                if path is not None and path.scene() == self.scene:
                    self.scene.removeItem(path)

            # Clear current drawing if in progress
            if self.current_path_item and self.current_path_item.scene() == self.scene:
                self.scene.removeItem(self.current_path_item)
                
            # Reset label tracking structures
            self.labels.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.label_counter = 0
            self.current_label = []
            self.current_path_item = None

            # Preserve current image state
            if self.processed_data is not None:
                self._update_display()  # Refresh display without changing processed data

            self.statusBar.showMessage("Labels reset - processing preserved", 3000)

        except Exception as e:
            self.statusBar.showMessage(f"Reset error: {str(e)}", 5000)


    def pil2qimage(self, im):
        """Convert PIL Image to QImage."""
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
            data = im.tobytes()
            qimage = QImage(data, im.width, im.height, QImage.Format_RGB888)
        elif im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
            data = im.tobytes()
            qimage = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
        else:
            im = im.convert("RGBA")
            data = im.tobytes()
            qimage = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
        return qimage


    def perform_flood_fill(self, x: int, y: int):
        # Create boundary mask with existing paths and image borders
        boundary = Image.new("L", self.image.size, 255)  # White = fillable
        draw = ImageDraw.Draw(boundary)
        
        # Draw image borders as boundaries (black)
        draw.rectangle([(0, 0), self.image.size], outline=0, width=2)
        
        # Draw all existing paths
        for entry in self.undo_stack:
            if 'path' in entry:
                path = entry['path'].path()
                pen_width = max(1, int(entry['path'].pen().width()))
                points = [(path.elementAt(i).x, path.elementAt(i).y) 
                        for i in range(path.elementCount())]
                
                # Draw lines between points
                if len(points) >= 2:
                    for i in range(len(points)-1):
                        draw.line([points[i], points[i+1]], fill=0, width=pen_width)
                
                # Fill closed polygons
                if len(points) >=3 and points[0] == points[-1]:
                    draw.polygon(points, fill=0)

        # Check boundary conditions
        if boundary.getpixel((x, y)) == 0:
            self.statusBar.showMessage("Can't fill on boundary!", 2000)
            return

        # Perform flood fill with boundary constraints
        fill_mask = boundary.copy()
        ImageDraw.floodfill(fill_mask, (x, y), 128, border=0)
        
        # Create overlay from filled region
        fill_area = fill_mask.point(lambda p: 255 if p == 128 else 0)
        color_layer = Image.new("RGBA", self.image.size, self.current_color.name())
        overlay = Image.composite(color_layer, Image.new("RGBA", self.image.size), fill_area)

        # Add to scene and update undo stack
        qimage = self.pil2qimage(overlay)
        overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self.scene.addItem(overlay_item)
        
        self.undo_stack.append({
            'type': 'flood_fill',
            'overlay': overlay_item,
            'mask': fill_area,
            'color': self.current_color  # Store the color used
        })
        self.redo_stack.clear()


    def save_label_log(self) -> None:
        """
        Save label points to a log file.

        This method appends the current label's points to a log file. The log file is
        created in a directory with the same name as the image, suffixed with '_logs'.
        The label points are appended to the log file with the label index as the first
        line, and each point is written on a new line in the format '(x, y)'.

        Preconditions:
        - `self.image` must be set to a valid image.
        - `self.labels` must contain the points of the label.

        Postconditions:
        - The label's coordinates are logged.
        - The label counter is incremented and a status message is displayed.

        :return: None
        """

        if not self.image or not self.labels:
            return
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        log_dir = f'{image_name}_logs'
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, f'{image_name}_log.txt')
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f'\nLabel_{self.label_counter}:\n')
            for point in self.current_label:
                log_file.write(f'({point.x()}, {point.y()})\n')
        self.statusBar.showMessage(f'Label {self.label_counter} saved!', 3000)


    def save_labels(self) -> None:
        if not self.image:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Labels', '', 'PNG Files (*.png)'
        )
        if not file_path:
            return

        if not file_path.lower().endswith('.png'):
            file_path += '.png'

        # Create white background image
        white_bg = Image.new('RGBA', self.image.size, (255, 255, 255, 255))
        
        # Draw all annotation elements
        self._draw_all_annotations(white_bg)
        
        # Save final image
        white_bg.convert('RGB').save(file_path)
        self.statusBar.showMessage(f'Labels saved as {file_path}', 3000)


    def _draw_all_annotations(self, bg_image: Image.Image) -> None:
        """Draw both closed polygons and unclosed paths with proper styling"""
        # Draw closed polygons from labels
        for polygon, color in self.labels:
            self._draw_single_annotation(bg_image, polygon, color, is_closed=True)
        
        # Draw unclosed paths from undo stack
        for entry in self.undo_stack:
            if entry.get('type') == 'draw' and entry.get('path'):
                path_item = entry['path']
                color = entry['label'][1]
                path = path_item.path()
                
                # Extract points from QPainterPath
                points = []
                for i in range(path.elementCount()):
                    elem = path.elementAt(i)
                    points.append(QPoint(int(elem.x), int(elem.y)))
                
                # Only draw if not closed
                if len(points) >= 2 and points[0] != points[-1]:
                    pen_width = max(1, path_item.pen().width())
                    self._draw_single_annotation(
                        bg_image, 
                        points, 
                        color, 
                        is_closed=False, 
                        pen_width=pen_width
                    )

            if entry.get('type') == 'flood_fill':
                color = entry.get('color')
                fill_mask = entry.get('mask')
                if color and fill_mask:
                    # color_layer = Image.new("RGBA", self.image.size, color.name())
                    color_layer = Image.new("RGBA", self.image.size, entry['color'].name())
                    bg_image.paste(color_layer, mask=fill_mask)


    def _draw_single_annotation(self, 
                            bg_image: Image.Image, 
                            points: list, 
                            color: QColor, 
                            is_closed: bool, 
                            pen_width: int = 2) -> None:
        """Draw a single annotation (polygon or line) onto background image"""
        mask = Image.new('L', bg_image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Convert QPoints to tuples
        point_tuples = [(p.x(), p.y()) for p in points if isinstance(p, QPoint)] \
                    if isinstance(points[0], QPoint) else points
        
        if is_closed and len(point_tuples) >= 3:
            # Draw closed polygon
            draw.polygon(point_tuples, fill=255)
        elif len(point_tuples) >= 2:
            # Draw lines with original pen width
            for i in range(len(point_tuples)-1):
                draw.line(
                    [point_tuples[i], point_tuples[i+1]], 
                    fill=255, 
                    width=pen_width,
                    joint="curve"  # Smooth line joints
                )
        
        # Apply color layer
        color_layer = Image.new('RGBA', bg_image.size, color.name())
        bg_image.paste(color_layer, mask=mask)

    def undo(self) -> None:
        # Handle in-progress drawings first
        if self.drawing:
            if self.current_path_item and self.current_path_item.scene() == self.scene:
                self.scene.removeItem(self.current_path_item)
            self.current_path_item = None
            self.current_label = []
            self.drawing = False
            self.statusBar.showMessage("Drawing canceled", 3000)
            return

        if self.undo_stack:
            entry = self.undo_stack.pop()

            if entry['type'] == 'flood_fill':
                if entry.get('overlay') and entry['overlay'].scene() == self.scene:
                    self.scene.removeItem(entry['overlay'])
            
            # Handle different entry types
            if entry['type'] == 'draw':
                # Remove visual elements if they exist in the scene
                if entry.get('overlay') and entry['overlay'].scene() == self.scene:
                    self.scene.removeItem(entry['overlay'])
                if entry.get('path') and entry['path'].scene() == self.scene:
                    self.scene.removeItem(entry['path'])
            
            elif entry['type'] == 'erase':
                # Restore erased elements
                for op in entry.get('operations', []):
                    original = op.get('original_entry')
                    if original and original.get('path') and original['path'].scene() is None:
                        self.scene.addItem(original['path'])
                    if original and original.get('overlay') and original['overlay'].scene() is None:
                        self.scene.addItem(original['overlay'])

            self.redo_stack.append(entry)
            self.label_counter = len(self.undo_stack)
            self.statusBar.showMessage(f"Undo: Operation reversed ({len(self.undo_stack)} left)", 3000)

    def redo(self) -> None:
        if self.redo_stack:
            entry = self.redo_stack.pop()

            if entry['type'] == 'flood_fill':
                if entry.get('overlay') and entry['overlay'].scene() is None:
                    self.scene.addItem(entry['overlay'])
            
            # Handle different entry types
            if entry['type'] == 'draw':
                # Add back elements if they're not already in the scene
                if entry.get('overlay') and entry['overlay'].scene() is None:
                    self.scene.addItem(entry['overlay'])
                if entry.get('path') and entry['path'].scene() is None:
                    self.scene.addItem(entry['path'])
            
            elif entry['type'] == 'erase':
                # Re-apply erasure
                for op in entry.get('operations', []):
                    original = op.get('original_entry')
                    if original and original.get('path') and original['path'].scene() == self.scene:
                        self.scene.removeItem(original['path'])
                    if original and original.get('overlay') and original['overlay'].scene() == self.scene:
                        self.scene.removeItem(original['overlay'])

            self.undo_stack.append(entry)
            self.label_counter = len(self.undo_stack)
            self.statusBar.showMessage(f"Redo: Operation restored ({len(self.redo_stack)} left)", 3000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSegmentationApp()
    window.show()
    sys.exit(app.exec_())