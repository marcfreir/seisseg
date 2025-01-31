import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QColorDialog, QLabel, QVBoxLayout, 
    QWidget, QPushButton, QHBoxLayout, QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem, QGraphicsPathItem, QInputDialog, QAction
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QPainterPath, QShowEvent
from PyQt5.QtCore import Qt, QPoint, QEvent
from PyQt5.QtSvg import QGraphicsSvgItem
from PIL import Image, ImageDraw
import tifffile
from skimage.filters import gabor, sobel
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        self.zoom_factor = 1.25
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

        # Add original image storage
        self.original_image = None  # Store original image for resetting
        self.current_image = None   # Store modified image

        # Graphics View
        self.view = CustomGraphicsView(self)
        self.layout.addWidget(self.view)
        self.scene = self.view.scene

        # Add SVG logo to the scene
        self.logo_item = QGraphicsSvgItem('img/seisseg_logo_cc.svg')

        self.scene.addItem(self.logo_item)

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

        # Initialize undo and redo stacks
        self.undo_stack = []
        self.redo_stack = []

    def showEvent(self, event: QShowEvent) -> None:
        """Handle window show event to fit the logo initially"""
        super().showEvent(event)
        if self.logo_item is not None and self.image is None:
            self.view.fitInView(self.logo_item, Qt.KeepAspectRatio)

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
            freq, ok = QInputDialog.getDouble(self, 'Gabor Filter', 'Frequency:', 0.1, 0.01, 1.0, 2)
            theta, ok = QInputDialog.getDouble(self, 'Gabor Filter', 'Theta (0-π):', 0, 0, np.pi, 2)

            if ok:
                image_array = img_as_ubyte(self.processed_data)
                filt_real, filt_imag = gabor(image_array, frequency=freq, theta=theta)
                filtered = np.sqrt(filt_real**2 + filt_imag**2)
                
                # Normalize
                filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
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
            radius, ok = QInputDialog.getInt(self, 'LBP', 'Radius:', 1, 1, 5)
            n_points, ok = QInputDialog.getInt(self, 'LBP', 'Points:', 8, 4, 24)
            
            if ok:
                image_array = img_as_ubyte(self.processed_data)
                
                # Calculate LBP
                lbp = local_binary_pattern(image_array, n_points, radius, method='uniform')
                
                # Normalize for display
                lbp = (lbp / lbp.max() * 255).astype(np.uint8)
                self.processed_data = lbp
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
            # img_gray = self.current_image.convert('L')
            # image_array = np.array(img_gray)

            image_array = self.processed_data
            
            # Calculate gradient using Sobel
            gradient = sobel(image_array)
            # self._update_display(gradient)
            self.processed_data = gradient

            self._update_display()

    def apply_hog(self) -> None:
        """
        Apply Histogram of Oriented Gradients (HOG) feature extraction.

        Converts the processed image data to uint8 and calculates the HOG features.
        The HOG image is normalized and used to update the display.

        This method updates the internal processed_data with the normalized HOG image.
        """

        if self.processed_data is not None:
            image_array = img_as_ubyte(self.processed_data)  # Convert to uint8
            
            # Calculate HOG
            _, hog_image = hog(image_array, 
                            orientations=8, 
                            pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), 
                            visualize=True)
            
            # Normalize HOG image
            hog_image = hog_image.astype(np.float32)
            hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
            self.processed_data = hog_image
            self._update_display()

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

        if not self.image or not self.labels:
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
        """
        Handle a left mouse button press event.

        If the press event is inside the image area, start drawing a new label.
        Create a new QPainterPath and add it to the scene.

        :param scene_pos: The position of the press event in scene coordinates.
        :return: None
        """

        if self.image:
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
        """
        Handle a left mouse button move event.

        If the move event occurs while drawing is active and within the image
        boundaries, this method appends the new position to the current label 
        path and updates the corresponding QGraphicsPathItem in the scene.

        :param scene_pos: The position of the move event in scene coordinates.
        :type scene_pos: QPoint
        :return: None
        """

        if self.drawing and self.image:
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                self.current_label.append(QPoint(x, y))
                self.current_path.lineTo(x, y)
                self.current_path_item.setPath(self.current_path)

    def handle_left_release(self, scene_pos: QPoint) -> None:
        """
        Handle a left mouse button release event.

        If the release event occurs while drawing is active, this method stops
        drawing, checks if the polygon is closed by right-clicking, and fills the
        label if it is.

        :param scene_pos: The position of the release event in scene coordinates.
        :type scene_pos: QPoint
        :return: None
        """

        if self.drawing:
            self.drawing = False
            # Check if right-click to close polygon
            if len(self.current_label) > 2:
                self.current_label.append(self.current_label[0])
                self.current_path.lineTo(self.current_label[0].x(), self.current_label[0].y())
                self.current_path_item.setPath(self.current_path)
                self.fill_label()

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
        
        if self.image and self.current_label:
            polygon = [(p.x(), p.y()) for p in self.current_label]
            mask = Image.new('L', self.image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon, fill=255)

            color_layer = Image.new('RGBA', self.image.size, self.current_color.name())
            overlay = Image.alpha_composite(self.image.convert('RGBA'), color_layer.convert('RGBA'))
            overlay.putalpha(mask)

            qimage = QImage(overlay.tobytes(), overlay.width, overlay.height, 
                            overlay.width * 4, QImage.Format_RGBA8888)
            overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
            self.scene.addItem(overlay_item)
            self.labels.append((self.current_label, self.current_color))

            # Store both overlay and path in the undo stack
            self.undo_stack.append({
                'label': (self.current_label.copy(), self.current_color),
                'overlay': overlay_item,
                'path': self.current_path_item
            })
            self.redo_stack.clear()  # Clear redo stack on new action
            self.labels.append((self.current_label.copy(), self.current_color))   # Add to labels list

            self.save_label_log()

            label_name = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
            nth_label = label_name[self.label_counter] if self.label_counter < len(label_name) else f'{self.label_counter + 1}th'
            self.statusBar.showMessage(f'{nth_label} label created successfully!', 3000)
            self.label_counter += 1

    def reset_labels(self) -> None:
        """
        Resets the labels and clears the scene.

        This method clears all existing labels and the scene, then re-adds the image
        to the scene if it is set. It also resets the label counter.

        Preconditions:
        - `self.image` may be set to a valid image or None.

        Postconditions:
        - The scene is cleared of all items.
        - The image is re-added to the scene if it exists.
        - The labels list is cleared.
        - The label counter is reset to zero.

        :return: None
        """

        self.scene.clear()
        if self.image:
            qimage = QImage(self.image.tobytes(), self.image.width, self.image.height, 
                           self.image.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.image_item)
        self.labels = []
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.label_counter = 0

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
        """
        Save all labels as a single PNG image.

        This method prompts the user for a file path to save the labels, then creates
        a white background with the labels and saves it as a PNG. The filename is
        ensured to end with '.png' if it does not already.

        Preconditions:
        - `self.image` must be set to a valid image.
        - `self.labels` must contain the points of the labels.

        Postconditions:
        - The labels are saved as a PNG image.
        - A status message is displayed indicating the file path of the saved image.

        :return: None
        """

        if self.image:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 'Save Labels', '', 'PNG Files (*.png)'
            )
            if file_path:
                # Ensure the filename ends with .png
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                
                # Create white background with labels
                white_bg = Image.new('RGBA', self.image.size, (255, 255, 255, 255))
                for label in self.labels:
                    polygon, color = label
                    mask = Image.new('L', self.image.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.polygon([(p.x(), p.y()) for p in polygon], fill=255)
                    color_layer = Image.new('RGBA', self.image.size, color.name())
                    white_bg.paste(color_layer, mask=mask)
                
                # Convert to RGB before saving to avoid alpha channel issues
                white_bg.convert('RGB').save(file_path)
                self.statusBar.showMessage(f'Labels saved as {file_path}', 3000)


    def undo(self) -> None:
        if self.undo_stack:
            entry = self.undo_stack.pop()
            # Remove the label from the labels list
            if self.labels:
                self.labels.pop()
            # Remove visual elements from the scene
            self.scene.removeItem(entry['overlay'])
            self.scene.removeItem(entry['path'])
            # Add to redo stack
            self.redo_stack.append(entry)
            # Update counter
            self.label_counter = len(self.labels)
            self.statusBar.showMessage(f"Undo: Label removed ({len(self.undo_stack)} left)", 3000)

    def redo(self) -> None:
        if self.redo_stack:
            entry = self.redo_stack.pop()
            # Add the label back to the labels list
            self.labels.append(entry['label'])
            # Add visual elements back to the scene
            self.scene.addItem(entry['overlay'])
            self.scene.addItem(entry['path'])
            # Add to undo stack
            self.undo_stack.append(entry)
            # Update counter
            self.label_counter = len(self.labels)
            self.statusBar.showMessage(f"Redo: Label restored ({len(self.redo_stack)} left)", 3000)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSegmentationApp()
    window.show()
    sys.exit(app.exec_())