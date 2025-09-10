## Dependencies needed
import napari
from napari.layers import Shapes
from napari.utils.notifications import show_info
import pandas as pd
import numpy as np
import random
import tifffile as tiff
import scimap as sm 
from tifffile import imread
import dask.array as da
import zarr
import os
import ast
from pathlib import Path
from magicgui import magicgui
from PyQt5.QtWidgets import (
    QMessageBox, 
    QDialog, 
    QVBoxLayout, 
    QCheckBox, 
    QDialogButtonBox, 
    QApplication,
    QWidget,
    QPushButton
)
from PyQt5.QtCore import QSettings, Qt
import sys
from dask_image.imread import imread as daskread
from io import BytesIO
import re
from magicgui.widgets import PushButton
from napari.layers import Shapes, Image

# Initial configuration
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Widget Selection")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self.widget_list = [
            "Open image", "Open mask", "Load shapes",
            "Contrast limits", "Save shapes", "Crop ROI",
            "Count cells", "Export cells", "Metadata",
            "Voronoi", "Save Viewport", "Load points", 
            "Circle with n cells", "Extract Cells in Shape","Gating", "Close all"  # Fixed comma
        ]
        
        self.settings = QSettings("MyLab", "NapariTools")
        
        layout = QVBoxLayout()
        self.checkboxes = {}
        
        for widget in self.widget_list:
            cb = QCheckBox(widget)
            cb.setChecked(self.settings.value(widget, False, type=bool))
            self.checkboxes[widget] = cb
            layout.addWidget(cb)
            
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.save_settings)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
        self.setLayout(layout)
    
    def save_settings(self):
        for widget, cb in self.checkboxes.items():
            self.settings.setValue(widget, cb.isChecked())
        self.accept()

app = QApplication.instance() or QApplication(sys.argv)
dialog = SettingsDialog()
if not dialog.exec_():
    sys.exit()

viewer = napari.Viewer()

# -------------------------------------------------------------------------------
# Widget implementations
# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------
# Widget implementations - Extract Cells in Shape
# -------------------------------------------------------------------------------


@magicgui(
    call_button='Open image',
    layout='vertical',
    image_path={
        "label": "Image Path",
        "filter": "*.tif *.tiff *.ome.tif",
        "mode": "r"
    },
    contrast_limit_txt={
        "label": "Contrast Limits (optional)",
        "filter": "*.txt",
        "mode": "r",
        "nullable": True
    },
    ab_list_path={
        "label": "Channel Names",
        "filter": "*.txt",
        "mode": "r",
        "nullable": True
    }
)
def open_large_image(image_path: Path = Path("."), 
                    contrast_limit_txt: Path = None,
                    ab_list_path: Path = None):
    """Open a multichannel image with pyramidal handling"""
    try:
        if not image_path.is_file():
            show_info("Please select a valid image file")
            return

        # Handle channel names
        channel_names = []
        if ab_list_path and ab_list_path.is_file():
            try:
                ab_df = pd.read_csv(ab_list_path)
                channel_names = list(ab_df["ABS"]) if "ABS" in ab_df.columns else []
            except Exception as e:
                show_info(f"Error reading channel names: {str(e)}")
                return

        # Handle contrast limits
        contrast_limits = None
        if contrast_limit_txt and contrast_limit_txt.is_file():
            try:
                with open(contrast_limit_txt, 'r') as f:
                    contrast_limits = ast.literal_eval(f.read())
            except Exception as e:
                show_info(f"Error reading contrast limits: {str(e)}")
                return

        # Load image metadata and data
        with tiff.TiffFile(image_path) as tf:
            series = tf.series[0]
            axes = series.axes

            # Determine number of channels
            if 'C' in axes:
                c_idx = axes.index('C')
                num_channels = series.shape[c_idx]
            else:
                num_channels = 1  # Single-channel image

            is_pyramidal = len(series.levels) > 1

            # Load image data
            if is_pyramidal:
                pyramid = [da.from_zarr(zarr.open(tf.aszarr(level=i))) 
                          for i in range(len(series.levels))]
            else:
                pyramid = [da.from_zarr(zarr.open(tf.aszarr()))]

            # Add channel dimension if missing
            if 'C' not in axes:
                pyramid = [level[np.newaxis, ...] for level in pyramid]

            # Auto-generate pyramid for large non-pyramidal images
            if not is_pyramidal:
                base_level = pyramid[0]
                base_height = base_level.shape[-2]
                base_width = base_level.shape[-1]
                
                # Generate pyramid if dimensions exceed GPU texture limits
                if base_height > 16384 or base_width > 16384:
                    current_level = base_level
                    while any(dim > 4096 for dim in current_level.shape[-2:]):
                        next_level = current_level[..., ::2, ::2]  # 2x2 binning
                        pyramid.append(next_level)
                        current_level = next_level
                    is_pyramidal = True

            # Generate automatic channel names if needed
            if not channel_names:
                channel_names = [f"Channel_{i+1}" for i in range(num_channels)]

        # Add image to viewer
        viewer.add_image(
            pyramid,
            channel_axis=0,
            name=channel_names,
            contrast_limits=contrast_limits,
            visible=False,
            multiscale=is_pyramidal
        )

        show_info(f"Image loaded: {image_path.name}\n" 
                 f"Dimensions: {pyramid[0].shape}\n"
                 f"Pyramid levels: {len(pyramid)}")

    except Exception as e:
        show_info(f"Critical error: {str(e)}")

# -------------------------------------------------------------------------------
# Widget implementations - Open Mask
# -------------------------------------------------------------------------------

@magicgui(call_button='Open mask', layout='vertical')
def open_mask(mask_path=Path()):
    seg_m = tiff.imread(mask_path)
    if (len(seg_m.shape) > 2) and (seg_m.shape[0] > 1):
        seg_m = seg_m[0]
    viewer.add_labels(seg_m, name='MASK')


# -------------------------------------------------------------------------------
# Widget implementations - Load Shapes
# -------------------------------------------------------------------------------

@magicgui(call_button='Load Shapes', layout='vertical', shapes_path={"mode": "d"})
def load_shapes(shapes_path: Path):
    """Load shapes from JSON (new format) or TXT (old format)"""
    shapes_path = Path(shapes_path)
    if not shapes_path.is_dir():
        show_info("Please select a valid directory")
        return

    import json, re

    for filename in shapes_path.glob("*"):
        if filename.name.startswith('._'):  # evitar archivos ocultos
            continue
        try:
            if filename.suffix.lower() == ".json":
                # 🔹 Nuevo formato JSON
                with open(filename, 'r') as f:
                    shape_list = json.load(f)

                for shape in shape_list:
                    s_type = shape.get("type", "polygon")
                    vertices = np.array(shape.get("vertices", []), dtype=np.float32)

                    if len(vertices) == 0:
                        continue

                    viewer.add_shapes(
                        [vertices],
                        shape_type=s_type,
                        edge_width=1,
                        edge_color='#777777',
                        face_color='red',
                        name=f"{filename.stem}_{s_type}"
                    )

                show_info(f"Loaded shapes from {filename.name} (JSON)")

            elif filename.suffix.lower() == ".txt":
                # 🔹 Formato antiguo con array de NumPy
                with open(filename, 'r') as f:
                    content = f.read()

                match = re.search(r'array\(([\s\S]*?)(?:,\s*dtype=(\w+))?\)', content, re.DOTALL)
                if not match:
                    raise ValueError("Invalid numpy array format in TXT")

                array_str = match.group(1).strip()
                dtype_str = match.group(2) if match.group(2) else 'float32'
                array_data = ast.literal_eval(array_str)
                shape_array = np.array(array_data, dtype=getattr(np, dtype_str))

                viewer.add_shapes(
                    shape_array,
                    shape_type='polygon',  # 🔴 en TXT siempre se guarda como polígono
                    edge_width=1,
                    edge_color='#777777',
                    face_color='red',
                    name=filename.stem
                )

                show_info(f"Loaded shapes from {filename.name} (TXT)")

        except Exception as e:
            show_info(f"Error loading {filename.name}:\n{str(e)}")



# -------------------------------------------------------------------------------
# Widget implementations - Save contrast limits
# -------------------------------------------------------------------------------


@magicgui(call_button='Save contrast limits', layout='vertical', output_file={"mode": "d"})
def save_contrast_limits(output_file: Path, ab_list_path=Path(), name=""):
    contrast_limit = []
    ab = pd.read_csv(ab_list_path)
    ab = list(ab["ABS"])
    for antibody in ab:
        contrast_limit.append(viewer.layers[antibody].contrast_limits)

    with open(output_file / f"{name}.txt", "w") as output:
        output.write(str(contrast_limit))



# -------------------------------------------------------------------------------
# Widget implementations - Save Shape
# -------------------------------------------------------------------------------


@magicgui(call_button='Save shapes', layout='vertical', output_file={"mode": "d"})
def save_shapes(output_file: Path, shape_name=""):
    try:
        if shape_name not in viewer.layers:
            show_info(f'No shape layer named "{shape_name}" found')
            return

        shapes_layer = viewer.layers[shape_name]
        shapes_data = shapes_layer.data          # lista de arrays de vértices
        shape_types = shapes_layer.shape_type    # lista de tipos de cada shape

        # --------- Guardar en JSON (nuevo formato) ---------
        save_list = []
        for shape, s_type in zip(shapes_data, shape_types):
            save_list.append({
                "type": s_type,
                "vertices": shape.tolist()
            })

        json_path = output_file / f"{shape_name}.json"
        import json
        with open(json_path, "w") as f:
            json.dump(save_list, f, indent=2)

        # --------- Guardar en TXT (formato antiguo, uno por shape) ---------
        txt_dir = output_file / f"{shape_name}_txt"
        txt_dir.mkdir(exist_ok=True)

        for i, shape in enumerate(shapes_data):
            txt_path = txt_dir / f"{shape_name}_{i+1}.txt"
            with open(txt_path, "w") as f:
                f.write(repr(np.array(shape, dtype=np.float32)))

        show_info(
            f"Saved {len(shapes_data)} shapes to:\n"
            f"- {json_path.name} (JSON)\n"
            f"- {txt_dir}/ (TXT, {len(shapes_data)} files)"
        )

    except Exception as e:
        show_info(f"Error saving shapes: {str(e)}")





# -------------------------------------------------------------------------------
# Widget implementations - Extract Cells in Shape
# -------------------------------------------------------------------------------


@magicgui(call_button='Cut and Save ROIs', filepath={"mode": "d"})
def cut_mask(filepath: Path, shape_name=""):
    if 'MASK' not in viewer.layers:
        show_info('No mask layer named "MASK" found')
        return
    if shape_name not in viewer.layers:
        show_info(f'No shape layer named "{shape_name}" found')
        return

    mask_to_cut = viewer.layers['MASK'].data
    shape = mask_to_cut.shape
    selected_area = viewer.layers[shape_name].to_labels(labels_shape=shape)
    removable_cells = []
    for i in range(mask_to_cut.shape[0]):
        for j in range(mask_to_cut.shape[1]):
            cell = mask_to_cut[i, j]
            if selected_area[i, j] > 0 and cell not in removable_cells and cell > 0:
                removable_cells.append(cell)
    df = pd.DataFrame({'cellid': removable_cells})
    df = df.astype(int)
    df.to_csv(filepath / f'{shape_name}_selected_cell_ids.csv', index=False)



# -------------------------------------------------------------------------------
# Widget implementations - Close All
# -------------------------------------------------------------------------------

@magicgui(call_button='Close all', layout='vertical')
def close_all():
    viewer.layers.clear()



# -------------------------------------------------------------------------------
# Widget implementations - View metadata
# -------------------------------------------------------------------------------

@magicgui(call_button='View metadata', layout='vertical')
def view_metadata(adata_path=Path(), image_name="", metadata_column=""):
    path = str(adata_path)
    
    # Read the actual CSV file to detect column names
    try:
        df = pd.read_csv(path, nrows=0)
        # Find case-insensitive match for CellID
        cellid_variations = [col for col in df.columns if col.lower() == 'cellid']
        detected_cell_id = cellid_variations[0] if cellid_variations else 'CellID'
    except Exception as e:
        print(f"Error reading columns: {e}")
        detected_cell_id = 'CellID'  # Fallback
    
    # Load data with detected CellID column
    adata = sm.pp.mcmicro_to_scimap(
        path,
        remove_dna=False,
        remove_string_from_name=None,
        log=False,
        random_sample=None,
        CellId=detected_cell_id,  # Use detected column name
        split='X_centroid',
        custom_imageid=None,
        min_cells=None,
        output_dir=None
    )
    
    
    adata = adata[adata.obs['imageid'] == image_name]
    available_phenotypes = list(adata.obs[metadata_column].unique())
    
    for i in available_phenotypes:
        coordinates = adata[adata.obs[metadata_column] == i]
        coordinates = pd.DataFrame({
            'y': coordinates.obs["Y_centroid"],
            'x': coordinates.obs["X_centroid"]
        })
        points = coordinates.values
        r = lambda: random.randint(0, 255)
        point_color = '#%02X%02X%02X' % (r(), r(), r())
        viewer.add_points(points, size=25, face_color=point_color, 
                        visible=False, name=i)


# Modified to accept more column "CellID" names
# Cruz Osuna

# -------------------------------------------------------------------------------
# Widget implementations - Count selected cells
# -------------------------------------------------------------------------------

@magicgui(call_button='Count selected cells', layout='vertical')
def count_selected_cells(shape_name: str = "", cell_info_csv: Path = Path()):
    if 'MASK' not in viewer.layers:
        show_info('No mask layer named "MASK" found')
        return
    if shape_name not in viewer.layers:
        show_info(f'No shape layer named "{shape_name}" found')
        return

    mask_layer = viewer.layers['MASK']
    mask_data = mask_layer.data
    shape_layer = viewer.layers[shape_name]
    shape_data = shape_layer.to_labels(labels_shape=mask_data.shape)

    overlapping_cells = mask_data[shape_data > 0]
    unique_cells = np.unique(overlapping_cells)
    unique_cells = unique_cells[unique_cells != 0]
    cell_count = len(unique_cells)

    show_info(f'Total cells within "{shape_name}": {cell_count}')




# -------------------------------------------------------------------------------
# Widget implementations - Save cells in selected ROI
# -------------------------------------------------------------------------------

@magicgui(call_button='Save cells in selected ROI', layout='vertical', output_csv={"mode": "d"})
def save_selected_cells(output_csv: Path, shape_name: str = "", cell_info_csv: Path = Path(), output_file_name: str = ""):
    if 'MASK' not in viewer.layers:
        show_info('No mask layer named "MASK" found')
        return
    if shape_name not in viewer.layers:
        show_info(f'No shape layer named "{shape_name}" found')
        return

    mask_layer = viewer.layers['MASK']
    mask_data = mask_layer.data
    shape_layer = viewer.layers[shape_name]
    shape_data = shape_layer.to_labels(labels_shape=mask_data.shape)

    overlapping_cells = mask_data[shape_data > 0]
    unique_cells = np.unique(overlapping_cells)
    unique_cells = unique_cells[unique_cells != 0]
    cell_count = len(unique_cells)

    show_info(f'Total cells within "{shape_name}": {cell_count}')

    try:
        cell_info_df = pd.read_csv(cell_info_csv)
    except Exception as e:
        show_info(f'Error reading cell information file: {e}')
        return

    cell_id_column = None
    for col in ['cellid', 'CellID', 'cell_id', 'Cell_Id', 'cellID']:
        if col in cell_info_df.columns:
            cell_id_column = col
            break
    if cell_id_column is None:
        show_info('No cell ID column found in cell information file')
        return

    selected_cells_info = cell_info_df[cell_info_df[cell_id_column].isin(unique_cells)]

    try:
        selected_cells_info.to_csv(output_csv / f"{output_file_name}.csv", index=False)
        show_info(f'Information on {cell_count} selected cells saved in {output_csv}')
    except Exception as e:
        show_info(f'Error saving selected cells file: {e}')




# -------------------------------------------------------------------------------
# Widget implementations - Voronoi plot
# -------------------------------------------------------------------------------

@magicgui(call_button='Voronoi plot', layout='vertical', output_dir={"mode": "d"})
def voronoi_plot(output_dir: Path, adata_path=Path(), shape_name="", image_name="", cluster_name="", file_name=""):
    path = str(adata_path)
    adata = sm.pp.mcmicro_to_scimap(path, remove_dna=False, remove_string_from_name=None, log=False,
                                    random_sample=None, CellId='CellID', split='X_centroid',
                                    custom_imageid=None, min_cells=None, output_dir=None)
    
    # Get shape boundaries using bounding box
    shapes = viewer.layers[shape_name].data
    shape_bounds = np.array([shape[:, -2:] for shape in shapes])  # Get XY coordinates
    x_min, y_min = np.min(shape_bounds, axis=(0, 1))
    x_max, y_max = np.max(shape_bounds, axis=(0, 1))
    x_1, x_2 = x_min, x_max
    y_1, y_2 = y_min, y_max
    
    n_colors = {0: "#D3D3D3", 1: '#D3D3D3', 2: "#D3D3D3", 3: "#FF0000", 4: "#D3D3D3",
                5: "#D3D3D3", 6: '#D3D3D3', 7: "#FFD343", 8: "#D3D3D3", 9: "#D3D3D3"}
    sm.pl.voronoi(adata, color_by=cluster_name, x_coordinate='X_centroid', y_coordinate='Y_centroid', imageid='imageid',
                  subset=image_name, x_lim=[x_1, x_2], y_lim=[y_1, y_2], plot_legend=True, flip_y=True,
                  overlay_points=cluster_name, voronoi_alpha=0.7, voronoi_line_width=0.3, overlay_point_size=8,
                  overlay_point_alpha=1, legend_size=15, overlay_points_colors=n_colors, colors=n_colors,
                  fileName=f"{file_name}.pdf", saveDir=str(output_dir))




# -------------------------------------------------------------------------------
# Widget implementations - Save Viewport
# -------------------------------------------------------------------------------

@magicgui(
    call_button='Save Viewport',
    layout='vertical',
    output_dir={"label": "Output Directory", "mode": "d"},
    filename={"label": "Filename", "tooltip": "Without extension"},
    image_layer={
        "label": "Image Layer", 
        "choices": lambda _: [layer.name for layer in viewer.layers if isinstance(layer, napari.layers.Image)]
    }
)
def save_viewport(
    output_dir: Path = Path.home(),
    filename: str = "viewport_snapshot",
    image_layer: str = None
):
    """Save current field of view as TIFF"""
    try:
        if not image_layer:
            show_info("Please select an image layer")
            return
            
        layer = viewer.layers[image_layer]
        
        # Get current view parameters
        view = viewer.window.qt_viewer
        canvas_size = view.canvas.size
        camera_zoom = view.camera.zoom

        # Calculate visible area in data coordinates
        transform = layer._transforms[0:2]  # Get spatial transforms
        visible_rect = view.camera.rect
        top_left = transform.inverse(visible_rect.top_left)
        bottom_right = transform.inverse(visible_rect.bottom_right)

        # Convert to pixel coordinates
        y_start = int(max(0, top_left[0]))
        y_end = int(min(layer.data.shape[-2], bottom_right[0]))
        x_start = int(max(0, top_left[1]))
        x_end = int(min(layer.data.shape[-1], bottom_right[1]))

        # Handle multiscale images
        if layer.multiscale:
            # Calculate optimal pyramid level
            base_scale = layer.data[0].shape[-2:]
            scale_factors = [
                (base_scale[0]/level_data.shape[-2], 
                base_scale[1]/level_data.shape[-1]
            ) for level_data in layer.data
            ]
            
            # Find level closest to current zoom
            target_scale = 1 / camera_zoom
            level = np.argmin([
                abs((sf[0] + sf[1])/2 - target_scale) 
                for sf in scale_factors
            ])
            
            data = layer.data[level]
            sf_y, sf_x = scale_factors[level]

            # Adjust coordinates for pyramid level
            y_start = int(y_start / sf_y)
            y_end = int(y_end / sf_y)
            x_start = int(x_start / sf_x)
            x_end = int(x_end / sf_x)
        else:
            data = layer.data

        # Extract viewport data with channel handling
        if data.ndim == 2:
            viewport = data[y_start:y_end, x_start:x_end]
        elif data.ndim == 3:  # Handle CYX format
            viewport = data[:, y_start:y_end, x_start:x_end]
        elif data.ndim == 4:  # Handle TCYX format
            viewport = data[0, :, y_start:y_end, x_start:x_end]
        else:
            show_info("Unsupported image dimensions")
            return

        # Save TIFF
        output_path = output_dir / f"{filename}.tiff"
        tiff.imwrite(output_path, viewport)
        show_info(f"Viewport saved:\n{output_path.name}")

    except Exception as e:
        show_info(f"Error saving viewport: {str(e)}")




# -------------------------------------------------------------------------------
# Widget implementations - Load Points
# -------------------------------------------------------------------------------

@magicgui(call_button='Load Points', layout='vertical', points_path={"mode": "r", "filter": "*.csv"})
def load_points(points_path: Path):
    """Load sampling points layer from CSV"""
    try:
        # Read CSV with points data
        points_df = pd.read_csv(points_path)
        
        # Validate required columns
        if not {'x', 'y'}.issubset(points_df.columns):
            show_info("CSV must contain 'x' and 'y' columns")
            return

        # Extract coordinates and optional properties
        points_data = points_df[['x', 'y']].values
        properties = {
            'label': points_df['label'].tolist() if 'label' in points_df.columns else None
        }

        # Create points layer with optional text labels
        points_layer = viewer.add_points(
            points_data,
            name=points_path.stem,
            size=10,
            face_color='magenta',
            edge_color='black',
            properties=properties,
            text='label' if 'label' in points_df.columns else None
        )

        # Set initial visibility settings
        points_layer.visible = True
        show_info(f"Loaded {len(points_data)} points from {points_path.name}")

    except Exception as e:
        show_info(f"Error loading points: {str(e)}")


# -------------------------------------------------------------------------------
# Widget implementations - Circle with n cells
# -------------------------------------------------------------------------------


# Widget button and handler
pick_center_button = QPushButton("Pick Center with Click")

def on_pick_center_click():
    """Triggered when 'Pick Center with Click' button is pressed"""
    show_info("Click on the image to select center...")

    def get_click(layer, event):
        """Mouse click handler with proper coordinate conversion"""
        if event.type == 'mouse_press' and event.button == 1:
            display_coords = event.position
            data_coords = layer.world_to_data(display_coords)
            
            # Swap X/Y for physical coordinates
            physical_x = data_coords[-1]  # This is actually Y in image terms
            physical_y = data_coords[-2]  # This is actually X in image terms
            
            # Store coordinates
            create_circle_for_n_cells.center_x_display.value = display_coords[0]
            create_circle_for_n_cells.center_y_display.value = display_coords[1]
            create_circle_for_n_cells.center_x_physical.value = physical_x
            create_circle_for_n_cells.center_y_physical.value = physical_y
            
            show_info(f"Display: X={display_coords[0]:.1f}, Y={display_coords[1]:.1f}\n"
                     f"Physical: X={physical_x:.1f}, Y={physical_y:.1f}")
            layer.mouse_drag_callbacks.remove(get_click)

    if viewer.layers:
        viewer.layers[0].mouse_drag_callbacks.append(get_click)
    else:
        show_info("No image layer available!")

pick_center_button.clicked.connect(on_pick_center_click)

# Main widget function
@magicgui(
    call_button='Create Circle',
    layout='vertical',
    cell_info_csv={"label": "Cell Data CSV", "mode": "r", "filter": "*.csv"},
    center_x_display={
        'visible': False,
        'min': -1e10,
        'max': 1e10,
        'tooltip': 'X coordinate in display space'
    },
    center_y_display={
        'visible': False,
        'min': -1e10,
        'max': 1e10,
        'tooltip': 'Y coordinate in display space'
    },
    center_x_physical={'visible': False, 'min': -1e10, 'max': 1e10},
    center_y_physical={'visible': False, 'min': -1e10, 'max': 1e10},
    num_cells={
        'min': 1,
        'max': 1_000_000,
        'step': 1,
        'tooltip': 'Number of cells to include in the circle'
    },
    shape_name={
        'tooltip': 'Base name for the new shape layer'
    }
)
def create_circle_for_n_cells(
    cell_info_csv: Path = None,
    center_x_display: float = 0.0,
    center_y_display: float = 0.0,
    center_x_physical: float = 0.0,
    center_y_physical: float = 0.0,
    shape_name: str = "ROI_Sample_#Circle",
    num_cells: int = 1000
):
    """Create a circle containing exactly n cells from CSV data"""
    try:
        # Validate inputs
        if not cell_info_csv or not cell_info_csv.exists():
            show_info("Please select a valid CSV file")
            return
            
        img_layer = next((l for l in viewer.layers if isinstance(l, Image)), None)
        if not img_layer:
            show_info("Load an image layer first!")
            return

        # Get image properties with axis swap
        scale_y, scale_x = img_layer.scale[-2:]  # Swap scale factors
        translate_y, translate_x = img_layer.translate[-2:]  # Swap translations

        # Load and validate cell data
        df = pd.read_csv(cell_info_csv)
        
        # Verify required columns
        required_columns = ['X_centroid', 'Y_centroid']
        if not all(col in df.columns for col in required_columns):
            show_info(f"CSV must contain columns: {required_columns}")
            return

        # Convert physical coordinates to display coordinates with axis swap
        df['display_x'] = (df['Y_centroid'] - translate_x) / scale_x  # Swap X/Y
        df['display_y'] = (df['X_centroid'] - translate_y) / scale_y  # Swap X/Y

        # Calculate distances from center
        df['distance'] = np.sqrt(
            (df['display_x'] - center_x_display)**2 + 
            (df['display_y'] - center_y_display)**2
        )
        
        # Sort and find radius
        df_sorted = df.sort_values('distance')
        target_num = min(num_cells, len(df))
        
        if target_num == 0:
            show_info("No cells found in the dataset!")
            return
            
        radius_display = df_sorted.iloc[target_num-1]['distance']

        # Generate circle points
        theta = np.linspace(0, 2*np.pi, 100)
        circle_pts = np.array([[
            center_x_display + radius_display * np.cos(t),
            center_y_display + radius_display * np.sin(t)
        ] for t in theta])

        # Create unique layer name
        base_name = shape_name.split('#')[0]
        existing_names = {layer.name for layer in viewer.layers}
        suffix = 1
        while f"{base_name}{suffix}" in existing_names:
            suffix += 1
        final_name = f"{base_name}{suffix}"

        # Add debug visualization
        debug_points = df_sorted.head(target_num)[['display_x', 'display_y']].values
        viewer.add_points(
            debug_points,
            size=10,
            face_color='red',
            name=f'DEBUG_{final_name}',
            visible=True
        )

        # Add shape to viewer
        viewer.add_shapes(
            data=[circle_pts],
            shape_type='polygon',
            edge_color='#ffdd00',
            edge_width=5,
            face_color='#0000ff22',
            name=final_name,
            scale=(1.0, 1.0),
            translate=(0.0, 0.0)
        )

        show_info(
            f"Circle created at:\n"
            f"Display X: {center_x_display:.1f}, Y: {center_y_display:.1f}\n"
            f"Mapped Cells: {target_num}/{num_cells}\n"
            f"Radius: {radius_display:.1f}px"
        )

    except Exception as e:
        show_info(f"Error: {str(e)}")

# Widget container creation
def create_circle_widget():
    container = QWidget()
    layout = QVBoxLayout()
    container.setLayout(layout)
    layout.addWidget(create_circle_for_n_cells.native)
    layout.addWidget(pick_center_button)
    return container


# -------------------------------------------------------------------------------
# Widget implementations - Extract Cells in Shape
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Widget implementations - Extract Cells in Shape (Corregido)
# -------------------------------------------------------------------------------

@magicgui(
    call_button='Extract Cells in Shape',
    layout='vertical',
    sample={"label": "Sample Name"},
    cell_csv={"label": "Cell Data CSV", "mode": "r", "filter": "*.csv"},
    shape_name={"label": "Shape Layer", "choices": lambda _: [layer.name for layer in viewer.layers if isinstance(layer, Shapes)]},
    output_mode={"label": "Output Mode", "choices": ["New CSV", "Add label to existing"]},
    label_column={"label": "Column Name", "visible": False},
    label_value={"label": "Annotation", "visible": False},
    output_dir={"label": "Output Directory", "mode": "d", "visible": True},
    output_name={"label": "New File Name (Optional)", "visible": True}
)
def extract_cells_in_shape(
    sample: str,
    cell_csv: Path,
    shape_name: str,
    output_mode: str = "New CSV",
    label_column: str = "ROI_Label",
    label_value: str = "Selected",
    output_dir: Path = Path(),
    output_name: str = ""
):
    """Extract cell data within specified shape and either save to new CSV or add label to existing"""
    try:
        # Validate common inputs
        if not cell_csv.exists():
            show_info("Cell CSV file not found")
            return
            
        # Find shape layer
        shape_layer = next((layer for layer in viewer.layers if layer.name == shape_name and isinstance(layer, Shapes)), None)
        if shape_layer is None:
            show_info(f"Shape layer '{shape_name}' not found")
            return

        # Automatically get the first image layer
        img_layer = next((layer for layer in viewer.layers if isinstance(layer, Image)), None)
        if img_layer is None:
            show_info("No image layers found in the viewer")
            return

        # Read only necessary columns for better performance
        use_cols = ['X_centroid', 'Y_centroid', 'CellID', 'Sample']
        df = pd.read_csv(cell_csv, usecols=use_cols)
        
        # Verify required columns
        if not all(col in df.columns for col in use_cols):
            missing = set(use_cols) - set(df.columns)
            show_info(f"Missing columns in CSV: {', '.join(missing)}")
            return
            
        # Filter by sample early to reduce dataset size
        sample_df = df[df['Sample'] == sample].copy()
        if len(sample_df) == 0:
            show_info(f"No cells found for sample: {sample}")
            return

        # Get dimensions and scale factors
        if img_layer.multiscale:
            base_level = img_layer.data[0]
            scale_factor = img_layer.scale[-2:]
            y_dim, x_dim = base_level.shape[-2:]
        else:
            scale_factor = img_layer.scale[-2:]
            y_dim, x_dim = img_layer.data.shape[-2:]

        # Create mask from shape (once)
        mask = shape_layer.to_labels(labels_shape=(y_dim, x_dim))

        # Convert coordinates to pixels using vectorized operations
        x_coords = (sample_df['X_centroid'].values / scale_factor[1]).astype(int)
        y_coords = (sample_df['Y_centroid'].values / scale_factor[0]).astype(int)
        
        # Create mask for valid coordinates
        valid_coords = (x_coords >= 0) & (x_coords < x_dim) & (y_coords >= 0) & (y_coords < y_dim)
        
        # Initialize shape mask
        in_shape_mask = np.zeros(len(sample_df), dtype=bool)
        
        # Check which cells are inside the shape (vectorized)
        if np.any(valid_coords):
            valid_x = x_coords[valid_coords]
            valid_y = y_coords[valid_coords]
            in_shape_mask[valid_coords] = mask[valid_y, valid_x] > 0

        # Filter DataFrame
        filtered_df = sample_df[in_shape_mask]
        cell_count = len(filtered_df)
        
        if cell_count == 0:
            show_info("No cells found within the specified shape")
            return

        # MODE 1: Export to new CSV
        if output_mode == "New CSV":
            output_path = output_dir / f"{output_name}.csv"
            filtered_df.to_csv(output_path, index=False)
            show_info(f"Saved {cell_count} cells from sample '{sample}' to:\n{output_path}")
        
        # MODE 2: Add label to existing CSV
        else:
            # Validate label parameters
            if not label_column:
                show_info("Please specify a label column name")
                return
            
            # Read full file only if necessary
            if not output_name:  # If we're going to overwrite
                full_df = pd.read_csv(cell_csv)
            else:
                full_df = df  # We already have the necessary columns
                
            # Create or update column
            if label_column not in full_df.columns:
                full_df[label_column] = ""
                
            # Create mask for update
            selected_ids = set(filtered_df['CellID'])
            update_mask = full_df['CellID'].isin(selected_ids) & (full_df['Sample'] == sample)
            
            # Update only selected cells
            full_df.loc[update_mask, label_column] = label_value
            
            # Save results
            if output_name:
                output_path = output_dir / f"{output_name}.csv"
                full_df.to_csv(output_path, index=False)
                show_info(f"Added label to {cell_count} cells. Saved as:\n{output_path}")
            else:
                full_df.to_csv(cell_csv, index=False)
                show_info(f"Added label to {cell_count} cells in original file")

    except Exception as e:
        show_info(f"Error: {str(e)}")

# Connect mode change to show/hide controls
@extract_cells_in_shape.output_mode.changed.connect
def on_output_mode_changed(output_mode: str):
    if output_mode == "Add label to existing":
        extract_cells_in_shape.label_column.show()
        extract_cells_in_shape.label_value.show()
    else:
        extract_cells_in_shape.label_column.hide()
        extract_cells_in_shape.label_value.hide()

# Set initial visibility
extract_cells_in_shape.label_column.hide()
extract_cells_in_shape.label_value.hide()

# -------------------------------------------------------------------------------
# Gating
# -------------------------------------------------------------------------------

@magicgui(
    call_button='Gating',
    layout='vertical'
)
def gate_finder(
    from_gate: int,
    to_gate:int,
    increment: float,
    path_data = Path(),
    marker_of_interest=""
):
    path_data = str(path_data)
    #Load data
    adata = sm.pp.mcmicro_to_scimap(path_data, remove_dna = True, log=False, unique_CellId=True, CellId='CellID', split='X_centroid')
    adata.raw = adata
    #Maked a copy of the data
    #bdata = adata.copy()
    #Generate the dataframe
    data = pd.DataFrame(adata.raw.X, index=adata.obs.index, columns=adata.var.index)[[marker_of_interest]]
    #Apply log transform
    data = np.log1p(data)

    # Generate a dataframe with various gates   
    def gate(g, d):
        dd = d.values
        dd = np.where(dd < g, np.nan, dd)
        # np.warnings.filterwarnings('ignore')
        np.seterr('ignore')
        dd = np.where(dd > g, 1, dd)
        dd = pd.DataFrame(dd, index=d.index, columns=['gate-' + str(g)])
        return dd
    
    # Identify the list of increments
    inc = list(np.arange(from_gate, to_gate, increment))
    inc = [round(num, 3) for num in inc]

    # Apply the function
    r_gate = lambda x: gate(g=x, d=data)  # Create lamda function
    gated_data = list(map(r_gate, inc))  # Apply function

    # Concat all the results into a single dataframe
    gates = pd.concat(gated_data, axis=1)

    # Recover the channel names from adata
    channel_names = []
    for layer in viewer.layers:
        channel_names.append(layer.name)

    # subset the gates to include only the image of interest
    gates = gates.loc[adata.obs.index,]

    def add_phenotype_layer(adata, gates, phenotype_layer, x, y, viewer):
        cells = gates[gates[phenotype_layer] == 1].index
        coordinates = adata[cells]
        # Flip Y axis if needed
        coordinates = pd.DataFrame({'y': coordinates.obs[y], 'x': coordinates.obs[x]})
        # points = coordinates.values.tolist()
        points = coordinates.values
        # import time
        # start = time.time()
        viewer.add_points(
            points,
            size=20,
            face_color='white',
            visible=False,
            name=phenotype_layer,
        )
    x_coordinate='X_centroid'
    y_coordinate='Y_centroid'
    for i in gates.columns:
        add_phenotype_layer(
            adata=adata,
            gates=gates,
            phenotype_layer=i,
            x=x_coordinate,
            y=y_coordinate,
            viewer=viewer
        )


# -------------------------------------------------------------------------------
# Final configuration
# -------------------------------------------------------------------------------

# 1. Define widget mapping
widget_map = {
    "Open image": open_large_image,
    "Open mask": open_mask,
    "Load shapes": load_shapes,
    "Contrast limits": save_contrast_limits,
    "Save shapes": save_shapes,
    "Crop ROI": cut_mask,
    "Count cells": count_selected_cells,
    "Export cells": save_selected_cells,
    "Metadata": view_metadata,
    "Voronoi": voronoi_plot,
    "Load points": load_points,
    "Save Viewport": save_viewport,
    "Close all": close_all,
    "Circle with n cells": create_circle_widget,
    "Extract Cells in Shape": extract_cells_in_shape,
    "Gating": gate_finder
}

# 2. Define tab configuration
tab_config = {
    "Input": ["Open image", "Open mask", "Load shapes", "Load points"],
    "Analysis": [
        "Count cells", "Metadata", "Voronoi", 
        "Circle with n cells", "Extract Cells in Shape", "Gating"
    ],
    "Export": ["Contrast limits", "Save shapes", "Crop ROI", "Save Viewport"],
    "Tools": ["Close all"]
}

# 3. Add widgets to viewer
for tab_name, widgets in tab_config.items():
    tab_widgets = []
    for w_name in widgets:
        if dialog.checkboxes[w_name].isChecked():
            tab_widgets.append((w_name, widget_map[w_name]))
    
    if tab_widgets:
        for w_name, widget in tab_widgets:
            if w_name == "Circle with n cells":
                # Handle special container widget
                viewer.window.add_dock_widget(
                    widget(),
                    name=w_name,
                    area='right',
                    allowed_areas=['right', 'left']
                )
            else:
                # Handle magicgui widgets
                viewer.window.add_dock_widget(
                    widget,
                    name=w_name.replace(" ", "_").lower(),
                    area='right',
                    allowed_areas=['right', 'left']
                )

@magicgui(call_button='⚙️ Configure Widgets')
def config_widgets():
    dialog = SettingsDialog()
    if dialog.exec_():
        viewer.window._dock_widgets.clear()
        for tab_name, widgets in tab_config.items():
            current_widgets = []
            for w_name in widgets:
                if dialog.checkboxes[w_name].isChecked():
                    current_widgets.append((w_name, widget_map[w_name]))
            
            if current_widgets:
                for w_name, widget in current_widgets:
                    if w_name == "Circle with n cells":
                        viewer.window.add_dock_widget(
                            widget(),
                            name=w_name,
                            area='right',
                            allowed_areas=['right', 'left']
                        )
                    else:
                        viewer.window.add_dock_widget(
                            widget,
                            name=w_name.replace(" ", "_").lower(),
                            area='right',
                            allowed_areas=['right', 'left']
                        )

viewer.window.add_dock_widget(config_widgets, area='right')

napari.run()

