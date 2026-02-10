## Dependencies needed
import napari
from napari import Viewer
from napari.utils.notifications import show_info
import pandas as pd
import numpy as np
import random
import scimap as sm 
import tifffile
from tifffile import imread
import dask.array as da
import zarr
import os
import ast
import matplotlib.pyplot as plt
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
import gudhi as gd
from shapely.geometry import Point, Polygon
from napari.layers import Shapes


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
            "Circle with n cells", "Extract Cells in Shape","Tag cells", 
            "Gating", "Run phenotype calling", "Build Rips Complex", "Close all"  # Fixed comma
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
        with tifffile.TiffFile(image_path) as tf:
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
    seg_m = tifffile.imread(mask_path)
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

                # Crear listas para todos los shapes del archivo
                all_shapes = []
                all_shape_types = []
                
                for shape in shape_list:
                    s_type = shape.get("type", "polygon")
                    vertices = np.array(shape.get("vertices", []), dtype=np.float32)

                    if len(vertices) == 0:
                        continue
                    
                    all_shapes.append(vertices)
                    all_shape_types.append(s_type)

                if all_shapes:  # Solo agregar si hay shapes válidos
                    viewer.add_shapes(
                        all_shapes,
                        shape_type=all_shape_types,
                        edge_width=1,
                        edge_color='#777777',
                        face_color='red',
                        name=filename.stem  # 🔹 SOLO el nombre del archivo sin sufijos
                    )

                show_info(f"Loaded {len(all_shapes)} shapes from {filename.name} (JSON)")

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
                    name=filename.stem  # 🔹 SOLO el nombre del archivo sin sufijos
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
        tifffile.imwrite(output_path, viewport)
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
        from napari.layers import Shapes, Image    
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
# Widget implementation – Extract Cells in Shape (Fast Optimized Version)
# -------------------------------------------------------------------------------


@magicgui(
    call_button='Extract Cells in Shape',
    layout='vertical',
    sample={"label": "Sample Name"},
    cell_csv={"label": "Cell Data CSV", "mode": "r", "filter": "*.csv"},
    shape_name={"label": "Shape Layer", "choices": lambda _: [ly.name for ly in viewer.layers if isinstance(ly, Shapes)]},
    output_mode={"label": "Output Mode", "choices": ["New CSV", "Add label to existing"]},
    label_column={"label": "Column Name", "visible": False},
    label_value={"label": "Annotation", "visible": False},
    output_dir={"label": "Output Directory", "mode": "d"},
    output_name={"label": "New File Name (Optional)"}
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
    """Extract cell data within a drawn shape, using fast polygon-based filtering."""
    try:
        # 1️⃣ Validación de entradas
        if not cell_csv.exists():
            show_info("Cell CSV file not found")
            return

        shape_layer = next((ly for ly in viewer.layers if ly.name == shape_name and isinstance(ly, Shapes)), None)
        if shape_layer is None:
            show_info(f"Shape layer '{shape_name}' not found")
            return

        if not shape_layer.selected_data:
            show_info("Please select a shape in the Shapes layer.")
            return

        # 2️⃣ Lectura optimizada del CSV
        use_cols = ['X_centroid', 'Y_centroid', 'CellID', 'Sample']
        dtypes = {
            'X_centroid': 'float32',
            'Y_centroid': 'float32',
            'CellID': 'int32',
            'Sample': 'category'
        }
        df = pd.read_csv(cell_csv, usecols=use_cols, dtype=dtypes)

        if sample not in df['Sample'].unique():
            show_info(f"No cells found for sample: {sample}")
            return

        sample_df = df[df['Sample'] == sample]

        # 3️⃣ Obtener polígono (solo el primero seleccionado)
        shape_data = shape_layer.data[next(iter(shape_layer.selected_data))]
        polygon = Polygon(shape_data[:, [1, 0]])  # (x, y) order para Shapely

        # 4️⃣ Verificación de pertenencia vectorizada
        points = MultiPoint(np.column_stack((sample_df['X_centroid'], sample_df['Y_centroid'])))
        mask = np.fromiter((polygon.contains(p) for p in points), dtype=bool)
        filtered_df = sample_df.loc[mask]
        cell_count = len(filtered_df)

        if cell_count == 0:
            show_info("No cells found within the specified shape")
            return

        # 5️⃣ Exportar resultados
        if output_mode == "New CSV":
            output_path = output_dir / f"{output_name or f'{sample}_ROI'}.csv"
            filtered_df.to_csv(output_path, index=False)
            show_info(f"Saved {cell_count} cells from sample '{sample}' to:\n{output_path}")

        else:  # Add label to existing
            full_df = pd.read_csv(cell_csv)
            if label_column not in full_df.columns:
                full_df[label_column] = ""

            selected_ids = set(filtered_df['CellID'])
            mask_update = (full_df['Sample'] == sample) & full_df['CellID'].isin(selected_ids)
            full_df.loc[mask_update, label_column] = label_value

            output_path = output_dir / f"{output_name or cell_csv.stem}_labeled.csv"
            full_df.to_csv(output_path, index=False)
            show_info(f"Added label to {cell_count} cells. Saved as:\n{output_path}")

    except Exception as e:
        show_info(f"Error: {str(e)}")


# -------------------------------------------------------------------------------
# Visibilidad dinámica de parámetros
# -------------------------------------------------------------------------------

@extract_cells_in_shape.output_mode.changed.connect
def on_output_mode_changed(output_mode: str):
    if output_mode == "Add label to existing":
        extract_cells_in_shape.label_column.show()
        extract_cells_in_shape.label_value.show()
    else:
        extract_cells_in_shape.label_column.hide()
        extract_cells_in_shape.label_value.hide()

# Estado inicial
extract_cells_in_shape.label_column.hide()
extract_cells_in_shape.label_value.hide()


# -------------------------------------------------------------------------------
# Tag cells
# -------------------------------------------------------------------------------

@magicgui(
    call_button='Tag cells',
    layout='vertical', output_dir={"mode": "d"}
)
def tag_cells(
    output_dir: Path,
    shape_layer_name:str,
    sample_name:str,
    tag_column:str,
    tag:str,
    file_path = Path(),
    image_path = Path(),
    output_name: str = ""
):
    from PIL import Image, ImageDraw
    file_path = str(file_path)
    image_path = str(image_path)

    #Read data
    data = pd.read_csv(file_path)

    #Subset data
    sample_data = data[['X_centroid', 'Y_centroid', 'CellID']][
        data['Sample'] == sample_name].astype(int)

    #Turn coordenates to list
    sample_data['tuple'] = list(
        zip(sample_data['X_centroid'],
            sample_data['Y_centroid'])
    )
    #Load single channel pyramid image
    tiff = tifffile.TiffFile(image_path)
    if 'Faas' not in tiff.pages[0].software:
        if len(tiff.series[0].levels) > 1:
            dna = [zarr.open(s[0].aszarr(), mode='r') for s in tiff.series[0].levels]
            dna = [da.from_zarr(z) for z in dna]
            min_val = dna[-1].min()
            max_val = dna[-1].max()
        else:
            img = tiff.pages[0].asarray()
            dna = [img[::4**i, ::4**i] for i in range(4)]
            dna = [da.from_array(z) for z in dna]
            min_val = dna[-1].min()
            max_val = dna[-1].max()
        max_val = max(max_val, min_val + 1)
        vmin, vmax = da.compute(min_val, max_val)
    else:  # support legacy OME-TIFF format
        if len(tiff.series) > 1:
            dna = [zarr.open(s[0].aszarr()) for s in tiff.series]
            dna = [da.from_zarr(z) for z in dna]
            min_val = dna[-1].min()
            max_val = dna[-1].max()
        else:
            img = tiff.pages[0].asarray()
            dna = [img[::4**i, ::4**i] for i in range(4)]
            dna = [da.from_array(z) for z in dna]
            min_val = dna[-1].min()
            max_val = dna[-1].max()
        max_val = max(max_val, min_val + 1)
        vmin, vmax = da.compute(min_val, max_val)
        

    # create pillow image to convert into boolean mask
        
    img = Image.new(
        'L', (dna[0].shape[1], dna[0].shape[0]))      

    shapes_layer = viewer.layers[shape_layer_name]
    shape_types = shapes_layer.shape_type
    vertices = shapes_layer.data

    layer_data = list(zip(shape_types, vertices))

    for shapes, verts in layer_data:
        # snap any floating point verts to array
        selection_verts = np.round(verts).astype(int)  
        vertices = list(zip(selection_verts[:, 1],
                            selection_verts[:, 0])
        )

        # update pillow image with polygon
        ImageDraw.Draw(img).polygon(
            vertices, outline=1, fill=1)

    # convert pillow image into boolean numpy array
    ROI_mask = np.array(img, dtype=bool)

    # use numpy fancy indexing to get centroids
    # where boolean mask is True
    xs, ys = zip(*sample_data['tuple'])
    inter1 = ROI_mask[ys, xs]
    sample_data['inter1'] = inter1
    idxs_to_tag= list(sample_data['CellID'][sample_data['inter1']])

    # tag cells in column

    if tag_column not in data.columns:
        data[tag_column] = '' 

    if len(idxs_to_tag) > 0: #for cell_ids in idxs_to_tag.items()
        global_idx_to_tag = data[(data['Sample'] == sample_name)&
                            (data['CellID'].isin(idxs_to_tag))].index
        data.loc[global_idx_to_tag, tag_column] = tag

    output_path = output_dir / f"{output_name}.csv"
    data.to_csv(output_path, index=False)

# -------------------------------------------------------------------------------
# Gating
# -------------------------------------------------------------------------------

@magicgui(
    call_button='Gating',
    layout='vertical'
)
def gate_finder(
    from_gate: float,
    to_gate:float,
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
        dd = pd.DataFrame(dd, index=d.index, columns=[marker_of_interest + '_gate-' + str(g)])
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
# Phenotype cells
# -------------------------------------------------------------------------------

@magicgui(
    call_button='Run phenotype calling',
    output_directory={"mode": "d"},
    layout='vertical'
)
def phenotype_cells(
                    phenotype_threshold_percent: int = 0,
                    phenotype_threshold_abs: int = 0,
                    phenotype_label: str = "phenotype",
                    image_ID_column: str = "Sample",
                    path_data= Path(),
                    phenotype_key= Path(),
                    manual_gating =  bool,
                    gate_value: float = 0.5,
                    gating_file = Path(),
                    print_phenotype_proportions: bool = True,
                    save_phenotype_proportions: bool = True,
                    sample_name = "",
                    output_directory: Path = Path(),

):
    phenotype_key = str(phenotype_key)
    path_data = str(path_data)
    gating_file = str(gating_file)

    #Load the data and generate de anndata object
    adata = sm.pp.mcmicro_to_scimap(path_data, remove_dna = True, 
                                    log=False, unique_CellId=True, 
                                    CellId='CellID', split='X_centroid')
    
    #Load the phenotype key
    phenotype = pd.read_csv(phenotype_key, sep = ",")
    
    #Load the gating file
    if manual_gating is False:
        #Rescale the data
        adata = sm.pp.rescale(adata, gate = None, imageid = image_ID_column, verbose = True)
        sm.tl.phenotype_cells(adata, phenotype, gate=gate_value, label=phenotype_label, imageid=image_ID_column, 
                              pheno_threshold_percent=phenotype_threshold_percent, 
                              pheno_threshold_abs=phenotype_threshold_abs, verbose=True)
    else: 
        manual_gate = pd.read_csv(gating_file, sep = ",")
        #Rescale the data
        adata = sm.pp.rescale(adata, gate = manual_gate, imageid = image_ID_column, verbose = True)
        sm.tl.phenotype_cells(adata, phenotype, gate=gate_value, label=phenotype_label, 
                              imageid=image_ID_column,pheno_threshold_percent=phenotype_threshold_percent,
                              pheno_threshold_abs=phenotype_threshold_abs, verbose=True)
    
    #Generate csv with cell type proportions
    a = pd.DataFrame(adata.obs['phenotype'].value_counts())
    a.reset_index(inplace=True)
    a = a.rename(columns = {'index':'type'})

    #Print plot with celltype proportions
    if print_phenotype_proportions is True:
        # Calculate proportions
        a["proportion"] = a["count"] / a["count"].sum()

        # Create barplot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(a["phenotype"], a["proportion"], color=plt.cm.tab20.colors)

        # Add labels above bars (show number of cells)
        for bar, count in zip(bars, a["count"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    str(count), ha='center', va='bottom', fontsize=10)
        # Style
        ax.set_ylabel("Proportion of cells")
        ax.set_xlabel("Phenotype")
        ax.set_title("Phenotype proportions " + sample_name)
        ax.set_ylim(0, a["proportion"].max() + 0.1)  # some space for labels
        plt.xticks(rotation=45, ha = "right")
        plt.tight_layout()
        plt.show()

    #Save plot and csv file with celltype proportions
    if save_phenotype_proportions is True:
        plt.savefig(output_directory / f"{sample_name}_phenotype_proportions.png", dpi=300, bbox_inches="tight")
        a.to_csv(output_directory / f"{sample_name}_phenotype_proportions.csv", index= False)
        
    sm.hl.scimap_to_csv(adata, layer='raw', output_dir=output_directory, 
                        file_name= sample_name + "_phenotype_annotated", 
                        CellID='CellID', verbose=True)
    show_info(
            f"Phenotype calling complete!"
        )




# -------------------------------------------------------------------------------
# Rips complex (Improved version - with persistence diagram)
# -------------------------------------------------------------------------------

# --- Helper functions ----------------------------------------------------------

def order_polygon_vertices(coords: np.ndarray) -> np.ndarray:
    """Order polygon vertices counterclockwise based on polar angle."""
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    return coords[np.argsort(angles)]

def polygon_area(coords: np.ndarray) -> float:
    """Compute polygon area using the Shoelace formula."""
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    """Get convex hull vertices for higher-dimensional simplices."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except:
        # Fallback: return ordered vertices
        return order_polygon_vertices(points)

def get_shape_layer_names(widget=None) -> list:
    """Get available shape layer names from the current viewer."""
    try:
        viewer = napari.current_viewer()
        if viewer is None:
            return []
        return [layer.name for layer in viewer.layers 
                if isinstance(layer, Shapes) and len(layer.data) > 0]
    except Exception:
        return []

def filter_cells_in_shape(csv_path: Path, shape_layer_name: str, sample_name: str = None) -> pd.DataFrame:
    """Filter cells that are within the selected shape."""
    try:
        # Load cell data
        df = pd.read_csv(csv_path)
        
        # Verify required columns
        required_cols = {"CellID", "X_centroid", "Y_centroid"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Filter by sample if provided
        if sample_name and "Sample" in df.columns:
            df = df[df["Sample"] == sample_name]
            if len(df) == 0:
                raise ValueError(f"No cells found for sample: {sample_name}")
        
        # Get the shape layer
        viewer = napari.current_viewer()
        if viewer is None:
            raise ValueError("No Napari viewer available")
        
        shape_layer = None
        for layer in viewer.layers:
            if layer.name == shape_layer_name and isinstance(layer, Shapes):
                shape_layer = layer
                break
        
        if shape_layer is None:
            raise ValueError(f"Shape layer '{shape_layer_name}' not found")
        
        if len(shape_layer.data) == 0:
            raise ValueError("No shapes found in the selected layer")
        
        # Use the first shape
        shape_data = shape_layer.data[0]
        
        # Create polygon from shape data - note the coordinate order
        # Napari uses (row, col) but our data uses (X, Y) = (col, row)
        polygon = Polygon([(point[1], point[0]) for point in shape_data])
        
        # Filter cells within the polygon
        cells_inside = []
        for idx, row in df.iterrows():
            point = Point(row['X_centroid'], row['Y_centroid'])
            if polygon.contains(point):
                cells_inside.append(idx)
        
        filtered_df = df.loc[cells_inside].copy()
        
        if len(filtered_df) == 0:
            raise ValueError("No cells found within the selected shape")
        
        show_info(f"Found {len(filtered_df)} cells within shape")
        return filtered_df
        
    except Exception as e:
        show_info(f"Error filtering cells: {str(e)}")
        return pd.DataFrame()

# --- Main widget ---------------------------------------------------------------

@magicgui(
    call_button="Build Rips Complex",
    csv_path={"label": "Cell Data CSV", "mode": "r", "filter": "*.csv"},
    shape_layer={
        "label": "Shape Layer", 
        "choices": get_shape_layer_names
    },
    sample_name={
        "label": "Sample Name (Optional)",
        "tooltip": "If CSV contains multiple samples, specify which one to use"
    },
    radius={"label": "Connection radius", "min": 1.0, "max": 200.0, "step": 1.0},
    max_dim={"label": "Max dimension", "min": 1, "max": 5, "tooltip": "1=edges, 2=triangles, 3+=higher-dimensional simplices"},
    output_path={"label": "Output CSV Path", "mode": "w", "filter": "*.csv"},
    show_points={"label": "Show points", "tooltip": "Display cell centroids"},
    show_edges={"label": "Show edges", "tooltip": "Display 1-simplices (connections)"},
    show_triangles={"label": "Show triangles", "tooltip": "Display 2-simplices"},
    show_loops={"label": "Show loops", "tooltip": "Display higher-dimensional simplices (dimension > 2)"},
    loop_opacity={"label": "Loop opacity", "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Opacity for higher-dimensional simplices"},
    generate_persistence_diagram={"label": "Generate Persistence Diagram", "tooltip": "Create and display persistence diagram"}
)
def rips_widget(
    csv_path: Path,
    shape_layer: str,
    sample_name: str = "",
    radius: float = 20.0,
    max_dim: int = 3,
    output_path: Path = Path("rips_complex_results.csv"),
    show_points: bool = True,
    show_edges: bool = True,
    show_triangles: bool = True,
    show_loops: bool = True,
    loop_opacity: float = 0.15,
    generate_persistence_diagram: bool = True
) -> pd.DataFrame:
    """Build Rips complex from cells within a selected shape and save results to specified path."""
    
    # === 1. Filter cells within the shape ===
    filtered_df = filter_cells_in_shape(csv_path, shape_layer, sample_name)
    if filtered_df.empty:
        return pd.DataFrame()

    # === 2. Prepare points ===
    points = filtered_df[["Y_centroid", "X_centroid"]].to_numpy()
    
    # === 3. Build Rips complex ===
    try:
        rips = gd.RipsComplex(points=points, max_edge_length=radius)
        st = rips.create_simplex_tree(max_dimension=max_dim)

        # Calculate persistence for the diagram
        persistence = st.persistence()
        
        simplices = []
        edges = []
        triangles = []
        loops = []
        loop_dimensions = []

        for simplex, filt in st.get_filtration():
            dim = len(simplex) - 1
            
            # Skip degenerate simplices
            if len(set(simplex)) != len(simplex):
                continue

            # Record simplex information
            cell_ids = ";".join(str(filtered_df.iloc[i]["CellID"]) for i in simplex)
            coords = ";".join(
                f"({filtered_df.iloc[i]['Y_centroid']:.1f},{filtered_df.iloc[i]['X_centroid']:.1f})" 
                for i in simplex
            )

            simplices.append({
                "simplex": tuple(simplex),
                "dimension": dim,
                "filtration_value": round(filt, 4),
                "CellIDs": cell_ids,
                "coords": coords
            })

            # Collect geometric representations
            if dim == 1 and show_edges:
                edge_coords = points[list(simplex)]
                edges.append(edge_coords)

            elif dim == 2 and show_triangles:
                tri_coords = points[list(simplex)]
                tri_coords = order_polygon_vertices(tri_coords)
                if polygon_area(tri_coords) > 1e-6:  # Avoid degenerate triangles
                    triangles.append(tri_coords)

            elif dim >= 3 and show_loops:
                # For higher-dimensional simplices, create convex hull or ordered polygon
                loop_coords = points[list(simplex)]
                try:
                    # Try to get convex hull for better visualization
                    hull_coords = get_convex_hull_vertices(loop_coords)
                    if len(hull_coords) >= 3:  # Need at least 3 points for a polygon
                        loops.append(hull_coords)
                        loop_dimensions.append(dim)
                except:
                    # Fallback: use ordered vertices
                    ordered_coords = order_polygon_vertices(loop_coords)
                    loops.append(ordered_coords)
                    loop_dimensions.append(dim)

        simplices_df = pd.DataFrame(simplices)

    except Exception as e:
        show_info(f"Error building Rips complex: {str(e)}")
        return pd.DataFrame()

    # === 4. Save results ===
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        simplices_df.to_csv(output_path, index=False)
        
        # Also save the filtered cell data for reference
        filtered_cells_path = output_path.parent / f"{output_path.stem}_filtered_cells.csv"
        filtered_df.to_csv(filtered_cells_path, index=False)
        
    except Exception as e:
        show_info(f"Error saving results: {str(e)}")
        return simplices_df  # Return data even if save fails

    # === 5. Generate Persistence Diagram ===
    if generate_persistence_diagram:
        try:
            # Create persistence diagram
            plt.figure(figsize=(8, 6))
            gd.plot_persistence_diagram(persistence)
            base_name = Path(csv_path).stem
            plt.title(f"Persistence Diagram: {base_name}\nRadius: {radius}, Max Dim: {max_dim}")
            
            # Save the diagram
            diagram_path = output_path.parent / f"{output_path.stem}_persistence_diagram.png"
            plt.savefig(diagram_path, dpi=600, bbox_inches='tight')
            plt.show()  # Display the diagram
            #plt.close()
            
            show_info(f"Persistence diagram saved to: {diagram_path}")
            
        except Exception as e:
            show_info(f"Error generating persistence diagram: {str(e)}")

    # === 6. Visualization in Napari ===
    try:
        viewer = napari.current_viewer()
        if not viewer:
            show_info("No Napari viewer available")
            return simplices_df

        base_name = f"ShapeROI_r{radius}"

        # Points layer
        if show_points and len(points) > 0:
            viewer.add_points(
                points, 
                name=f"{base_name}_Cells", 
                size=8, 
                face_color="yellow",
                edge_color="black",
                opacity=0.8
            )

        # Edges layer
        if show_edges and edges:
            viewer.add_shapes(
                edges, 
                shape_type="line", 
                edge_color="cyan", 
                edge_width=2,
                name=f"{base_name}_Connections"
            )

        # Triangles layer  
        if show_triangles and triangles:
            viewer.add_shapes(
                triangles, 
                shape_type="polygon",
                edge_color="magenta", 
                face_color="magenta",
                edge_width=1.5,
                opacity=0.3, 
                name=f"{base_name}_Triangles"
            )

        # Loops layer (higher-dimensional simplices)
        if show_loops and loops:
            # Create color map based on dimension
            dimension_colors = {
                3: [0.2, 0.8, 0.2, loop_opacity],  # Green for 3D
                4: [0.8, 0.4, 0.0, loop_opacity],  # Orange for 4D  
                5: [0.6, 0.2, 0.6, loop_opacity],  # Purple for 5D
            }
            
            # Group loops by dimension for better visualization
            for dim in set(loop_dimensions):
                dim_loops = [loop for loop, d in zip(loops, loop_dimensions) if d == dim]
                color = dimension_colors.get(dim, [0.5, 0.5, 0.5, loop_opacity])
                
                viewer.add_shapes(
                    dim_loops,
                    shape_type="polygon",
                    edge_color=[c * 0.8 for c in color[:3]] + [1.0],  # Brighter edges
                    face_color=color,
                    edge_width=2.0,
                    name=f"{base_name}_Dim{dim}_Simplices"
                )

        # Summary information
        loop_counts = {}
        for dim in loop_dimensions:
            loop_counts[dim] = loop_counts.get(dim, 0) + 1
        
        loop_info = "\n".join([f"• Dim {dim}: {count} simplices" 
                              for dim, count in sorted(loop_counts.items())])
        
        persistence_info = ""
        if generate_persistence_diagram:
            persistence_info = f"• Persistence diagram: {diagram_path.name}\n"
        
        show_info(
            f"Rips complex built successfully!\n"
            f"• Cells in shape: {len(filtered_df)}\n"
            f"• Connections: {len(edges)}\n" 
            f"• Triangles: {len(triangles)}\n"
            f"• Higher-dimensional simplices:\n{loop_info if loop_counts else '   None'}\n"
            f"{persistence_info}"
            f"• Results saved to: {output_path.name}\n"
            f"• Filtered cells saved to: {filtered_cells_path.name}"
        )

    except Exception as e:
        show_info(f"Error during visualization: {str(e)}")

    return simplices_df

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
    "Tag cells": tag_cells,
    "Gating": gate_finder,
    "Run phenotype calling": phenotype_cells,
    "Build Rips Complex": rips_widget
}

# 2. Define tab configuration
tab_config = {
    "Input": ["Open image", "Open mask", "Load shapes", "Load points"],
    "Analysis": [
        "Count cells", "Metadata", "Voronoi", 
        "Circle with n cells", "Extract Cells in Shape", "Tag cells", "Gating", "Run phenotype calling", "Build Rips Complex"
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

