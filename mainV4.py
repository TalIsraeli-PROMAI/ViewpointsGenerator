import os
import viewpoints_generation

angles = [(0, 0), (0, 180), (0, 90), (0, 270), (90, 0), (-90, 0), (35.26, 45.0), (35.26, 135.0), (35.26, -45.0), (35.26, -135.0), (-35.26, 45.0), (-35.26, 135.0), (-35.26, -45.0), (-35.26, -135.0), (20.91, 90.0), (20.91, -90.0), (-20.91, 90.0), (-20.91, -90.0), (69.09, 0.0), (69.09, 180.0), (-69.09, 0.0), (-69.09, 180.0), (0.0, 20.91), (0.0, 159.09), (0.0, -20.91), (0.0, -159.09)]

# Set paths
DATA_DIR = "./data"
#obj_filename = os.path.join(DATA_DIR, "solid/RNX-J1-000.obj")
obj_filename = os.path.join(DATA_DIR, "example_mesh/example00.obj")
#obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

all_meshes = ['cow_mesh/cow.obj']



for mesh_path in all_meshes:
    try:
        abs_path = os.path.join(DATA_DIR, mesh_path)
        vpm = viewpoints_generation.MeshViewpointsModel(abs_path, r"./generated_images/", 128, viewpoints_generation.PhotoMode.Vector)
        print("Generating viewpoints for " + vpm.mesh_name)
        vpm.render_viewpoints(angles)
    except FileNotFoundError:
        print("Specified file not found: " + abs_path)

