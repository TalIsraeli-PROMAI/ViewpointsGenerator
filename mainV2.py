import os
import sys
import torch
import pytorch3d
import matplotlib.pyplot as plt
import viewpoints_generation


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

angles = [(0, 0), (0, 180), (0, 90), (0, 270), (90, 0), (-90, 0), (35.26, 45.0), (35.26, 135.0), (35.26, -45.0), (35.26, -135.0), (-35.26, 45.0), (-35.26, 135.0), (-35.26, -45.0), (-35.26, -135.0), (20.91, 90.0), (20.91, -90.0), (-20.91, 90.0), (-20.91, -90.0), (69.09, 0.0), (69.09, 180.0), (-69.09, 0.0), (-69.09, 180.0), (0.0, 20.91), (0.0, 159.09), (0.0, -20.91), (0.0, -159.09)]

# Set paths
DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "solid/RNX-J1-000.obj")
#obj_filename = os.path.join(DATA_DIR, "example_mesh/example00.obj")
#obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")


temp = viewpoints_generation.MeshViewpointsModel(obj_filename)
temp.render_viewpoints(angles)

# Load obj file
# mesh = load_objs_as_meshes([obj_filename], device=device) # loading a mesh with materials
obj_name = obj_filename.split('/')[-1].split('.')[0]
mesh = viewpoints_generation.load_object_without_material(obj_filename, [1,1,1], device)
viewpoints_generation.normalize_mesh(mesh)

cameras, lights, renderer = viewpoints_generation.create_camera_and_rendering(2.5, 3072, viewpoints_generation.Shaders.HardFlat, device)
example = renderer(mesh)
viewpoints_generation.save_rendered_as_image(example, "./generated_images/_.png")

#viewpoints_images = viewpoints.create_view_points(mesh, renderer, lights, device, angles)
#image_grid(viewpoints_images.cpu().numpy(), rows=4, cols=5, rgb=True)

_viewpoints = viewpoints_generation.render_view_points(mesh, renderer, lights, device, angles, viewpoints_generation.LightMode.Fixed)
counter = 0
for image in _viewpoints:
    viewpoints_generation.save_rendered_as_image(image, "./generated_images_2/_" + str(counter) + ".png")
    counter = counter + 1

"""
counter = 0
for image in viewpoints_images:
    viewpoints.save_rendered_as_image(image, "./generated_images/_" + str(counter) + ".png")
    counter = counter + 1"""
#plt.show()
