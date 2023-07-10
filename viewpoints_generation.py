"""
Written by Tal Israeli
This module uses pytorch3d to render a 3d model into different viewpoints and export it as an image
The rendering is done by pytorch3d's renderer, thus is running on the GPU - the higher the resolution - more GPU
is needed to compute the rendering process. Each machine should define its own maximum supported resolution.
"""
import logging
import os
import sys
import warnings
import math
import torch
import pytorch3d
import numpy as np
from PIL import Image
from enum import Enum
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, PointLights, DirectionalLights, Materials, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, HardFlatShader, TexturesUV, TexturesVertex
)

OUTPUT_PATH = r"./generated_images/"
DEFAULT_CAMERA_DISTANCE = 2.2
DEFAULT_RESOLUTION = 2048


class Shaders(Enum):
    SoftPhong = 0
    HardFlat = 1


class LightMode(Enum):
    Fixed = 0
    BeforeMesh = 1


class PhotoMode(Enum):
    Single = 0  # Saves all the viewpoint images as a single file
    Vector = 1  # Saves all the viewpoint images as a vector (1, len(viewpoints)
    Merge = 2  # Saves all the viewpoint images as a squared image (the viewpoints should be a squarable number)


class MeshViewpointsModel:
    """
    This class represents a single model and its viewpoints, each model must have its own class.
    Instantiating the class with a model that has been running before will override old angles and append new ones.
    """
    def __init__(self, path_to_model, path_to_output=OUTPUT_PATH, output_resolution=DEFAULT_RESOLUTION,
                 photo_mode=PhotoMode.Single):
        """
        :param path_to_model: string, the absolute path for the 3d model
        :param path_to_output: string, path to the output folder
        :param output_resolution: int, the output resolution: [intxint]
        :param photo_mode: PhotoMode, states how to save the images. See PhotoMode class for more info
        """
        self.__init_device()
        self.mesh = load_object_without_material(path_to_model, [1, 1, 1], self.device)
        self.mesh_name = path_to_model.split('/')[-1].split('.')[0]
        self.photo_mode = photo_mode
        self.save_path = path_to_output
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        normalize_mesh(self.mesh)
        self.cameras, self.lights, self.renderer = create_camera_and_rendering(
            DEFAULT_CAMERA_DISTANCE, output_resolution, Shaders.SoftPhong, self.device)

    def __init_device(self):
        # Device Setup
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
            warnings.warn("Pytorch3D device is set to cpu! it may result in slow rendering!")
        self.device = device

    def render_viewpoints(self, angles_array, light_mode=LightMode.Fixed):
        viewpoints = render_view_points(self.mesh, self.renderer, self.lights, self.device, angles_array, light_mode)
        if self.photo_mode is PhotoMode.Merge:
            save_rendered_as_merged(viewpoints, self.save_path + "/" + self.mesh_name + '_merged' + '.png')
        elif self.photo_mode is PhotoMode.Vector:
            save_rendered_as_vector(viewpoints, self.save_path + "/" + self.mesh_name + '_vector' + '.png')
        else:  # PhotoMode = Single
            for v in viewpoints:
                save_rendered_as_image(v['image'],
                                       self.save_path + "/" + self.mesh_name + "_" + str(v['elev']) + "_" + str(
                                           v['azim']) + '.png')


def load_object_without_material(path, color, torch_device):
    """
    loading obj file and generating a default material
    :param path: string - the absolute path to the 3d model
    :param color: [float,float,float] represents rgb values
    :param torch_device: device
    :return: mesh
    """
    verts, faces_idx, _ = load_obj(path, load_textures=False)
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]
    for vert in verts_rgb[0]:
        vert[0] = color[0]  # r
        vert[1] = color[1]  # g
        vert[2] = color[2]  # b
    textures = TexturesVertex(verts_features=verts_rgb.to(torch_device))

    return Meshes(verts=[verts.to(torch_device)], faces=[faces.to(torch_device)], textures=textures)


def normalize_mesh(mesh):
    """
    Normalize the scale of the mesh to fit a sphere with a radius of 1, centered at (0,0,0).
    The operations are done on the mesh sent, thus the function returns nothing
    :param mesh: any mesh, but a mesh already normalized has no effect
    :return: None
    """
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_((-center))
    mesh.scale_verts_((1.0 / float(scale)))


def create_camera_and_rendering(dist_from_mesh, square_resolution, shader_type, torch_device):
    """
    :param dist_from_mesh: float, dist from the mesh in the space
    :param square_resolution: int, resolution of the output image (R x R)
    :param shader_type: Shader Enum, the type of the shader to use.
    :param torch_device: torch device
    :return: camera, lights, renderer
    """
    if type(shader_type) is not Shaders:
        warnings.warn("shader_type must be of type viewpoints.Shaders. default shader HardFlat will be used")
        shader_type = Shaders.HardFlat
    R, T = look_at_view_transform(dist_from_mesh, 0, 0)
    cameras = FoVPerspectiveCameras(device=torch_device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=square_resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0  # necessary for meshes with alot of triangles and vertices
    )
    lights = PointLights(device=torch_device, location=[[0.0, 0.0, +1.0]])
    if shader_type is Shaders.HardFlat:
        shader = HardFlatShader(device=torch_device, cameras=cameras, lights=lights)
    else:
        shader = SoftPhongShader(device=torch_device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    return cameras, lights, renderer


def save_rendered_as_image(rendered_image, dest_save_path):
    """
    saves the rendered output
    :param rendered_image:  [1, res_x, res_y, 4] OR [res_x, res_y, 4] arr, the result tensor of rendering
    :param dest_save_path: string of the path to save
    """
    if rendered_image.shape[0] != 1:  # [res_x, res_y, 4] shape
        rendered_image = rendered_image[None, :, :, :]
    np_image = rendered_image[0, ..., :3].cpu().numpy()
    image = Image.fromarray((np_image * 255).astype(np.uint8))
    image.save(dest_save_path)


def save_rendered_as_merged(rendered_images, dest_save_path):
    """
    Concats all the images to a single image with a squared (W x H, W=H) resolution.
    :param rendered_images: tensor of rendered images, which are supposed to be concated
    :param dest_save_path: string, the path where the image should be saved
    :return: None
    """
    images_for_grid = []
    resolution = 0
    for viewpoint in rendered_images:
        render = viewpoint['image'] # get the actual image from the viewpoint dict
        if render.shape[0] != 1:  # [res_x, res_y, 4] shape
            render = render[None, :, :, :]
        np_image = render[0, ..., :3].cpu().numpy()
        resolution = np_image.shape[1]
        image = Image.fromarray((np_image * 255).astype(np.uint8))
        images_for_grid.append(image)
    grid_size = math.ceil(math.sqrt(len(images_for_grid)))
    canvas = Image.new('RGB', (resolution * grid_size, resolution * grid_size), (250, 250, 250))
    counter = 0
    for col in range(0, grid_size):
        for row in range(0, grid_size):
            if counter < len(images_for_grid):
                canvas.paste(images_for_grid[counter], (row * resolution, col * resolution))
            counter += 1
    canvas.save(dest_save_path)


def save_rendered_as_vector(rendered_images, dest_save_path):
    """
    Concats all the images to a single image as a flat vector with a resolution of W x H:
    W = (single image width) * (number of images)
    H = (single image height)
    This function is good for creating viewpoints for a neuron network input
    :param rendered_images: tensor of rendered images, which are supposed to be concated
    :param dest_save_path: string, the path where the image should be saved
    :return: None
    """
    images_for_vector = []
    resolution = 0
    for viewpoint in rendered_images:
        render = viewpoint['image'] # get the actual image from the viewpoint dict
        if render.shape[0] != 1:  # [res_x, res_y, 4] shape
            render = render[None, :, :, :]
        np_image = render[0, ..., :3].cpu().numpy()
        resolution = np_image.shape[1]
        image = Image.fromarray((np_image * 255).astype(np.uint8))
        images_for_vector.append(image)
    canvas = Image.new('RGB', (resolution * len(rendered_images), resolution), (250, 250, 250))
    for i in range(0, len(images_for_vector)):
        canvas.paste(images_for_vector[i], (i*resolution, 0))
    canvas.save(dest_save_path)


def render_view_points(mesh, renderer, lights, torch_device, angles, lightmode=LightMode.Fixed):
    """
    creates an array of images that represent a mesh from different viewpoints.
    Important: this is done by iterating the mesh and the viewpoints. Another method can be
    done by batching which is better for machine learning outputs
    :param mesh: the mesh of an object we want to collect the viewpoints for
    :param renderer: the pytorch3d renderer (rasterizer + shader) to render the model with
    :param lights: light, lightning object
    :param torch_device: device
    :param angles: array of the angles in polar coordinates for camera initialization. [elev, azim]
    :param lightmode: LightMode enum, states the lightning on the model
    :return: array of the rendered images and its angles as a dictionary: {image:image, elev:float, azim:float}.
    The angles are used to define in which angle the image is taken, and specify the image file name.
    In case of unused angle, a new file will be created, thus not overriding existing image file
    """
    # prepare data for rendering
    viewpoints_array = []
    elev = []
    azim = []
    for angle in angles:
        elev.append(angle[0])
        azim.append(angle[1])
    for i in range(len(angles)):
        image = render_single_view_point(mesh, renderer, elev[i], azim[i], lights, torch_device)
        viewpoint_dict = {'image': image, 'elev': elev[i], 'azim': azim[i]}
        viewpoints_array.append(viewpoint_dict)
    return viewpoints_array


def render_single_view_point(mesh, renderer, elev, azim, light, torch_device):
    """
    Renders a single viewpoint as an image
    :param mesh: mesh to performing rendering on
    :param renderer: pre-built renderer
    :param elev: elevation of the camera
    :param azim: azimuth of the camera
    :param light: lights for the scene
    :param torch_device: device
    :return: a tensor contains the image
    """
    R, T = look_at_view_transform(dist=DEFAULT_CAMERA_DISTANCE, elev=elev, azim=azim)
    camera = FoVPerspectiveCameras(device=torch_device, R=R, T=T)
    viewpoint = renderer(mesh, cameras=camera, lights=light)
    return viewpoint


def create_view_points(mesh, renderer, lights, torch_device, angles, lightmode=LightMode.Fixed):
    """
    creates an array of images that represent a mesh from different viewpoints
    $legacy: this can be done with batching
    :param mesh: the mesh of an object we want to collect the viewpoints for
    :param renderer: the pytorch3d renderer (rasterizer + shader) to render the model with
    :param lights: light, lightning object
    :param torch_device: device
    :param angles: array of the angles in polar coordinates for camera initialization. [elev, azim]
    :param lightmode: LightMode enum, states the lightning on the model
    :return: rendered images array
    """
    batch_size = len(angles)  # the number of different viewpoints from which we want to render the mesh.

    # Create a batch of meshes by repeating the original mesh.
    meshes = mesh.extend(batch_size)
    # Create the cameras, a new camera for each viewpoint
    elev = []
    azim = []
    for angle in angles:
        elev.append(angle[0])
        azim.append(angle[1])
    R, T = look_at_view_transform(dist=DEFAULT_CAMERA_DISTANCE, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=torch_device, R=R, T=T)
    images = renderer(meshes, cameras=cameras, lights=lights)
    return images
