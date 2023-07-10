import os
import viewpoints_generation

"""
This script will create a thumbnail of the input dataset at a parent directory called ""
IMPORTANT: The script assume the dataset is already at the format of:
|- DATASET_NAME
    |- Dataset
        |- [label_folders]
"""
angles = [(0, 0),  (0, 45), (0, 90), (0, 180), (0, 270), (90, 0), (45, 0), (225, 0), (45, 45)]  # these are the angles that are optimized for a thumbnail

# All the datasets we have PROCESSED with an .obj format
processed_dataset_output_a = r"C:/3D Datasets/MCB_A_Merged"
processed_dataset_output_b = r"C:/3D Datasets/MCB_B_Merged"
cadnet_path = r"C:\3D Datasets\CADNET_OBJ"
abc_path = r"C:\3D Datasets\ABC_OBJ"


def generate_thumbnails_for_dataset(dataset_path, override=True):
    """
    Creates a folder named "Thumbnails" that contains the labels with images of the models in various angles
    :param dataset_path: string, absolute path of the dataset, assuming it has 'Dataset' folder with the labels inside
    :param override: boolean, it will create a new image for files that exist if true, and skip if false
    :return: None
    """
    models_path = dataset_path + "/Dataset"
    if not os.path.isdir(dataset_path + "/Thumbnails"):
        os.makedirs(dataset_path + "/Thumbnails")
    for root, dirs, files in os.walk(os.path.abspath(models_path), topdown=False):
        for label in dirs:
            destination_path = dataset_path + '/Thumbnails' + '/' + label
            filenames = next(os.walk(models_path + "/" + label), (None, None, []))[2]  # [] if no file
            for filename in filenames:
                if os.path.exists(destination_path + "/" + (models_path + "/" + label + "/" + filename).split('/')[-1].split('.')[0] + '_merged' + '.png') and not override:
                    print("Skip " + filename)
                    continue
                vpm = viewpoints_generation.MeshViewpointsModel(
                    models_path + "/" + label + "/" + filename,
                    destination_path, 256, viewpoints_generation.PhotoMode.Merge)
                print("Generating viewpoints for " + vpm.mesh_name)
                vpm.render_viewpoints(angles)


def generate_thumbnails_for_unlabeled_dataset(dataset_path, override=True):
    """
    Generates the thumnails for a dataset that has a single file in each folder, usually in this struct:
    |- Dataset
        |- Item#1
            Item#1.obj
        |- Item#2
            Item#2.obj
        etc...
    The thumbnails folder will be created at the parent directory
    :param dataset_path: string, the parent directory of the dataset
    :param override: boolean, states if to override files that exists
    :return: None
    """
    models_path = dataset_path
    if not os.path.isdir(dataset_path + "/Thumbnails"):
        os.makedirs(dataset_path + "/Thumbnails")
    for root, dirs, files in os.walk(os.path.abspath(models_path), topdown=False):
        for model in dirs:
            destination_path = dataset_path + '/Thumbnails'
            filenames = next(os.walk(models_path + "/" + model), (None, None, []))[2]  # [] if no file
            for filename in filenames:  # most of the time there will be one file
                if os.path.exists(destination_path + "/" + (models_path + "/" + model + "/" + filename).split('/')[-1].split('.')[0] + '_merged' + '.png') and not override:
                    print("Skip " + filename)
                    continue
                vpm = viewpoints_generation.MeshViewpointsModel(
                    models_path + "/" + model + "/" + filename,
                    destination_path, 256, viewpoints_generation.PhotoMode.Merge)
                print("Generating viewpoints for " + vpm.mesh_name)
                vpm.render_viewpoints(angles)


generate_thumbnails_for_unlabeled_dataset(abc_path, False)
