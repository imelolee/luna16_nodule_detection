import os
import numpy as np 
import SimpleITK as sitk

subset_id_list = []
root_path = "/home/sharedata/datasets/luna16/LUNG16/subset3"
name_list = os.listdir("/home/sharedata/datasets/luna16/LUNG16/subset3")

for name in name_list:
    pa_id = name[:-4]
    if pa_id not in subset_id_list:
        subset_id_list.append(pa_id)

# print(subset_id_list)

for name in subset_id_list:
    img = sitk.ReadImage(os.path.join(root_path, name + ".mhd"))
    if np.abs(np.array(img.GetDirection()) - np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))).sum() >= 1:
        print(f"id: {name} have wrong direction setting, current dir is {img.GetDirection()}")

