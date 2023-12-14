# Splits images into train and test folders

from glob import glob
import os
import random

random.seed(16)

gt_base = "/home/zlai/data/SIDD_S_sRGB/GT/"
n_base = "/home/zlai/data/SIDD_S_sRGB/N/"

gt_train_path = gt_base + "train/"
gt_test_path = gt_base + "test/"
n_train_path = n_base + "train/"
n_test_path = n_base + "test/"

new_base = "/home/zlai/data/SIDD_S_sRGB/"

dir_list = [gt_test_path, gt_train_path, n_test_path, n_train_path]
new_dir_list = [new_base + "val/GT/", new_base + "train/GT/", new_base + "val/N/", new_base + "train/N/", new_base + "test/GT/", new_base + "test/N/"]

for dir in new_dir_list:
    if not os.path.isdir(dir):
        os.makedirs(dir)

# sort so same order- same names
gt_files = sorted(glob(gt_base + "train/*.PNG") + glob(gt_base + "test/*.PNG"))
n_files = sorted(glob(n_base + "train/*.PNG") + glob(n_base + "test/*.PNG"))

# Zip the list of files together and convert them to a list
file_pairs = list(zip(gt_files, n_files))

# Shuffle the list of pairs
random.shuffle(file_pairs)

total_samples = len(file_pairs)
train_samples = int(total_samples * 0.6)
val_samples = train_samples + int(total_samples * 0.2)

for idx, (gt, n) in enumerate(file_pairs): 
    gt = gt.replace("\\", "/")
    n = n.replace("\\", "/")

    print("Moving files {} and {}".format(gt, n))

    if idx <= train_samples:
        # Write files to train folders
        gt_name = gt.split("/")[-1]
        os.rename(gt, new_base + "train/GT/" + gt_name)

        n_name = n.split("/")[-1]
        os.rename(n, new_base + "train/N/" + n_name)
    elif idx > train_samples and idx <= val_samples:
        gt_name = gt.split("/")[-1]
        os.rename(gt, new_base + "val/GT/" + gt_name)

        n_name = n.split("/")[-1]
        os.rename(n, new_base + "val/N/" + n_name)
    else:
        # Write files to the test folders
        gt_name = gt.split("/")[-1]
        os.rename(gt, new_base + "test/GT/" + gt_name)

        n_name = n.split("/")[-1]
        os.rename(n, new_base + "test/N/" + n_name)

'''
[scene-instance-number]_[scene_number]_[smartphone-camera-code]_[ISO-level]_[shutter-speed]_[illuminant-temperature]_[illuminant-brightness-code]

"smartphone-camera-code" is one of the following:
GP: Google Pixel
IP: iPhone 7
S6: Samsung Galaxy S6 Edge
n6: Motorola nexus 6
G4: LG G4

"illuminant-brightness-code" is one of the following:
L: low light
n: normal brightness
H: high exposure

'''

# Bayer pattern color filter for each camera:
# bayer = {"GP": "bggr", "IP": "rggb", "S6": "grbg", "n6": "bggr", "G4": "bggr"}