"""
Script for modifying the name of the
"""

from glob import glob
import os


data_path = "/home/jm/Documents/research/self-driving-car/donkeycar/mycar/data/tub_combine_2/tub_2_19-12-22/"

for filename in os.listdir(data_path):
    src = data_path + filename
    dst = data_path + "b" + filename

    # rename() function will
    # rename all the files
    os.rename(src, dst)
