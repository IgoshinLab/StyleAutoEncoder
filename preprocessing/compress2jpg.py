import os
import cv2
import numpy as np
from multiprocessing import Pool
import pandas as pd

class ProcessImage:
    def __init__(self, base_dir, new_dir, resize_by=1/5):
        self.base_dir = base_dir
        self.new_dir = new_dir
        self.resize_by = resize_by

    def process(self, img_name):
        outfile = img_name.split('.')[0] + '.jpg'
        if os.path.exists(os.path.join(self.new_dir, outfile)):
            # print("file : " + img_name + " exists!")
            return None
        # print("file : " + img_name)
        img_path = os.path.join(self.base_dir, img_name)
        read = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        read = cv2.GaussianBlur(read, (5, 5), 0)  # Remove the grids

        if self.resize_by != 1:
            shape = np.array([read.shape[1] * self.resize_by, read.shape[0] * self.resize_by], dtype=int)
            read = cv2.resize(read, shape, cv2.INTER_LANCZOS4)

        cv2.imwrite(os.path.join(self.new_dir, outfile), read, [int(cv2.IMWRITE_JPEG_QUALITY), 60])



def filterImages(base_path, new_path, label_sheet, resize_by=1/5):
    df = pd.read_excel(label_sheet, sheet_name='Sheet1')
    run_passed = {}
    for index, row in df.iterrows():
        if not isinstance(row["Error"], str):
            run_idx = int(os.path.basename(row["Image"]).split('_')[0][-4:])
            scope_idx = int(os.path.basename(row["Image"]).split('scope')[1].split('-')[0])
            if run_idx not in run_passed:
                run_passed[run_idx] = [scope_idx]
            else:
                run_passed[run_idx].append(scope_idx)
    os.makedirs(new_path, exist_ok=True)
    infolders = os.listdir(base_path)
    p = Pool(10)
    while infolders:
        infolder = infolders.pop()
        if not os.path.isdir(os.path.join(base_path, infolder)):
            continue
        elif os.path.basename(infolder).startswith("Scope"):
            run_idx = int(os.path.dirname(infolder)[-4:])
            print("%s,%s" % (os.path.dirname(infolder), os.path.basename(infolder)))
            scope_idx = int(os.path.basename(infolder)[5:])
            if run_idx in run_passed:
                if scope_idx in run_passed[run_idx]:
                    previous_path = os.path.join(base_path, infolder)
                    current_path = os.path.join(new_path, infolder)
                    os.makedirs(current_path, exist_ok=True)
                    infiles = [img_file for img_file in os.listdir(previous_path) if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'))]
                    processor = ProcessImage(previous_path, current_path, resize_by=resize_by)
                    p.map(processor.process, infiles)
        else:
            subfolders = os.listdir(os.path.join(base_path, infolder))
            infolders_ = [os.path.join(infolder, subfolder) for subfolder in subfolders]
            infolders.extend(infolders_)


# file_dir = "/mnt/WelchLab/Test_Movies"
#
# filterImages(file_dir, '/media/xavier/SHGP31/dataset/Welch/trainingset2/test',
#              "/mnt/WelchLab/Test_Movies/Movie_list.xlsx", resize_by=.5)

filterImages("/mnt/WelchLab/Zethus",
             '/home/xavier/Documents/dataset/Welch/Zethus',
             "/mnt/WelchLab/Zethus/Movie_list.xlsx", resize_by=.5)

import xlsxwriter

def prepare_data_sheet(root_dir, out_dir):

    iter_list = [os.path.join(root_dir, file_dir) for file_dir in os.listdir(root_dir)]
    img_list = []
    while iter_list:
        current_dir = iter_list.pop()
        if current_dir.endswith("1400.tif"):
            img_list.append(current_dir)
        elif os.path.isdir(current_dir):
            iter_list.extend([os.path.join(current_dir, file_dir) for file_dir in os.listdir(current_dir)])
    print(img_list)
    workbook = xlsxwriter.Workbook(out_dir)
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, "Image")
    worksheet.write(0, 1, "Phenotype")
    worksheet.write(0, 2, "Error")
    worksheet.write(0, 3, "Other")

    row = 1
    col = 0
    for img_name in img_list:
        worksheet.write(row, col, os.path.basename(img_name))
        worksheet.write(row, col + 1, "Test")
        row += 1

    workbook.close()


#prepare_data_sheet("/mnt/WelchLab/Zethus", "/mnt/WelchLab/Zethus/Movie_list.xlsx")