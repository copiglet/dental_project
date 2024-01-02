import subprocess
import os
import glob2
import shutil
import yaml
import time
from pathlib import Path
from distutils.dir_util import copy_tree
from tqdm import trange
from dicom2nifti import dicom_series_to_nifti
import numpy as np
import nibabel as nib
from scipy.ndimage import label, center_of_mass
# DICOM 시리즈 경로 설정

def check_input_dcm(input_path):
    """
    This function has two main checking operation
    (1) Check whether the input dir is empty or not
    (2) Check if each dcm dir contains a non-dcm file

                type        description
    parameter : str         input path
    return    : list, list  normal dcm dir list, removed dcm dir list
    """

    print("======== checking input validation...")

    # input dir`s dcm dir list
    input_set = next(os.walk(input_path))[1]

    # if input directory is empty, program is exit >> (1)
    if not input_set:
        print("input directory is empty!")
        exit()

    # if dcm directory include not dcm, it is continue >> (2)
    delete_set = []
    for inp in input_set:
        dcm_list = os.listdir(os.path.join(input_path, inp))

        print(f"checking... {inp}")
        for index in trange(len(dcm_list)):
            if not dcm_list[index].endswith(".dcm"):
                print(f"{inp} include not dciom file!")
                delete_set.append(inp)
                input_set.remove(inp)
    return [os.path.join(input_path, inp) for inp in input_set], delete_set


def dcm2nii(input_path, nii_path):
    """
    This function convert dcm to nii.gz

                type        description
    parameter : str, str    input path, save path for nii.gz
    return    : None
    """

    print("======== convert dcm to nii.gz ...")

    # get a useful dcm data list
    # if you want to know abnormal dcm dir, print "removed_list"
    input_list, removed_list = check_input_dcm(input_path)

    # convert dcm to nii.gz using dicom2nifti library
    for inp in input_list:
        name = inp.split("/")[-1]
        # converted_path = os.path.join(nii_path, name)
        # if not os.path.exists(converted_path):
        #     os.mkdir(converted_path)
        # dicom_series_to_nifti(inp, os.path.join(nii_path, f"{name}_0000.nii.gz"),delete_nifti=False)
        try:
            dicom_series_to_nifti(inp, os.path.join(nii_path, f"{name}_0000.nii.gz"))
        except:
            print("== dcm2nifty error ===")
            cmd = f"/xutil/dcm2niix -b y -z y -o {nii_path} -f {name+'_0000'} -s n -v n {inp}"
            print("== run :" + cmd)
            os.system(cmd)
            os.system(f"rm -f {nii_path}/{name}*.json")



    print("=========================================")  

def post_process1(mask1, mask2):
    # 두 개의 nii.gz 마스크 파일 로드
    mask1_nii = nib.load(mask1)
    mask2_nii = nib.load(mask2)

    # 데이터 배열 추출
    mask1_data = mask1_nii.get_fdata()
    mask2_data = mask2_nii.get_fdata()

    # 두 마스크에서 겹치는 부분 찾기 (논리적 AND 연산)
    overlap_data = np.logical_and(mask1_data, mask2_data)

    # 새로운 NIfTI 이미지 생성 (헤더와 affine 정보는 첫 번째 마스크에서 복사)
    overlap_nii = nib.Nifti1Image(overlap_data.astype(np.uint8), mask1_nii.affine, mask1_nii.header)

    # 새로운 NIfTI 이미지 저장
    nib.save(overlap_nii, mask2)

def post_process2(input_file, output_file):
    nii_img = nib.load(input_file)
    img_data = nii_img.get_fdata().astype(np.int16)  # Assure data is integer

    # Connected components labeling
    labeled_array, num_features = label(img_data)

    # 레이블별로 마스크의 수와 중심을 찾는다
    component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
    component_centers = center_of_mass(img_data, labeled_array, index=range(1, num_features + 1))

    # 가장 큰 덩어리의 중심을 찾는다
    largest_component_index = np.argmax(component_sizes)
    largest_component_center_z = component_centers[largest_component_index][2]

    # 가장 큰 덩어리의 중심으로부터 z축으로 90 이상 떨어진 덩어리를 제거한다
    for i in range(num_features):
        if abs(component_centers[i][2] - largest_component_center_z) > 205:
            img_data[labeled_array == i+1] = 0

    # 결과를 새로운 NIFTI 파일로 저장한다
    new_nii = nib.Nifti1Image(img_data, nii_img.affine)
    nib.save(new_nii, output_file)

def merge(path, output_file):
    merged_data = None
    output_path = path
    for file in os.listdir(output_path):
        if file.endswith(".nii.gz"):
            # 마스크 로딩
            # print(os.path.join(output_path,file))
            img = nib.load(os.path.join(output_path,file))
            data = img.get_fdata()
            # print(file)
            # 마스크에 label 값 할당
            if file.endswith('RC-Seg_Pred.nii.gz'):
                data = data * 3
                data[data != 0] += 0.1
            elif file.endswith('MERGED-Seg_Pred.nii.gz'):
                data = data * 1
                data[data != 0] += 0.3
            elif file == output_path.split('/')[-1].split('_0000')[0] + '.nii.gz':
                data = data * 2
                data[data != 0] += 0.5
            # else:
            #     data = data * 2
            #     data[data != 0] += 0.5

            # print(np.unique(data))
            if merged_data is None:
                merged_data = data
            else:
                # 합치기
                merged_data = merged_data + data
            # print(np.unique(merged_data))
    b = np.unique(merged_data)
    merged_data[merged_data == 1.3] = 1
    merged_data[merged_data == 2.5] = 4
    merged_data[merged_data == 3.1] = 3
    merged_data[merged_data == 3.8] = 4
    merged_data[merged_data == 4.3] = 2
    merged_data[merged_data == 4.4] = 3
    merged_data[merged_data == 5.6] = 3
    merged_data[merged_data == 6.8] = 4
    merged_data[merged_data == 6.9] = 3
    merged_data[merged_data == 7.4] = 3
    merged_data[merged_data == 9.9] = 3
    merged_img = nib.Nifti1Image(merged_data, img.affine)
    nib.save(merged_img, output_file)

## running

input_path = './input'
nii_path = './nii_input'
start = time.time()
dcm2nii(input_path, nii_path)
os.environ["nnUNet_results"] = os.path.join('./', "Models")
os.environ["MKL_THREADING_LAYER"] = "GNU"
shutil.rmtree(os.path.join(input_path,os.listdir(input_path)[0]))
for ser in os.listdir(nii_path):
    nifti_path = os.path.join(nii_path,ser)
    output_path = os.path.join('./output',ser.split('.n')[0])
    output_file = os.path.join(output_path, ser.split('.n')[0]+'_merged.nii.gz')
    if os.path.isdir(output_path)==False:
        os.mkdir(output_path)
    
    command = [
        'python3', 
        './script/predict_CBCTSeg.py',
        '-i', nifti_path,
        '-o', output_path,
        '-dm', './Models/ALL_MODELS',
        '-ss', 'MAND', 'MAX',
        '-sf', 'False',
        '-vtk', 'False'
    ]

    command2 = [
        'python3', 
        './script/predict_CBCTSeg.py',
        '-i', nifti_path,
        '-o', output_path,
        '-dm', './Models/ALL_MODELS',
        '-hd', 'SMALL',
        '-ss', 'RC',
        '-sf', 'False',
        '-vtk', 'False'
    ]
    
    command3 = [
        'nnUNetv2_predict',
        '-i', nii_path,
        '-o', output_path,
        '-d', 'Dataset080_TOOTH',
        '-c', '3d_lowres',
        '-f', '0'
    ]
    print('Making MANDIBLE And MAXILLA')
    subprocess.run(command)
    print('Making Root Carnal')
    subprocess.run(command2)

    print('Making Tooth')
    subprocess.run(command3)

    print('Post_processing')

    mand_max = os.path.join(output_path, [i for i in os.listdir(output_path) if i.endswith('MERGED-Seg_Pred.nii.gz')][0])
    rc = os.path.join(output_path, [i for i in os.listdir(output_path) if i.endswith('RC-Seg_Pred.nii.gz')][0])
    # tooth = os.path.join(output_path, [i for i in os.listdir(output_path) if i.endswith('RC-Seg_Pred.nii.gz') == False and i.endswith('MERGED-Seg_Pred.nii.gz') == False][0])
    tooth = os.path.join(output_path, ser.split('_0000')[0]+'.nii.gz')

    
    post_process1(mand_max, tooth)

    # post_process2(tooth, tooth)

    post_process1(tooth, rc)

    print('Making Merged')

    print(output_path,",",output_file)
    merge(output_path,output_file)

    shutil.move(nifti_path, os.path.join(output_path,ser))
    print("operation time :", time.time() - start)
