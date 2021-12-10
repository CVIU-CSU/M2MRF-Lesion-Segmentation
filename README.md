# M2MRF: Many-to-Many Reassembly of Features for Tiny Lesion Segmentation in Fundus Images

This repo is the official implementation of paper ["M2MRF: Many-to-Many Reassembly of Features for Tiny Lesion Segmentation in Fundus Images"](https://arxiv.org/abs/2111.00193v2).

<!-- ## Introduction -->

## Environment

This code is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

-   pytorch=1.6.0
-   mmsegmentation=0.8.0
-   mmcv=1.2.0

```
conda create -n m2mrf python=3.7 -y
conda activate m2mrf

conda install pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch -y
pip install mmcv-full==1.2.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html -i https://pypi.douban.com/simple/
pip install opencv-python
pip install scipy
pip install tensorboard tensorboardX
pip install sklearn
pip install terminaltables
pip install matplotlib

cd M2MRF-Lesion-Segmentation
chmod u+x tools/*
pip install -e .
```

## Training and testing

```
# prepare dataset
python tools/prepare_labels.py
python tools/augment.py

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 tools/dist_train.sh configs/m2mrf/fcn_hr48-M2MRF-C_40k_idrid_bdice.py 4

# test
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=12345 tools/dist_test.sh configs/m2mrf/fcn_hr48-M2MRF-C_40k_idrid_bdice.py /path/to/fcn_hr48-M2MRF-C_40k_idrid_bdice_iter_40000.pth 4 --eval mIoU
```

## Results and models

We evaluate our method on [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) and [DDR](https://github.com/nkicsl/DDR-dataset).

### IDRiD

| method  | &nbsp;&nbsp;mIOU&nbsp;&nbsp; | mAUPR | download                                                                                                                                                       |
| ------- | :--------------------------: | :---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M2MRF-A |            49.86             | 67.15 | [config](configs/m2mrf/fcn_hr48-M2MRF-A_40k_idrid_bdice.py) &#124; [model](https://drive.google.com/file/d/1rRN4-X0HDwa0srJaEKodzyQOOxuYLXaQ/view?usp=sharing) |
| M2MRF-B |            49.33             | 66.71 | [config](configs/m2mrf/fcn_hr48-M2MRF-B_40k_idrid_bdice.py) &#124; [model](https://drive.google.com/file/d/1tERKxM_qnbJ3L261g_CaPdRgyuJbdhj5/view?usp=sharing) |
| M2MRF-C |            50.17             | 67.55 | [config](configs/m2mrf/fcn_hr48-M2MRF-C_40k_idrid_bdice.py) &#124; [model](https://drive.google.com/file/d/11YoorrgNds960WTNypDs4qissjgoLZd1/view?usp=sharing) |
| M2MRF-D |            49.96             | 67.32 | [config](configs/m2mrf/fcn_hr48-M2MRF-D_40k_idrid_bdice.py) &#124; [model](https://drive.google.com/file/d/1LkwmrtHEuahCMR1dJxSBPfiUCPf4uZDz/view?usp=sharing) |

### DDR

| method  | &nbsp;&nbsp;mIOU&nbsp;&nbsp; | mAUPR | download                                                                                                                                                     |
| ------- | :--------------------------: | :---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| M2MRF-A |            31.47             | 49.56 | [config](configs/m2mrf/fcn_hr48-M2MRF-A_60k_ddr_bdice.py) &#124; [model](https://drive.google.com/file/d/1HhZ5Ur3ZT-28nUzQRWjtlh5b1Cy4b9Lb/view?usp=sharing) |
| M2MRF-B |            30.56             | 49.86 | [config](configs/m2mrf/fcn_hr48-M2MRF-B_60k_ddr_bdice.py) &#124; [model](https://drive.google.com/file/d/1tza-ck_gX7k654FY6YVT71Pp4NIrEeHK/view?usp=sharing) |
| M2MRF-C |            30.39             | 49.20 | [config](configs/m2mrf/fcn_hr48-M2MRF-C_60k_ddr_bdice.py) &#124; [model](https://drive.google.com/file/d/1wFd6a4boC541ORL1Lz04t0MJsvf_8Skz/view?usp=sharing) |
| M2MRF-D |            30.76             | 49.47 | [config](configs/m2mrf/fcn_hr48-M2MRF-D_60k_ddr_bdice.py) &#124; [model](https://drive.google.com/file/d/1Evbixr3V6GTACCo48xdb4tbe95BFNCm3/view?usp=sharing) |

In the paper, we reported average performance over three repetitions, but our code only reported the best one among them.

## Citation

If you find this code useful in your research, please consider cite:

```latex
@misc{liu2021m2mrf,
      title={M2MRF: Many-to-Many Reassembly of Features for Tiny Lesion Segmentation in Fundus Images},
      author={Qing Liu and Haotian Liu and Wei Ke and Yixiong Liang},
      year={2021},
      eprint={2111.00193},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
