# 멋쟁이사자처럼 AI CV BOOTCAMP Plus 2기 Project1 코드

## 코드

[ipynb](./RT-DETRv2_with_vim_n_upsampling.ipynb)

## 사용법

아래의 코드를 colab에 업로드해서 활용하시면 됩니다.

git hub에 필요한 코드, 설정파일을 모두 업로드하여 그대로 실행하시면 됩니다.

결과물은 pth파일이 로컬 PC에 다운로드 됩니다.

## 환경설정

```python
# -c에 설정파일
!cd /content/AICV_pro1/rtdetrv2_pytorch &&export PYTHONPATH=$PWD:$PYTHONPATH && \
torchrun --nproc_per_node=1 --master_port=29500 \
  tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_vim_up_50e_kitti_T4.yml \
  --use-amp \
  --seed 42
```

colab에서 사용하는 하드웨어에 따라 설정파일을 바꿀 수 있습니다.

사용하는 하드웨어가 T4라면 -c configs/rtdetrv2/rtdetrv2_r18vd_vim_up_50e_kitti_T4.yml

사용하는 하드웨어가 L4라면 -c configs/rtdetrv2/rtdetrv2_r18vd_vim_up_50e_kitti_L4.yml

을 사용하면 됩니다.
