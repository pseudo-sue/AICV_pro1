# 멋쟁이사자처럼 AI CV BOOTCAMP Plus 2기 Project1

## 개요

안녕하세요! 멋쟁이사자처럼 AI CV BOOTCAMP Plus 2기 Project1의 2팀 입니다.

저희는 자율주행의 전방영상 객체인식에 적합한 모델을 개발을 목표로 프로젝트를 진행하였습니다.

RT-DETRv2를 활용해 YOLO 이상의 성능을 만들기 위해 구조 변경를 변경하였습니다.

데이터의 전처리, 모델의 구조 변경, 학습, 평가 모두 진행하였습니다.

## 데이터셋

kittit 데이터를 사용하였습니다.

[https://www.kaggle.com/datasets/klemenko/kitti-dataset](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fklemenko%2Fkitti-dataset)

## 평가지표

* mAP&mAR50-95 -> all,small,medium,large
* Inference time per image
* Num of Parameters

## 환경

* Google Colab T4

## 모델 수정사항

* **Vim 인코더 적용**
* **UpSampling 적용**

## 코드

[colab 코드](./COLAB_TOOL/README.md)
