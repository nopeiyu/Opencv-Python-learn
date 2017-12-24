opencv-Python-learn
============================
项目是我学习opencv-python的过程中记录的笔记，前期主要是基础api的使用，后期会涉及一些高级功能的研究。文献来自网络，主要参考了[open-python官网](http://opencv-python-tutroals.readthedocs.io/en/latest/)和OpenCV-Python-Toturial-中文版.pdf这本官网文档的翻译，错误之处欢迎批评指正！

[TOC]


1. 环境安装  
--------------------------------------------------
###1.1 anaconda python2.7安装 

> opencv-python用到了numpy等科学计算的python库，涉及机器学习的内容还要安装很多相关库，这里推荐安装anaconda这个软件，安装过程很简单，直接官网下载安装，记下安装目录

###1.2 opencv 2.4.13安装  
opencv-python的安装，这里介绍两种安装途径。
 
#### 1.2.1 使用exe安装包

> [opencv安装包下载](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.13/opencv-2.4.13.4-vc14.exe/download),下载之后安装到一个目录，将~\opencv\build\python\2.7\x64\cv2.pyd这个文件复制到~\Anaconda2\Lib\site-packagesq文件夹下就行了

#### 1.2.2 安装文件安装

> [opencv-python安装包下载](https://pypi.python.org/pypi/opencv-python)，下载安装包之后，用pip install ~/opencv_python-3.3.1.11-cp27-cp27m-win_amd64.whl命令行安装

安装完成之后，打开python环境，`import cv2`如果没错误就安装成功了
 
2. 利用Knn和Svm训练识别手写数字  
--------------------------------------------------
>1.图片位置img/digits.png，为50行100列共5000个手写数字  
>2.涉及的代码为digits.py、common.py  
>3.环境安装完成后直接运行就行了
