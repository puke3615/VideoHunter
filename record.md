## 基于深度学习的视频人脸检测

[TOC]

### 前期调研

网上查看相关论文、博客、github源码

### 方案选型

> 选择原则
>
> * 时间成本
>
>   完成视频中人脸检测项目，时间进度上还是很紧张的，确定方案要参考时间限制
>
> * 投入成本
>
>   偏向短期时间节点内能快速上手且使用的方式
>
> * 检测效率
>
>   视频中包含大量的帧画面，所以对于检测的速度方面选择尽可能快速的

网上查看相关检测的资料，选择了各个方面综合能力较好的预选方案，并做出对比。

|       模型       |  速度  |  精度  |  成本  | 小物体  | 训练周期 |
| :------------: | :--: | :--: | :--: | :--: | :--: |
| `OpenCV + CNN` |  快   |  一般  |  一般  |  一般  |  短   |
|  `YOLO/YOLO2`  |  一般  |  高   |  高   |  较低  |  长   |
|     `SSD`      |  一般  |  高   |  高   |  高   |  长   |



### 数据收集

#### Pubfig数据

[处理脚本](scripts/pubfig/preprocess.py)

由于版权问题，数据源是以链接的形式存在，于是下载脚本对链接图片进行下载，下载过程很多图片无法下载，无奈放弃该数据源

#### 爱情公寓

[处理脚本](scripts/video/reader.py)

* OpenCV对视频的每一帧进行人脸检测
* 将检测区域抠下来保存为人脸图片的数据源
* 对该部分数据源进行手动分类

### 模型训练

[CNN分类模型](v1/classifier.py)

[SSD检测模型](https://github.com/Machine-Learning-For-Research/ssd_keras)

### 模型测试

[图片版测试](v1/image.py)

[视频版测试](v1/video.py)

![](doc/images/1.gif)



*参考链接*

* [R-CNN、SPP-NET、Fast-R-CNN、Faster-R-CNN、YOLO、SSD总结](http://blog.csdn.net/eli00001/article/details/52292095)
* [Pubfig数据源](http://www.cs.columbia.edu/CAVE/databases/pubfig/)
* [OpenCv的视频处理相关Api](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html?highlight=videocapture)
* [基于Haar Cascades的人脸检测](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection)
* [基于深度学习的视频检测](http://blog.csdn.net/relar/article/details/51926078)

