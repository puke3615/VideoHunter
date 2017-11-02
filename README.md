## 基于深度学习的视频人脸检测

*这里旨在说明开发前大致准备工作，未记录详细开发过程，可参考 [项目Commit记录](https://github.com/puke3615/VideoHunter/commits/master)*

[TOC]

### 前期调研

网上查看相关论文、博客、github源码

### 方案选型

> 选择原则
>
> - 时间成本
>
>   完成视频中人脸检测项目，时间进度上还是很紧张的，确定方案要参考时间限制
>
> - 投入成本
>
>   偏向短期时间节点内能快速上手且使用的方式
>
> - 检测效率
>
>   视频中包含大量的帧画面，所以对于检测的速度方面选择尽可能快速的

网上查看相关检测的资料，选择了各个方面综合能力较好的预选方案，并做出对比。

|       模型       |  速度  |  精度  |  成本  | 小物体  | 训练周期 |
| :------------: | :--: | :--: | :--: | :--: | :--: |
| `OpenCV + CNN` |  快   |  一般  |  一般  |  一般  |  短   |
|  `YOLO/YOLO2`  |  一般  |  高   |  高   |  较低  |  长   |
|     `SSD`      |  一般  |  高   |  高   |  高   |  长   |

由于人脸在视频画面中相对较小，因此这里没有选择在小物体检测上有劣势的`YOLO`模型。`OpenCV + CNN`的方式精度上虽然不高，但是处理速度还是蛮快的，在实时检测方面效果不错，于是将该方案作为兜底方案。`SSD`模型检测是目前调研出的各方面相对表现都比较优异的，作为最终选择方案。

即：选择`OpenCV + CNN`作为兜底方案，`SSD`作为最终方案。

### 数据收集

#### Pubfig数据

[处理脚本](scripts/pubfig/preprocess.py#L128)

由于版权问题，数据源是以链接的形式存在，于是下载脚本对链接图片进行下载，下载过程很多图片无法下载，无奈放弃该数据源。

#### 爱情公寓

[处理脚本](scripts/video/reader.py#L71)

- OpenCV对视频的每一帧进行人脸检测
- 将检测区域抠下来保存为人脸图片的数据源
- 对该部分数据源进行手动分类

### 模型训练

#### [CNN分类模型](v1/classifier.py#L85)

由于人脸的图片比较小，再加上类别数量不多，所以这里用的是小型的CNN网络，结构如下

```python
def build_model(self):
	model = Sequential()
	model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(16, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(8, activation='softmax'))
	return model
```

#### [SSD检测模型](https://github.com/Machine-Learning-For-Research/ssd_keras)

`SSD`模型是在[基于keras的SSD开源项目](https://github.com/Machine-Learning-For-Research/ssd_keras)进行改造的，主要替换训练数据的载入方式和输出的类别。

### 模型测试

[图片版测试](v1/image.py)

![](doc/images/test2_prediction.jpg)

[视频版测试](v1/video.py)	[点击下载演示视频](result.mp4)





*参考链接*

- [R-CNN、SPP-NET、Fast-R-CNN、Faster-R-CNN、YOLO、SSD总结](http://blog.csdn.net/eli00001/article/details/52292095)
- [Pubfig数据源](http://www.cs.columbia.edu/CAVE/databases/pubfig/)
- [OpenCv的视频处理相关Api](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html?highlight=videocapture)
- [基于Haar Cascades的人脸检测](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection)
- [基于深度学习的视频检测](http://blog.csdn.net/relar/article/details/51926078)