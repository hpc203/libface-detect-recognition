# 一种适合工业级应用的基于深度学习的实时人脸检测与识别算法的C++实现
# 本套程序仅仅只依赖opencv库，就可以正常运行
模型文件在百度网盘里，下载链接是
 https://pan.baidu.com/s/1Cvw9925YhwsJosNB9fd5MQ 提取码: ev9g

下载完成后，把models文件夹与代码放在同一目录下，配置好opencv环境后就可以运行程序了。
我这边用的是opencv4.4.0，建议安装这个最新版的opencv库，不然就可能在程序运行时出现异常中断的。
在主函数里提供了4个功能，分别是：

(1).输入一幅图片，做人脸检测。

(2).输入一个文件夹，提取批量图片的人脸特征向量，然后把人脸特征向量保存为bin文件。

(3).读取人脸特征向量的bin文件，输入一幅图片，检测人脸并提取特征向量，然后计算特征向量的距离，做人脸识别。

(4).输入一幅图片，检测人脸，然后使用pfld网络检测人脸98个关键点

我的程序是在win10系统里运行的，如果切换到linux系统下运行，那么在执行第3个功能时，里面有遍历文件夹里的所有文件和目录的函数getAllFiles。
而这个函数在windows和linux里的代码实现是有所不同的。因此，如果你想在linux系统里运行人脸识别程序，那么需要按照代码里的注释说明去屏蔽一些代码打开另一些代码。
在windows系统里，路径分隔符可以是"/"或者"\\"，在Linux系统里，路径分隔符是"/"，为了保证系统的兼容性，在本程序里，
路径分隔符统一使用"/"

这套程序里的人脸检测和人脸特征向量提取模块已经久远了，在2021年在11月6日，我在github发布了使用OpenCV部署libface人脸检测和SFace人脸识别，
包含C++和Python两种版本的程序，仅仅只依赖OpenCV库就能运行。源码地址是：
https://github.com/hpc203/libface-sface_detect-recognition-opencv
