# show_attend_and_tell
![](/imgs/图像标题生成模型结构图.png)
本项目是用tensorflow实现的show attend and tell算法，vgg19网络使用的是tensorflow.contrib.slim包实现的，程序运行时需要加载预先训练好的vgg19模型，
连接是https://pan.baidu.com/s/1erk9oJEdGVxDWyWeQSiTjA  vgg_model.zip<br>
单个图片生成文本只需要运行Single_image_Catpion.ipynb，并加载caption.ckpt模型文件，链接是:<br>
https://pan.baidu.com/s/1KHIau63ruWu8afB0l_6W-w 文件名称是：mysave_model.zip<br>
批量进行图像生成文本只需运行the_test.ipynb，并加载tfrecords文件 https://pan.baidu.com/s/1eeLELbSoR9FABaunlRL07g  tfrecords.zip。<br>
tfrecords来源于flickr30数据集，一个30000多张图像，img转tfrecords文件只需要运行test_convert_tfrecord.ipynb，flickr30图像数据及其标注连接是:<br>
https://pan.baidu.com/s/1FJVwFpM5XThXqB1fPsBfNg  data.zip<br>
压缩文件按照项目文件.JPG进行解压缩。

