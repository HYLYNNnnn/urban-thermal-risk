# 导入需要使用的包
import csv
import os
from itertools import islice
import mxnet as mx
from mxnet import image, gpu
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import cv2

file_path = './pic'
out_path='./out'
filelist_out = os.listdir(out_path)
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
del_file('./image_resize')
del_file('./image_processed')

# 关于图片重构的尺寸参数
# 如果图像的长宽 大于 max_size 则需要重构。
max_size=1500
# reshape_size 为重构后的尺寸
reshape_size=1000

# 忽略警告
import warnings; warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore")

# 设定使用GPU或者CUP进行计算，没有安装GPU版本的MXNet请使用CPU ctx = mx.gpu(0)
ctx = mx.cpu(0)

# 读取标签信息 并 作为输出结果的列
def ReadLable():
    col_map_dict={}
    filepath='./lable.csv'
    with open(filepath, encoding='GBK') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # osm_id highway bridge angle Lng Lat ID
        for row in islice(reader, 1, None):
            id=int(row[0])-1
            Name=row[4]
            col_map_dict[id]=Name
    return col_map_dict

col_map =ReadLable()
# 定义函数对单张图片进行图像分割，并将结果存为pd.Series
def get_seg(file, model):
    img = image.imread(file)
    img = test_transform(img,ctx=ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    pred = []
    for i in range(150):
        pred.append((len(predict[predict==i])/(predict.shape[0]*predict.shape[1])))
    pred = pd.Series(pred).rename(col_map)
    return pred

model = gluoncv.model_zoo.get_model('psp_resnet101_ade',ctx=ctx,pretrained=True )

filelist = os.listdir(file_path)
col=['id','lng','lat','pixels']
for k in col_map.keys():
    col.append(col_map[k])
df = pd.DataFrame(columns=col)
print(df)

# 图片重构后的分辨率:需要根据原始图片的分辨率比例进行设置
for i in filelist: # 循环遍历所有的图片进行语义分割，并将结果存如pd.DataFrame
    # i 是图片名
    if i not in filelist_out:
        img_path = os.path.join(file_path, i)
        img_id = i
        # 图片的经纬度信息和朝向信息 如果图片名称中有这些信息，则可从图片名称 i 中解析。
        lng = 0
        lat = 0

        img = cv2.imread(img_path)
        # 读取完后将路径改成临时路径
        img_path=img_path.replace('pic','image_processed')
        # 获取原始图片尺寸
        ori_size=[img.shape[1],img.shape[0]]
        # 如果图片尺寸过大则进行图片的重构
        if ori_size[0]>max_size:
            Scale_Factor=ori_size[0]/reshape_size
            img_size = (int(ori_size[0]/Scale_Factor), int(ori_size[1]/Scale_Factor))
            print(i,ori_size,'Resize to:',img_size)
        else:
            img_size = (int(ori_size[0]), int(ori_size[1]))

        # img2 = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)INTER_CUBIC

        cv2.imwrite(img_path, img2)

        pixels = img_size[0] * img_size[1]
        data_i = pd.Series({'id': img_id, 'lng': lng, 'lat': lat, 'pixels': pixels}).append(get_seg(img_path, model))
        new_col = pd.DataFrame(data_i).T
        df = pd.concat([df, new_col], axis=0, join='outer', ignore_index=True)

        img = image.imread(img_path)
        copy_img =cv2.imread(img_path)
        img = test_transform(img,ctx=ctx)
        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        # 将predict进行处理，如果不是特定类别则直接转换位白色
        size=predict.shape
        for i in range(size[0]):
            for j in range(size[1]):
                this_type=int(predict[i,j])
                ######### 如果不是特定的类别 #########
                # 植物：4 tree;9 grass; 18 plant,flora,plant,life;
                # 水体：60 river;113 waterfall,,falls;21 water;26 sea;
                obj_types=[1]
                # id-1 即为对应的数值
                if this_type not in obj_types:
                    copy_img[i, j] = [255, 255, 255]
                else:
                    ##########  赋予特定的颜色 #########
                    copy_img[i, j] = img2[i,j]

        # copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(out_path,img_id),copy_img)
        # 输出结果的路径 csv文件
        df.to_csv("./only_seg_one_ade20k.csv")  # 将结果保存到csv
    else:
        print(i,'已经存在！')
