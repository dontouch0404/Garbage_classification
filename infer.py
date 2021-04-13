from dataset import Garbage_Loader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from torchvision import models
import torch.nn as nn
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
#%matplotlib inline
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 将输出映射为（0，1）之间的值
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x

# 读取txt将小标签与大标签对应，如[['其他垃圾_PE塑料袋', '0', '0'], ...]
with open('dir_label.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    labels = list(map(lambda x:x.strip().split('\t'), labels))


def computeTestSetAccuracy(test_loader, model):
    '''
       测试代码
       参数：
            test_loader - 测试集的 DataLoader
            model - 模型
    '''
    # 切换到评估模式
    model.eval()
    acc = 0
    for i, (image, label) in enumerate(test_loader):
        src = image.numpy()
        src = src.reshape(3, 224, 224)
        src = np.transpose(src, (1, 2, 0))
        image = image.cuda()
        label = label.cuda()
        pred = model(image)  # outputs = model(inputs)

        # 从pred到pred_id
        pred = pred.data.cpu().numpy()[0]
        score = softmax(pred)
        pred_id = np.argmax(score)  # 返回score数组中最大值的索引
        ex_id = (test_data.get_images(test_list))[i][1]  # 当前测试图片的小标签对应的下标
        # 若预测成功，则acc+1
        if (labels[int(ex_id)][0] == labels[pred_id][0]):
            acc += 1
        # 显示图片
        # plt.imshow(src)
        # plt.show()
        print("原始标签：{:s}，模型预测结果：{:s}".format(labels[int(ex_id)][0], labels[pred_id][0]))
    # 计算准确率
    avg_test_acc = acc / test_loader.__len__()
    print("Test accuracy：" + str(avg_test_acc))

    '''         Test accuracy：0.8283793347487615          '''



def predict(model, test_image_dir):
    '''
       预测单个图片代码
       参数：
            model - 模型
            test_image_dir - 测试图片的路径
    '''
    img = Image.open(test_image_dir).convert('RGB')  # 打开图片并将图片转换为RGB格式
    img = img.resize((244,244))
    # 展示图片
    plt.imshow(img)
    plt.show()
    # 把PIL.Image转换为tensor，并归一化到(0,1)
    transf = transforms.ToTensor()
    img = transf(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False) # 添加一个通道数
    # 评估模式
    model.eval()
    with torch.no_grad():
        start = time.time()
        img = img.cuda()
        # 模型预测
        pred = model(img)
        pred = pred.data.cpu().numpy()[0]
        score = softmax(pred)
        pred_id = np.argmax(score)  # 返回score数组中最大值的索引
        stop = time.time()
        print('cost time', stop - start)
        print("预测结果：{:s}".format(labels[pred_id][0]))
    return labels[pred_id][0]



if __name__ == "__main__":
    test_list = 'test.txt'
    # 预处理测试集（214个）
    test_data = Garbage_Loader(test_list, train_flag=False)
    test_loader = DataLoader(dataset=test_data, num_workers=2, pin_memory=True, batch_size=1)
    # 定义网络
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(fc_inputs, 214)
    )
    model = model.cuda()
    # 加载训练好的最佳模型
    checkpoint = torch.load('model_best_checkpoint_resnet50.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    # ------------------------------------ step 1/2 : 计算测试集的准确率 ------------------------------------
    computeTestSetAccuracy(test_loader, model)
    # ------------------------------------ step 2/2 : 预测单个图像的结果 ------------------------------------
    # predict(model, "垃圾图片库/有害垃圾_电池/img_电池_343.jpeg")

    # 打印模型的 state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())





'''
原始标签：有害垃圾_电池，模型预测结果：有害垃圾_电池
原始标签：可回收物_木制玩具，模型预测结果：可回收物_木制玩具
原始标签：可回收物_保温杯，模型预测结果：可回收物_保温杯
原始标签：可回收物_垫子，模型预测结果：其他垃圾_PE塑料袋
以及对应图片
'''
