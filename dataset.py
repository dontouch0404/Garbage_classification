import torch
from PIL import Image
import os
import glob
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms 
from PIL import ImageFile # 用于检查图片的质量
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 对数据集进行预处理
class Garbage_Loader(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path) # 处理数据集标签
        self.train_flag = train_flag

        # 对训练集样本进行数据增强
        self.train_tf = transforms.Compose([
                transforms.Resize(224), # 将图片resize到224 * 224的尺寸
                transforms.RandomHorizontalFlip(), # 以0.5的概率水平翻转给定的PIL图像
                transforms.RandomVerticalFlip(), # 以0.5的概率竖直翻转给定的PIL图像
                transforms.ToTensor(), # 把PIL.Image转换为tensor，并归一化到(0,1)

            ])
        # 对验证集样本进行数据增强
        self.val_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])

    # 处理图片标签
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines() # 作为列表返回文件中的所有行，{list:48045}
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info)) # 将list转换为[['路径', '小标签'],...]
        return imgs_info

    # 将图片等比例填充到尺寸为 224 * 224 的纯黑色图片上
    def padding_black(self, img):

        w, h  = img.size # 返回图片原始尺寸

        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

        size_fg = img_fg.size
        size_bg = 224

        img_bg = Image.new("RGB", (size_bg, size_bg)) # 创建一个新的尺寸为244 * 244的图片

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2, # 复制原来的图片，并将尺寸缩放到244 * 244
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    # 数据获取函数，获取路径下的图片及标签
    def __getitem__(self, index):
        img_path, label = self.imgs_info[index] # 获取图片的 相对路径 及 小标签
        img = Image.open(img_path) # 读取数据集
        img = img.convert('RGB') # 将图片转换为RGB格式
        img = self.padding_black(img) # 等比填充
        if self.train_flag:
            img = self.train_tf(img) # 对训练集图片进行数据增强处理
        else:
            img = self.val_tf(img) # 对验证集图片进行数据增强处理
        label = int(label)

        return img, label

    # 返回数据集的数量
    def __len__(self):
        return len(self.imgs_info)
 
    
if __name__ == "__main__":
    # 根据txt标签对图片进行预处理
    train_dataset = Garbage_Loader("train.txt", True)
    print("数据个数：", len(train_dataset))

    # 对数据进行包装处理，读取预处理好的train_dataset按照batch size封装成Tensor
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, # 从dataset数据库中每次抽出batch_size个数据
                                               shuffle=True) # 将数据集打乱

    # 输出Tensor中数据的size及小标签
    for image, label in train_loader:
        print(image.shape)
        print(label)
