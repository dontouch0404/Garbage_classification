from dataset import Garbage_Loader
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import time
import os
import shutil
from tensorboardX import SummaryWriter
from torchsummary import summary # 用于输出网络模型
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(output, target, topk=(1,)):
    """
        计算topk的准确率
    """
    with torch.no_grad(): # 避免计算梯度
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #修改了target.view()为target.reshape()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        class_to = pred[0].cpu().numpy()

        res = []
        for k in topk:
            #view改为了reshape
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, class_to

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
        根据 is_best 存模型，一般保存 valid acc 最好的模型
    """
    torch.save(state, filename) # 保存当前模型的整个网络，包括网络的整个结构和参数，filename为路径
    # 如果此模型为最佳模型，则上一步保存的模型覆盖到最佳模型中
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename) # 将名为filename的文件的内容（无元数据）复制到名为'model_best_' + filename的文件中



def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
        训练代码
        参数：
            train_loader - 训练集的 DataLoader
            model - 模型
            criterion - 损失函数
            optimizer - 优化器
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX 
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Set model to training mode，启用 BatchNormalization 和 Dropout
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # 计算数据加载时间
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        # input = input.to(device)
        # target = target.to(device)

        # 前向传播，使用模型计算输入数据的输出
        output = model(input)
        loss = criterion(output, target)

        # 测量准确率（包括当前轮次准确率与目前平均准确率），同时记录损失值（包括当前轮次损失率与目前平均损失率）
        [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5)) # 计算topk的准确率
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # 在向后传递中计算梯度
        optimizer.zero_grad() # 清理现有的梯度
        loss.backward() # 后向传播
        optimizer.step()  # 使用优化器的step函数更新参数

        # 计算经过的时间
        batch_time.update(time.time() - end)
        end = time.time()
        '''                     训练经过的时间（均值） 数据加载时间（均值） 当前训练集损失值（均值）   top1准确率（均值）       top5准确率（均值）
        Epoch: [0][2670/3003]	Time 0.376 (0.383)	Data 0.000 (0.024)	Loss 1.5936 (2.1519)	Prec@1 56.250 (49.534)	Prec@5 75.000 (76.261)
        Epoch: [0][2680/3003]	Time 0.374 (0.383)	Data 0.000 (0.024)	Loss 1.4319 (2.1501)	Prec@1 62.500 (49.580)	Prec@5 81.250 (76.285)
        top1：预测的label取最后概率向量里面最大的那一个作为预测结果，如过你的预测结果中概率最大的那个分类正确，则预测正确。否则预测错误
        top5：就是最后概率向量最大的前五名中，只要出现了正确概率即为预测正确。否则预测错误
        '''

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    # 每个epoch写入训练集的损失值变化和准确率变化
    # writer.add_scalar('loss/train_loss', losses.val, global_step=epoch)
    # writer.add_scalar('acc/train_acc', top1.val, global_step=epoch)

    writer.add_scalars('Train_val_loss', {'train_loss': losses.val}, epoch)
    writer.add_scalars('Train_val_acc', {'train_acc': top1.val}, epoch)


def validate(val_loader, model, criterion, epoch, writer, phase="VAL"):
    """
        验证代码
        参数：
            val_loader - 验证集的 DataLoader
            model - 模型
            criterion - 损失函数
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX 
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 转换模型到评估模式
    model.eval()

    # 验证集，无需自动梯度跟踪
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # 前向传播，使用模型计算输入数据的输出
            output = model(input)
            loss = criterion(output, target)

            # 测量准确率，并记录损失
            [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # 测量经过时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 每10组输出一次
            if i % 10 == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              phase, i, len(val_loader),
                              batch_time=batch_time,
                              loss=losses,
                              top1=top1, top5=top5))

        print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(phase, top1=top1, top5=top5))
    # 每个epoch写入验证集的损失值变化和准确率变化
    # writer.add_scalar('loss/valid_loss', losses.val, global_step=epoch)
    # writer.add_scalar('acc/valid_acc', top1.val, global_step=epoch)

    writer.add_scalars('Train_val_loss', {'val_loss': losses.val}, epoch)
    writer.add_scalars('Train_val_acc', {'val_acc': top1.val}, epoch)

    return top1.avg, top5.avg

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # -------------------------------------------- step 1/4 : 加载数据（每次注意修改batch_size和num_workers以及epochs） ---------------------------
    train_dir_list = 'train.txt'
    valid_dir_list = 'val.txt'
    # bitch_size从64改到16，bitch_size一般为2^n，常用的包括64, 128, 256
    batch_size = 64
    epochs = 80
    num_classes = 214 # 分类个数
    # 预处理训练集和验证集
    train_data = Garbage_Loader(train_dir_list, train_flag=True)
    valid_data = Garbage_Loader(valid_dir_list, train_flag=False)
    # 加载训练集和验证集（在这里修改num_workers从8到3，num_worker一般设置为3~5）
    train_loader = DataLoader(dataset=train_data, num_workers=8, pin_memory=True, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, num_workers=8, pin_memory=True, batch_size=batch_size)
    # 训练集数量：48045, 验证集数量：5652
    train_data_size = len(train_data)
    print('训练集数量：%d' % train_data_size)
    valid_data_size = len(valid_data)
    print('验证集数量：%d' % valid_data_size)
    # ------------------------------------ step 2/4 : 定义网络 ------------------------------------
    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    # !冻结除最终全连接层之外的所有网络的权重，以便不在backward()中计算梯度（做迁移学习的对比实验用）
    # for param in model.parameters():
    #     param.requires_grad = False
    # 提取fc层中固定的参数
    fc_inputs = model.fc.in_features # 2048
    # # 重置最终的全连接层，修改类别为num_classes = 214
    # model.fc = nn.Linear(fc_inputs, num_classes)
    # 迁移学习部分代码
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(fc_inputs, num_classes)
    )
    model = model.cuda() # 将模型加载到GPU上去
    # summary(model, (3, 224, 224))  # 输出网络的模型参数
    '''
    冻结全连接层之前的网络：
    Total params: 23,946,518
    Trainable params: 438,486
    Non-trainable params: 23,508,032
    对整个网络结构进行微调：
    Total params: 23,946,518
    Trainable params: 23,946,518
    Non-trainable params: 0
    '''
    #
    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    lr_init = 0.0001 # 学习率
    lr_stepsize = 20 # 学习率下降间隔数，若为20，则会在20、40、60、80…个step时，将学习率调整为lr*gamma
    weight_decay = 0.001 # 权重衰减
    # 定义损失函数，多分类采用 categorical_crossentropy
    criterion = nn.CrossEntropyLoss().cuda()
    # 定义adam优化算法（若冻结FC层之外层权重，则仅仅对FC进行参数更新）
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay) # 优化器
    # 调整学习率，其中gamma为学习率调整倍数。每20个epoch将学习率下降0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
    # 用于写tensorboardX，路径下存放画图用的文件
    writer = SummaryWriter('runs/resnet50')
    '''
    会发现刚刚的runs/resnet50文件夹里面有文件了。在命令行输入如下，载入刚刚做图的文件（那个./log要写完整的路径）
    tensorboard --logdir ./log     tensorboard --logdir runs/resnet50
    在浏览器输入：http://0.0.0.0:6006/，或者localhost:6006 就可以看到我们做的两个图了
    '''
    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    best_prec1 = 0 # 最佳准确率统计，初始化为0
    # 训练epochs次
    for epoch in range(epochs):
        # scheduler.step()
        # 对网络训练
        train(train_loader, model, criterion, optimizer, epoch, writer)
        # 在验证集上测试效果，返回验证集上的top1和top5
        valid_prec1, valid_prec5 = validate(valid_loader, model, criterion, epoch, writer, phase="VAL")
        is_best = valid_prec1 > best_prec1 # 如果验证集准确率大于当前最佳准确率，则is_best为1，否则为0
        best_prec1 = max(valid_prec1, best_prec1) # 更新当前最佳准确率
        # 模型保存方法（1）保存用于继续训练的checkpoint或者多个模型
        # 调用自定义方法，保存验证集上准确率最高的模型（根据 is_best 存模型，一般保存 valid acc 最好的模型）
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(), # 网络参数
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, is_best,
            filename='checkpoint_resnet50.pth.tar')
        # （2）仅保存加载整个state_dict
        if is_best: # 仅当模型为最优时，保存一个最佳模型
            torch.save(model.state_dict(), 'model_best_stateDict_checkpoint.pth.tar')

        # 避免pytorch跳过第一个学习率的调整
        scheduler.step()  # 对学习率进行调整
    writer.close()


'''大概17min跑完一个epoch
Epoch: [0][2670/3003]	Time 0.376 (0.383)	Data 0.000 (0.024)	Loss 1.5936 (2.1519)	Prec@1 56.250 (49.534)	Prec@5 75.000 (76.261)
Epoch: [0][2680/3003]	Time 0.374 (0.383)	Data 0.000 (0.024)	Loss 1.4319 (2.1501)	Prec@1 62.500 (49.580)	Prec@5 81.250 (76.285)
Epoch: [0][2690/3003]	Time 0.374 (0.383)	Data 0.000 (0.024)	Loss 1.3689 (2.1475)	Prec@1 75.000 (49.631)	Prec@5 87.500 (76.322)
Epoch: [0][2700/3003]	Time 0.376 (0.382)	Data 0.000 (0.024)	Loss 1.3616 (2.1452)	Prec@1 56.250 (49.671)	Prec@5 100.000 (76.361)
Epoch: [0][2710/3003]	Time 0.375 (0.382)	Data 0.000 (0.024)	Loss 0.9235 (2.1437)	Prec@1 75.000 (49.693)	Prec@5 93.750 (76.374)
大概2min跑完一个test-VAL
Test-VAL: [310/354]	Time 0.111 (0.355)	Loss 1.6978 (1.4383)	Prec@1 62.500 (63.666)	Prec@5 87.500 (87.078)
Test-VAL: [320/354]	Time 0.738 (0.355)	Loss 0.8327 (1.4349)	Prec@1 81.250 (63.785)	Prec@5 93.750 (87.150)
Test-VAL: [330/354]	Time 0.109 (0.358)	Loss 0.5828 (1.4340)	Prec@1 81.250 (63.822)	Prec@5 93.750 (87.198)
Test-VAL: [340/354]	Time 0.110 (0.359)	Loss 1.1016 (1.4373)	Prec@1 56.250 (63.728)	Prec@5 100.000 (87.243)
Test-VAL: [350/354]	Time 0.257 (0.357)	Loss 0.9977 (1.4363)	Prec@1 75.000 (63.746)	Prec@5 87.500 (87.197)

 * VAL Prec@1 63.747 Prec@5 87.208
'''