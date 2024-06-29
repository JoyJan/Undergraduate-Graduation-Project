import warnings

import librosa

warnings.filterwarnings("ignore")
import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchaudio
import sys
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm
import random
import os
import soundfile as sf
from IPython.display import Audio
import librosa
from skimage.transform import resize
from sklearn.metrics import classification_report,precision_score, recall_score,f1_score
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns  # 用于更美观的热图

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(10000)

# 定义目录和数据集参数
root = './logs/'
batch_size = 10
num_epochs = 4
classes = os.listdir('./data')
print('classes:', classes)

# 绘制训练过程函数
def draw_test_process1(title, iters, label_cost):
    plt.figure()
    plt.title(title, fontsize=24)
    plt.plot(iters, label_cost, '-', label=label_cost)
    plt.ylabel(str(title))
    plt.ylim([0, 1.05])
    plt.xlabel('epoch')
    plt.savefig('./result/' + title + '.jpg')

# 使用 Grad-CAM 的评估函数
def evaluate_with_gradcam(model, data_loader, device, epoch, target_layer):
    # precision1, recall1, f1score = [], [], []
    #
    # model.eval()
    # accu_num = torch.zeros(1).to(device)
    # accu_loss = torch.zeros(1).to(device)
    #
    # sample_num = 0
    # data_loader = tqdm(data_loader, file=sys.stdout)
    #
    # cam = GradCAM(model=model, target_layers=[target_layer])
    #
    # for step, data in enumerate(data_loader):
    #     images, labels = data
    #     sample_num += images.shape[0]
    #
    #     images = images.reshape(batch_size, 1, 128, 128).to(device, dtype=torch.float32)
    #     pred = model(images)
    #     pred_classes = torch.max(pred, dim=1)[1]
    #     accu_num += torch.eq(pred_classes, labels.to(device)).sum()
    #
    #     loss = nn.CrossEntropyLoss()(pred, labels.to(device))
    #     accu_loss += loss
    #
    #     # 为批次中的第一张图像生成 Grad-CAM 可视化
    #     grayscale_cam = cam(input_tensor=images[0:1])[0]
    #
    #
    #     # 将灰度图像转换为可叠加的 RGB
    #     original_image = images[0].cpu().numpy().squeeze()
    #     original_image_normalized = (original_image - original_image.min()) / (
    #                 original_image.max() - original_image.min())
    #     original_image_rgb = np.stack([original_image_normalized] * 3, axis=-1)
    #
    #     # 创建可视化结果，将类激活图叠加到原始灰度图上
    #     visualization = show_cam_on_image(original_image_rgb, grayscale_cam, use_rgb=True)
    #
    #     # 绘制并保存可视化结果
    #     # cmap='gray'
    #     plt.imshow(visualization)
    #     plt.title(f'Grad-CAM Visualization - Epoch {epoch}, Step {step}')
    #     plt.axis('off')
    #     plt.savefig(f'./result/gradcam_epoch{epoch}_step{step}.jpg')
    #     plt.close()
    #
    #     data_loader.desc = f"[valid epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}"
    #
    #     precision1.append(precision_score(labels.cpu(), pred_classes.cpu(), average='weighted'))
    #     recall1.append(recall_score(labels.cpu(), pred_classes.cpu(), average='weighted'))
    #     f1score.append(f1_score(labels.cpu(), pred_classes.cpu(), average='weighted'))
    precision1, recall1, f1score = [], [], []

    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    cam = GradCAM(model=model, target_layers=[target_layer])

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        images = images.reshape(batch_size, 1, 128, 128).to(device, dtype=torch.float32)
        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = nn.CrossEntropyLoss()(pred, labels.to(device))
        accu_loss += loss

        # 为批次中的第一张图像生成 Grad-CAM 可视化
        grayscale_cam = cam(input_tensor=images[0:1])[0]

        # 获取原始灰度图像并转换为 RGB 格式，以便生成可视化结果
        original_image = images[0].cpu().numpy().squeeze()
        original_image_normalized = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        original_image_rgb = np.stack([original_image_normalized] * 3, axis=-1)

        # 创建可视化结果，将类激活图叠加到原始灰度图上
        visualization = show_cam_on_image(original_image_rgb, grayscale_cam)

        # 绘制并保存可视化结果
        plt.imshow(visualization)
        plt.title(f'Grad-CAM Visualization - Epoch {epoch}, Step {step}')
        plt.axis('off')
        plt.savefig(f'./result/gradcam_epoch{epoch}_step{step}.jpg')
        plt.close()

        # 保存原始灰度图像
        plt.imshow(original_image_normalized, cmap='gray')
        plt.title(f'Original Image - Epoch {epoch}, Step {step}')
        plt.axis('off')
        plt.savefig(f'./result/original_epoch{epoch}_step{step}.jpg')
        plt.close()

        data_loader.desc = f"[valid epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}"
        precision1.append(precision_score(labels.cpu(), pred_classes.cpu(), average='weighted'))
        recall1.append(recall_score(labels.cpu(), pred_classes.cpu(), average='weighted'))
        f1score.append(f1_score(labels.cpu(), pred_classes.cpu(), average='weighted'))

    print('\n')
    print("Precision:", sum(precision1) / len(precision1))
    print("Recall:", sum(recall1) / len(recall1))
    print("F1 Score:", sum(f1score) / len(f1score))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# 数据集类和训练设置
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datatxt):
        super(MyDataset, self).__init__()
        with open(datatxt, 'r') as fh:
            self.imgs = [(line[:-2], int(line.split()[-1])) for line in fh]

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        audio_mel = resize_audio_mel(fn)
        return torch.from_numpy(audio_mel).float(), label

    def __len__(self):
        return len(self.imgs)

# 音频预处理函数
def resize_audio_mel(path):
    import soundfile as sf
    from skimage.transform import resize
    wav, sr = sf.read(path)
    if len(wav) < 16000:
        wav = np.pad(wav, (16000 - len(wav)) // 2, constant_values=0)
    else:
        wav = wav[:16000]
    transform = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, hop_length=512, fmax=16000)
    transform = resize(np.squeeze(transform), (128, 128))
    return transform

# 训练和测试数据集
train_data = MyDataset(datatxt=root + 'train.txt')
test_data = MyDataset(datatxt=root + 'val.txt')
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化 VGG16 模型以用于 Grad-CAM 可视化
vgg16 = models.vgg16(pretrained=False)
vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
vgg16.classifier[6] = nn.Linear(4096, len(classes))
target_layer = vgg16.features[-1]
vgg16 = vgg16.to(device)

# 训练设置
optimizer = optim.SGD(vgg16.parameters(), lr=0.00003, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 训练循环
precision1, recall1, f1score = [], [], []
for epoch in range(num_epochs):
    vgg16.train()
    train_loader = tqdm(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(batch_size, 1, 128, 128).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        

    # 使用 Grad-CAM 进行评估和可视化
    test_loss, test_acc = evaluate_with_gradcam(vgg16, test_loader, device, epoch, target_layer)

# 绘制训练进度
iters = np.arange(1, num_epochs + 1, 1)
draw_test_process1('Precision', iters, precision1)
draw_test_process1('Recall', iters, recall1)
draw_test_process1('F1 Score', iters, f1score)
