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

conf_matrix = torch.zeros(2, 2) # 种类数


def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(10000)

root = './logs/'

batch_size = 10
num_epochs = 4
classes = os.listdir('./data')
print('classes:',classes)

def draw_test_process1(title, iters, label_cost):
    plt.figure()
    plt.title(title, fontsize=24)
    plt.plot(iters, label_cost, '-', label=label_cost)
    plt.ylabel(str(title))
    plt.ylim([0, 1.05])
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig('./result/'+title+'.jpg')



def evaluate(model, data_loader, device, epoch):

    precison1 = []
    recall1 = []   
    f1score = []   


    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  
    accu_loss = torch.zeros(1).to(device) 

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        precison1.append(precision_score(y_true=labels.tolist(), y_pred=pred_classes.tolist()))
        recall1.append(recall_score(y_true=labels.tolist(), y_pred=pred_classes.tolist()))
        f1score.append(f1_score(y_true=labels.tolist(), y_pred=pred_classes.tolist()))

    print('\n')
    w = 0
    for j in range(len(precison1)):
        w += precison1[j]
    print("precision:",w/len(precison1))
    w = 0
    for j in range(len(recall1)):
        w += recall1[j]
    print("recall:",w/len(recall1))
    w = 0
    for j in range(len(f1score)):
        w += f1score[j]
    print("f1score:",w/len(f1score))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def resize_audio_mel(path):
    wav,sr = sf.read(path)
    
    arr_len = wav.shape[0]

    if arr_len<16000:
        
        wav = np.pad(wav,(16000-arr_len)//2,constant_values=0)
    else:
        wav = wav[:16000]

    transform = librosa.feature.melspectrogram(y = wav,sr=sr,n_mels=128,hop_length = 512,fmax=16000)
    transform = np.squeeze(transform)
    if transform.shape[0] !=128:
        transform =  transform.T
    transform = resize(transform, (128, 128))
    return transform



class MyDataset(torch.utils.data.Dataset): 
    def __init__(self, datatxt): 
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  
        imgs = [] 
        for line in fh:  
            line = line.rstrip() 
            words = line.split()
            imgs.append((line[:-2], int(words[-1])))  
            
        self.imgs = imgs

    def __getitem__(self, index): 
        fn, label = self.imgs[index]  
        audio_mel = resize_audio_mel(fn)
        return audio_mel, label  

    def __len__(self):  
        return len(self.imgs)

train_data = MyDataset(datatxt=root + 'train.txt')
test_data = MyDataset(datatxt=root + 'val.txt')

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True,)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle=False,)

print("PyTorch Version: ",torch.__version__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# vgg16 = models.vgg16(pretrained=False)
# resnet101 = models.resnet101(pretrained=True)
resnet50 = models.resnet34(pretrained=True)
# target_layer = [resnet50.layer4[-1]]
# cam = GradCAM(model=resnet50, target_layers=target_layer, use_cuda=True)

# for params in vgg16.parameters():
# for params in resnet101.parameters():
for params in resnet50.parameters():
    params.requires_grad = True

# # 修改VGG19的conv1层以接受1个通道的输入（音频特征图通常是单通道的）
# vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# resnet101.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# # 修改VGG19的全连接层以输出正确的类别数
# vgg16.classifier[6] = nn.Linear(4096, len(classes))

# vgg19.fc.out_features = len(classes)
# resnet101.fc.out_features = len(classes)
resnet50.fc.out_features = len(classes)

# net = vgg16
# net = resnet101
net = resnet50

net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00003, momentum=0.9)


loss22 = []
acc22 = []
precison1 = []  
recall1 = []    
f1score = []   

total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = tqdm(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(batch_size,1,128,128)
        images = images.to(torch.float32).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    net.eval()
    test_loss = 0.
    test_acc = 0.
    y_true = []
    y_pred = []
    with torch.no_grad(): 
        # test
        total_correct = 0
        total_num = 0
        for x, label in test_loader:   
            # [b, 3, 32, 32]
            # [b]
            y_true += label
            x = x.reshape(batch_size,1,128,128)
            x = x.to(torch.float32)
            x, label = x.to(device), label.to(device)  
            # [b, 10]
            logits = net(x)
            # [b]
            pred = logits.argmax(dim=1) 
            y_pred += pred
            # [b] vs [b] => scalar tensor 
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            # # Forward pass
            # logits = net(x)
            # # Backward pass
            # net.zero_grad()
            # target_layer = net.features[-1]  # 选择 VGG16 的最后一个卷积层
            # cam = GradCAM(model=net, target_layers=target_layer)
            # grayscale_cam = cam(input_tensor=x)
            # grayscale_cam = grayscale_cam[0, :]
            # visualization = show_cam_on_image(x.squeeze().numpy(), grayscale_cam)
            # # 保存 Grad-CAM 图像
            # plt.imshow(visualization)
            # plt.savefig('gradcam_sample.png')
            # print(correct)
    # acc = total_correct / total_num  
    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    import os
    list_names = os.listdir('./data')
    print(classification_report(y_true.cpu().numpy(),y_pred.cpu().numpy(),digits=5,labels=range(len(list_names)),target_names=list_names))

    precison1.append(precision_score(y_true.cpu().numpy(),y_pred.cpu().numpy(),pos_label='positive',average='weighted'))
    recall1.append(recall_score(y_true.cpu().numpy(),y_pred.cpu().numpy(),pos_label='positive',average='weighted'))
    f1score.append(f1_score(y_true.cpu().numpy(),y_pred.cpu().numpy(),pos_label='positive',average='weighted'))
    print(y_true.cpu().numpy(), y_pred.cpu().numpy(), list_names)


    torch.save(net, './logs/model_34.ckpt')



iters = np.arange(1,num_epochs+1,1)
draw_test_process1('precison', iters, precison1)
draw_test_process1('recall', iters, recall1)
draw_test_process1('f1score', iters, f1score)
print(precison1)