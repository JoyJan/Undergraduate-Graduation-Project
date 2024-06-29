import warnings
warnings.filterwarnings("ignore")
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent
import torch
import torchvision
import torchaudio
import soundfile as sf 
import librosa 
from torchvision import transforms
import os
from skimage.transform import resize
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor, QFont, QImage
from PyQt5.QtWidgets import QGraphicsOpacityEffect
import numpy as np
import time
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
from PyQt5.QtGui import QPainter, QBrush, QPixmap, QIcon
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classes = os.listdir('./data')

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

    # 保存图像
    plt.figure(figsize=(10, 10))  # 设置图像大小
    plt.imshow(transform, cmap='viridis')  # 使用特定的颜色映射显示图像
    plt.colorbar()  # 显示颜色条
    plt.savefig('mel_spectrogram.png', dpi=300)  # 保存图像文件
    plt.close()  # 关闭图像窗口

    return transform



class CustomButton(QPushButton):
    def __init__(self, text, parent=None):
        super(CustomButton, self).__init__(text, parent)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #4CAF50; color: white; border: 2px solid #4CAF50; border-radius: 10px; padding: 10px 20px;")
        self.setFont(QFont("Arial", 12, QFont.Bold))

class AudioPlayer(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the UI
        self.init_ui()
        self.net = torch.load('./logs/compressed50.ckpt')
        self.net.eval()
        self.name = 'None'

    def init_ui(self):
        layout = QVBoxLayout()

        self.load_button = CustomButton('Load WAV File', self)
        self.load_button.setFont(QFont("Arial", 24, QFont.Bold))  
        self.load_button.clicked.connect(self.load_wav_file)
        layout.addWidget(self.load_button)

        self.wav_label = QLabel('Loaded WAV File: None', self)
        self.wav_label.setGraphicsEffect(self.createOpacityEffect(0.6))
        self.wav_label.setFont(QFont("Arial", 18, QFont.Bold))  
        layout.addWidget(self.wav_label)

        self.play_button = CustomButton('Play', self)
        self.play_button.setFont(QFont("Arial", 24, QFont.Bold))  
        self.play_button.clicked.connect(self.play_wav_file)
        layout.addWidget(self.play_button)

        self.start_button = CustomButton('Recognition', self)
        self.start_button.setFont(QFont("Arial", 24, QFont.Bold))  
        self.start_button.clicked.connect(self.get_wav_length)
        layout.addWidget(self.start_button)

        self.length_label = QLabel('Result: None', self)
        self.length_label.setStyleSheet("font-size: 28pt; font-weight: bold; border: 4px solid #4CAF50; padding: 10px;")  
        layout.addWidget(self.length_label)

        # grad-cam机制
        self.grad_cam_label = QLabel(self)
        self.grad_cam_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.grad_cam_label)

        self.setLayout(layout)

        self.media_player = QMediaPlayer()
        self.file_path = ''

        self.setStyleSheet("background-color: #E0F2F1;") 

        self.load_button.setStyleSheet("background-color: #2196F3; color: white; border: 2px solid #1976D2; border-radius: 10px; padding: 10px 20px;")
        self.play_button.setStyleSheet("background-color: #64B5F6; color: white; border: 2px solid #1976D2; border-radius: 10px; padding: 10px 20px;")
        self.start_button.setStyleSheet("background-color: #1976D2; color: white; border: 2px solid #1976D2; border-radius: 10px; padding: 10px 20px;")


        self.setWindowTitle('WAV File Player')
        self.setGeometry(100, 100, 800, 400)  
        self.show()

    def createOpacityEffect(self, opacity):
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(opacity)
        return effect

    def load_wav_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter('WAV Files (*.wav)')
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setOptions(options)

        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            self.wav_label.setText(f'Loaded WAV File: {self.file_path}')

    def play_wav_file(self):
        if self.file_path:
            media_content = QMediaContent(QUrl.fromLocalFile(self.file_path))
            self.media_player.setMedia(media_content)
            self.media_player.play()

    def get_wav_length(self):
        if self.file_path:
            start_time = time.time() # 推理开始前的时间戳

            media_content = QMediaContent(QUrl.fromLocalFile(self.file_path))
            input_data = resize_audio_mel(self.file_path)
            input_data = input_data.reshape(1,1,128,128)
            input_data = torch.from_numpy(input_data)
            input_data = input_data.to(torch.float32).to(device)

            with torch.no_grad():
                output = self.net(input_data)
                predicted_class = torch.argmax(output, 1).item()
                self.name = classes[predicted_class]

            duration = self.media_player.duration()
            if self.name != 'None':
                self.length_label.setText(f'Result: {self.name}')
            else:
                self.length_label.setText('Result: None')

            end_time = time.time()  # 推理结束后的时间戳
            inference_time = end_time - start_time  # 推理所需的时间
            print(f"Inference time: {inference_time:.4f} seconds")

def pyqt():
    app = QApplication(sys.argv)
    player = AudioPlayer()
    sys.exit(app.exec_())



