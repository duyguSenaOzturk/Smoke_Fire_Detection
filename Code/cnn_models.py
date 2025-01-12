import glob

import numpy as np
from skimage.transform import resize
import cv2
import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torch.optim import Adam
from tqdm import tqdm
from spatial_transforms import (Compose, ToTensor, FiveCrops, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, TenCrops, FlippedImagesTest, CenterCrop)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeWarp(nn.Module):
    def __init__(self, base_model, method='squeeze'):
        super(TimeWarp, self).__init__()
        self.baseModel = base_model
        self.method = method

    def forward(self, x):
        x = x.to(device=device, dtype=torch.float)
        batch_size, time_steps, C, H, W = x.size()
        if self.method == 'loop':
            output = []
            for i in range(time_steps):
                # input one frame at a time into the basemodel
                x_t = self.baseModel(x[:, i, :, :, :])
                # Flatten the output
                x_t = x_t.view(x_t.size(0), -1)
                output.append(x_t)
            # end loop
            # make output as  (samples, timesteps, output_size)
            x = torch.stack(output, dim=0).transpose_(0, 1)
            output = None # clear var to reduce data  in memory
            x_t = None  # clear var to reduce data  in memory
        else:
            # reshape input  to be (batch_size * timesteps, input_size)
            x = x.contiguous().view(batch_size * time_steps, C, H, W)
            x = self.baseModel(x)
            x = x.view(x.size(0), -1)
            # make output as  (samples, timesteps, output_size)
            x = x.contiguous().view(batch_size, time_steps, x.size(-1))
        return x


class ExtractLastCell(nn.Module):
    def forward(self,x):
        out, _ = x
        return out[:, -1, :]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x # do nothing



class CNN_LSTM_RESNET50(nn.Module):
    def __init__(self, image_size=100, method='squeeze', num_classes=3):
        super(CNN_LSTM_RESNET50, self).__init__()

        # Create model
        dr_rate = 0.2
        pretrained = True
        rnn_hidden_size = 80
        rnn_num_layers = 1
        base_model = models.resnet50(pretrained=pretrained)
        base_model = nn.Sequential(*list(base_model.children())[:-1]) # Remove the last one layer of the base model
        i = 0
        for child in base_model.children():
            if i < 5: # This number could vary depending on how many layers you want to freeze
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            i += 1

        num_features = 2048 # Number of output filters in the last convolution layer in ResNet50
        if image_size == 100:
            num_features = 2048
        elif image_size == 160:
            num_features = 2048
        elif image_size == 224:
            num_features = 2048

        self.model = nn.Sequential(TimeWarp(base_model, method),
                                   nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, bidirectional=True),
                                   ExtractLastCell(),
                                   nn.Linear(160, 256), # You might need to adjust this number based on your new backbone
                                   nn.ReLU(),
                                   nn.Dropout(dr_rate),
                                   nn.Linear(256, num_classes)
                                   )

    # Feed forward function
    def forward(self, input):
        output = self.model(input)
        return output


class CNN_LSTM_VGG19(nn.Module):
    def __init__(self, image_size=100, method='squeeze', num_classes=3):
        super(CNN_LSTM_VGG19, self).__init__()

        # Create model
        dr_rate = 0.2
        pretrained = True
        rnn_hidden_size = 80
        rnn_num_layers = 1
        base_model = models.vgg19(pretrained=pretrained).features
        i = 0
        for child in base_model.children():
            if i < 28:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            i += 1

        num_features = 4608
        if image_size == 100:
            num_features = 4608
        elif image_size == 160:
            num_features = 12800
        elif image_size == 224:
            num_features = 25088
        elif image_size == 50:
            num_features = 512
        # Example of using Sequential
        self.model = nn.Sequential(TimeWarp(base_model, method=method),
                                   nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, batch_first=True, bidirectional=True),
                                   ExtractLastCell(),
                                   nn.Linear(160, 256),
                                   nn.ReLU(),
                                   nn.Dropout(dr_rate),
                                   nn.Linear(256, num_classes)
                                   )

    # Feed forward function
    def forward(self, input):
        output = self.model(input)
        return output