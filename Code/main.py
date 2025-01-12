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
from dataset import FiresenseDataset
from cnn_models import *


# Hyperparameters
batch_size = 32
learning_rate = 0.00005
weight_decay = 0.0001
num_epochs = 10
num_classes = 3

image_size = 100  # 100, 160, 224
numFrames = 10
stride = 2
input_type = 'rgb'  # flow, rgb, rgb_flow_combined (if rgb_flow_combined => flow_type=raw_flow)
flow_type = 'raw_flow'  # raw_flow, flow_mag_or, flow_dot_product
fusion_method = 'early' # early, late
method = 'squeeze'  # loop, squeeze
cnn_lstm_model = 'resnet50' # vgg19, resnet50
train_path = './dataset/train'
test_path = './dataset/test'
log_dir = 'logs/logs9/numFrames=10, stride=2, backbone=resnet50, lr=0.00005_2'


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = Normalize(mean=mean, std=std)

spatial_transform_rgb = Compose([MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], image_size),
                                 ToTensor(), normalize])
spatial_transform_flow = Compose([ToTensor(), normalize])


train_dataset = FiresenseDataset(train_path, spatial_transform_rgb=spatial_transform_rgb, spatial_transform_flow=spatial_transform_flow, numFrames=numFrames, stride=stride, image_size=image_size,
                                   input_type=input_type, flow_type=flow_type, fusion_method=fusion_method)
val_dataset = FiresenseDataset(test_path, spatial_transform_rgb=spatial_transform_rgb, spatial_transform_flow=spatial_transform_flow, numFrames=numFrames, stride=stride, image_size=image_size,
                                   input_type=input_type, flow_type=flow_type, fusion_method=fusion_method)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    val_dataset,
    batch_size=batch_size, shuffle=True
)
print(len(train_loader))
print(len(test_loader))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def class_score_fusion(outputs1, outputs2):
    # Perform fusion by averaging the scores
    fused_outputs = (outputs1 + outputs2) / 2

    return fused_outputs


# # define the log file path
# log_dir = './logs'

# tensorboard --logdir=runs
# Create a SummaryWriter instance to write the TensorBoard logs
writer = SummaryWriter(log_dir=log_dir)


if cnn_lstm_model == 'vgg19':
    model = CNN_LSTM_VGG19(image_size=image_size, method=method, num_classes=num_classes)
elif cnn_lstm_model == 'resnet50':
    model = CNN_LSTM_RESNET50(image_size=image_size, method=method, num_classes=num_classes)


if input_type == 'rgb_flow_combined' and fusion_method == 'late':
    if cnn_lstm_model == 'vgg19':
        model2 = CNN_LSTM_VGG19(image_size=image_size, method=method, num_classes=num_classes)
    elif cnn_lstm_model == 'resnet50':
        model2 = CNN_LSTM_RESNET50(image_size=image_size, method=method, num_classes=num_classes)
    model2.to(device)
    # Optimizer and loss function
    optimizer2 = Adam(model2.parameters(), lr=learning_rate, weight_decay=weight_decay)

model.to(device)
# Optimizer and loss function
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = nn.CrossEntropyLoss()

# calculating the size of training and test dataset
train_count = len(glob.glob(train_path+'\**\*'))
test_count = len(glob.glob(test_path+'\**\*'))

best_accuracy = 0.0
# store loss and accuracies on epochs to save later and access them
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# def check_nan_weights(model):
#     for param in model.parameters():
#         if torch.isnan(param).any():
#             return True
#     return False

def check_nan_weights(model):
    nan_exists = False
    nan_counts = []

    for name, param in model.named_parameters():
        nan_count = torch.isnan(param).sum().item()
        if nan_count > 0:
            nan_exists = True
            nan_counts.append((name, nan_count))
    
    return nan_exists, nan_counts

# Model training and saving the best model
for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    num_of_true_predictions = 0.0
    train_loss_sum = 0

    total_train_labels = []
    total_train_predicts = []

    for i, (images, flow_images, labels) in tqdm(enumerate(train_loader)):
        has_nan, nan_counts = check_nan_weights(model)
        if torch.cuda.is_available():
            # USE GPU
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()  # reset after starting of a new batch
        train_outputs = model(images)  # gives us predictions

        if input_type == 'rgb_flow_combined' and fusion_method == 'late':
            optimizer2.zero_grad()  # reset after starting of a new batch
            flow_images = Variable(flow_images.cuda())
            train_outputs2 = model2(flow_images)  # gives us predictions
            train_outputs = class_score_fusion(train_outputs, train_outputs2)

        train_loss = loss_function(train_outputs, labels)
        train_loss.backward()  # computes all grads (backprop)
        optimizer.step()  # update weights and biases

        if input_type == 'rgb_flow_combined' and fusion_method == 'late':
             optimizer2.step()  # update weights and biases

        # extend prediction list by the values on this epoch to use at the end of the epoch
        y_pred = train_outputs.cpu().data.numpy()
        total_train_labels.extend(labels.cpu())
        total_train_predicts.extend(list(np.argmax(y_pred, axis=1)))

        # Returns the class value of the maximum value of the predictions
        _, prediction = torch.max(train_outputs.data, 1)  # calculate prediction
        num_of_true_predictions += int(torch.sum(prediction == labels.data))

        # add losses
        train_loss_sum += train_loss.item() * images.size(0)

    train_accuracy = num_of_true_predictions / train_count
    # Add scalars to TensorBoard
    writer.add_scalar('Loss/Train', train_loss_sum, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)


# Evaluation on "test" dataset
    model.eval()

    test_accuracy = 0.0
    num_of_true_predictions = 0.0
    test_loss_sum = 0
    total_test_labels = []
    total_pred_labels = []
    for i, (images, flow_images, labels)  in enumerate(test_loader):
        if torch.cuda.is_available():
            # USE GPU
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        test_outputs = model(images)
        if input_type == 'rgb_flow_combined' and fusion_method == 'late':
            flow_images = Variable(flow_images.cuda())
            test_outputs2 = model2(flow_images)
            test_outputs = class_score_fusion(test_outputs, test_outputs2)

        test_loss = loss_function(test_outputs, labels)
        _, prediction = torch.max(test_outputs.data, 1)
        num_of_true_predictions += int(torch.sum(prediction == labels.data))

        test_outputs = test_outputs.cpu().data.numpy()
        total_pred_labels.extend(list(np.argmax(test_outputs, axis=1)))
        total_test_labels.extend(labels.cpu())

    # add losses
    test_loss_sum += test_loss.item() * images.size(0)

    test_accuracy = num_of_true_predictions / test_count
    # Add scalars to TensorBoard
    writer.add_scalar('Loss/Test', test_loss_sum, epoch)
    writer.add_scalar('Accuracy/Test', test_accuracy, epoch)

    # Calculating accuracies and metrics for both train and validation on all dataset results
    report = classification_report(total_test_labels, total_pred_labels)
    train_accuracy = accuracy_score(total_train_labels, total_train_predicts)
    train_accuracies.append(train_accuracy)
    valid_accuracy = accuracy_score(total_test_labels, total_pred_labels)
    test_accuracies.append(valid_accuracy)
    epoch_train_loss = train_loss_sum / train_count
    train_losses.append(epoch_train_loss)
    print('Epoch %d Train Loss: %.3f, Train Accuracy: %.3f\nClassification Report on Test:\n%s' %
          (epoch + 1, epoch_train_loss, train_accuracy, report))

    print("Confusion Matrix on Test:\n%s\n\n" % confusion_matrix(total_test_labels, total_pred_labels))

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy

# Close the SummaryWriter instance after finishing the training
writer.close()

