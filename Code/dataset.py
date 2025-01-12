import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

from optical_flow import convert_to_optical_flow_image, calculate_flow_mag_and_or

labels_map = {
    "fire": 0,
    "smoke": 1,
    "normal": 2,
}

class FiresenseDataset(Dataset):
    """Video dataset."""

    def __init__(self, root_dir, spatial_transform_rgb, spatial_transform_flow, 
                 numFrames, stride, image_size, input_type, flow_type, fusion_method):
        self.root_dir = root_dir
        self.spatial_transform_rgb = spatial_transform_rgb
        self.spatial_transform_flow = spatial_transform_flow
        self.numFrames = numFrames
        self.stride = stride
        self.image_size = image_size
        self.fusion_method = fusion_method

        assert input_type in ['flow', 'rgb', 'rgb_flow_combined']
        self.input_type = input_type

        assert flow_type in ['raw_flow', 'flow_mag_or', 'flow_dot_product']
        self.flow_type = flow_type
        file_list = glob.glob(self.root_dir + "\*")
        self.data = []
        for class_path in file_list:
            files = os.listdir(class_path)
            class_name = class_path.split("\\")[-1]
            for file_name in files:
                # print(video_path.split("\\")[-2]+ "/"+ video_path.split("\\")[-1])
                self.data.append([class_path + "/" + file_name, class_name])
        self.class_map = labels_map

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, class_name = self.data[idx]

        jpg_files = glob.glob(os.path.join(video_path, '*.jpg'))
        num_jpg_files = len(jpg_files)

        if self.input_type == 'rgb':
            inpSeq = []
            for i in range(0, num_jpg_files + 1, self.stride):
                if not (i > num_jpg_files):
                    if len(inpSeq) == self.numFrames:
                        break
                    clip_name = video_path.split("/")[-1]
                    fl_name = video_path + '/' + clip_name + '_frame' + str(i) + '.jpg'
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform_rgb(img.convert('RGB')))

        elif self.input_type == 'flow':
            inpSeq = []
            flowSeq = []
            for i in range(0, num_jpg_files + 1, self.stride):
                if not (i > num_jpg_files):
                    if len(inpSeq) == self.numFrames:
                        break
                    clip_name = video_path.split("/")[-1]
                    fl_name = video_path + '/' + clip_name + '_frame' + str(i) + '.jpg'
                    img = Image.open(fl_name).convert('RGB')

                    img = img.resize((self.image_size, self.image_size))
                    inpSeq.append(img)

            for i in range(len(inpSeq) - 1):
                # Create a mask array for storing the optical flow
                mask = np.zeros_like(inpSeq[i])
                if self.flow_type == 'raw_flow':
                    flow_image = convert_to_optical_flow_image(inpSeq[i], inpSeq[i+1], mask)
                    flowSeq.append(self.spatial_transform_flow(flow_image))
                elif self.flow_type == 'flow_mag_or':
                    flow_image, magnitude_scaled, orientation = calculate_flow_mag_and_or(inpSeq[i], inpSeq[i+1], mask)

                    # STACK OR CONCAT
                    combined_flow = np.stack((flow_image, magnitude_scaled, orientation), axis=2)
                    flowSeq.append(self.spatial_transform_flow(combined_flow))
                elif self.flow_type == 'flow_dot_product':
                    flow_image, magnitude_scaled, orientation = calculate_flow_mag_and_or(inpSeq[i], inpSeq[i+1], mask)

                    # DOT PRODUCT OF MAGNITUDE AND ORIENTATION
                    dot_product = np.multiply(magnitude_scaled, orientation)

                    # NaN values occured after first gradient step! and Loss is also NaN 
                    combined_flow = np.stack((dot_product, magnitude_scaled, orientation), axis=2)
                    flowSeq.append(self.spatial_transform_flow(combined_flow))

                # Display the resulting optical flow image DISPLAY
                # cv2.imshow("magnitude_scaled", magnitude_scaled)
                # cv2.imshow("orientation", orientation)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        elif self.input_type == 'rgb_flow_combined':
            inpSeq = []
            for i in range(0, num_jpg_files + 1, self.stride):
                if not (i > num_jpg_files):
                    if len(inpSeq) == self.numFrames:
                        break
                    clip_name = video_path.split("/")[-1]
                    fl_name = video_path + '/' + clip_name + '_frame' + str(i) + '.jpg'
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform_rgb(img.convert('RGB')))

            inpSeqFlow = []
            flowSeq = []
            for i in range(0, num_jpg_files + 1, self.stride):
                if not (i > num_jpg_files):
                    if len(inpSeqFlow) == self.numFrames:
                        break
                    clip_name = video_path.split("/")[-1]
                    fl_name = video_path + '/' + clip_name + '_frame' + str(i) + '.jpg'
                    img = Image.open(fl_name).convert('RGB')

                    img = img.resize((self.image_size, self.image_size))
                    inpSeqFlow.append(img)

            for i in range(len(inpSeqFlow) - 1):
                # Create a mask array for storing the optical flow
                mask = np.zeros_like(inpSeqFlow[i])
                flow_image = convert_to_optical_flow_image(inpSeqFlow[i], inpSeqFlow[i + 1], mask)
                flowSeq.append(self.spatial_transform_flow(flow_image))


        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)

        if self.input_type == 'rgb':
            # stack the tensors along a new first dimension
            stacked_tensor = torch.stack(inpSeq, dim=0)
        elif self.input_type == 'flow':
            stacked_tensor = torch.stack(flowSeq, dim=0)
        else:
            stacked_rgb = torch.stack(inpSeq, dim=0)
            stacked_flow = torch.stack(flowSeq, dim=0)

            if self.fusion_method == 'early':
                stacked_tensor = torch.cat((stacked_rgb, stacked_flow), dim=0)
                return stacked_tensor, 0, class_id
            elif self.fusion_method == 'late':
                return stacked_rgb, stacked_flow, class_id

        return stacked_tensor, 0, class_id