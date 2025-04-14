import time
import os, sys, h5py  
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
import gc
from torch import Tensor
import math

Sbox = np.array([99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254,
215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164,
114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49,
21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131,
44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32,
252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51,
133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188,
182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126,
61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222,
94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121,
231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186,
120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181,
102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105,
217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230,
66, 104, 65, 153, 45, 15, 176, 84, 187, 22],dtype=np.uint8)

HW = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2,
3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4,
5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2,
3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3,
4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5,
6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3,
4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4,
5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7,
8],dtype=np.uint8)


def linear_matmul(inputs, weight, bias=None):
    ret = torch.bmm(inputs, weight)
    ret += bias
    return ret.transpose(0, 1)

class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty((num_channels, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_channels, out_features), **factory_kwargs).unsqueeze(1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, inputs: Tensor) -> Tensor:
        return linear_matmul(inputs.transpose(0,1), self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def l1_regularization(model, l1_lamda1, l1_lamda2):
    l1_norm = sum(torch.sum(torch.abs(param)) for name, param in model.named_parameters() if 'shared_layer1' in name) * l1_lamda1
    l1_norm += sum(torch.sum(torch.abs(param)) for name, param in model.named_parameters() if 'shared_layer2' in name) * l1_lamda2
    #l1_norm = sum(p.abs().sum() for p in model.parameters() )
    return l1_norm

class convWIN_MCR(nn.Module):
    def __init__(self, kernel, stride, num_features):
        super(convWIN_MCR, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.conv_0 = nn.Sequential(nn.Conv1d(1, kernel, kernel_size=1, stride=1), nn.BatchNorm1d(kernel), nn.ReLU(), nn.AvgPool1d(kernel_size=stride*2, stride=stride))  #nn.AvgPool1d(kernel_size=2, stride=1)
        self.flat = nn.Flatten()

        conv_output_length = num_features
        pooled_output_length = ((conv_output_length - (stride * 2)) // stride) + 1
        self.input_size = kernel * pooled_output_length
        
        # shared layer
        self.shared_layer1 = nn.Linear(self.input_size, shared_layer_size1)
        self.bn1 = nn.BatchNorm1d(shared_layer_size1)
        self.relu1 = nn.ReLU()
        self.shared_layer2 = nn.Linear(shared_layer_size1, shared_layer_size2)
        self.bn2 = nn.BatchNorm1d(shared_layer_size2)
        self.relu2 = nn.ReLU()
        
        # output layers
        self.output_layers = Linear(shared_layer_size2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, input):
        x = input.view(input.shape[0], 1, -1)
        x = self.conv_0(x)
        x = self.flat(x)
        # shared layer
        x = self.shared_layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.shared_layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # non-shared layer and output layer
        outputs = x.unsqueeze(1).repeat(1, 256, 1)
        outputs = self.output_layers(outputs)
        outputs = self.relu(outputs)
            
        return outputs.squeeze(2)


class H5Dataset(Dataset):
    def __init__(self, file_path, start_traces, end_traces):
        self.file = h5py.File(file_path, 'r')
        self.traces_dset = self.file['trace']
        self.targets_dset = HW[self.file['label_parallel']]  #(1w, 256)
        print(self.targets_dset.shape)

        self.start_idx = start_traces
        self.end_idx = end_traces

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        global_idx = self.start_idx + idx

        trace = self.traces_dset[global_idx, :]
        trace = torch.tensor(trace, dtype=torch.float32)
        trace = trace - torch.mean(trace)
        trace = trace / (torch.max(torch.abs(trace)))
        trace = trace.unsqueeze(0) 
        
        targets = torch.tensor(self.targets_dset[global_idx, :], dtype=torch.float32)
        return trace, targets
    
    def __del__(self):
        self.file.close()

file_path = '....h5'
target = 'ASCAD_F_R'
target_byte = 2
start_traces_idx = 50000
end_traces_idx =  100000  
repeat_experi = 10
num_features = 100000

shared_layer_size1 = 800
shared_layer_size2 = 1000
output_size = 1
num_channels = 256
learning_rate = 0.001
epochs = 50
batch_size = 50
kernel = 8
device = torch.device(f'cuda')

for stride in [200, 500, 1000, 2000]:
    for i in range(start_traces_idx, end_traces_idx, 500):
        num_traces = min(i, end_traces_idx)
        dataset = H5Dataset(file_path=file_path, start_traces=0, end_traces=num_traces)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        for j in range(repeat_experi):
            print(f'Training with {num_traces} traces, experiment {j+1}/{repeat_experi}')
            model = convWIN_MCR(kernel=kernel, stride=stride, num_features=num_features).to(device)
            criterion = nn.MSELoss(reduction='none') 
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            start = time.time()
            losses_history = []
            for epoch in range(epochs):
                epoch_loss = torch.zeros(num_channels)
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs) 
                    loss_per_channel = criterion(outputs, labels.float()) 
                    weights = torch.ones(num_channels).float().to(device)
                    weighted_loss_per_channel = weights * loss_per_channel 
                    weighted_loss_per_channel = weighted_loss_per_channel.mean(dim=0)
                    total_loss = weighted_loss_per_channel.sum() 
                
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += weighted_loss_per_channel.cpu()
                        
                losses_history.append(epoch_loss / len(dataloader))

            end_time = time.time() - start
            print(end_time)
            loss_value = torch.stack(losses_history, dim=1).detach().cpu()
            loss_argsort = np.argsort(loss_value[:, epochs - 1])
            loss_ge = np.where(loss_argsort == 224)[0][0]
            print('loss_ge:',loss_ge)

            base_path = '.../'
            loss_path = f'convWIN-MCR_raw_{target}_hw_numft{num_features}_numtr{num_traces}_s{stride}_k{kernel}_{shared_layer_size1}_{shared_layer_size2}_{learning_rate}_{batch_size}_{epochs}epoch_{j}experi.npz'
            np.savez(os.path.join(base_path, loss_path), losses=loss_value, time=end_time)
            del model, losses_history, loss_value, loss_argsort, optimizer, criterion, loss_ge
            torch.cuda.empty_cache()
        gc.collect()
        del dataloader, dataset
