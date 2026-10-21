import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import os
import matplotlib.pyplot as plt

class RuscoDataset():

    def __init__(self, data_file):

        all_data = pd.read_csv( data_file, dtype={'xDate':np.float64,
                                                    'yDate':np.float64,
                                                    't2m_now':np.float64,
                                                    't2m_next':np.float64 } )

        self.input_data = (all_data[ ["xDate", "yDate", "t2m_now"] ]).copy()
        self.pred_temps = (all_data[ ["t2m_next"] ]).copy()
    
    def __len__(self):
        return len(self.pred_temps)
    
    def __getitem__(self, indx):

        xDate = self.input_data["xDate"].iloc[indx].item()
        yDate = self.input_data["yDate"].iloc[indx].item()
        now_temp = self.input_data["t2m_now"].iloc[indx].item()
        next_temp = self.pred_temps["t2m_next"].iloc[indx].item()

        dati_input = (xDate, yDate, now_temp)

        return torch.as_tensor(dati_input, dtype=torch.float32), torch.as_tensor(next_temp, dtype=torch.float32)


class StdNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.lay1 = nn.Linear(3, 512)
        self.lay2 = nn.Linear(512, 512)
        self.lay3 = nn.Linear(512, 512)
        self.lay4 = nn.Linear(512, 1)

    def forward(self, x):

        x = torch.sigmoid(self.lay1(x))
        x = torch.sigmoid(self.lay2(x))
        x = torch.sigmoid(self.lay3(x))
        x = self.lay4(x)

        return x[:, 0]

def test_net(net, test_loader):

    avg_error = 0

    for j, data in enumerate(test_loader, 0):

        input_data, out_temp = data[0].to(device), data[1].to(device)
        
        net_output = net(input_data)
        avg_error += (torch.abs(net_output - out_temp)) * (1/len(test_loader))

    return avg_error


test_dataset = RuscoDataset("./poggio_rusco/test_rusco_data.csv")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

KiNet = StdNet()
net_parameters = torch.load("./NetParams/KiNetV1_256.pth",weights_only=True)
KiNet.load_state_dict(net_parameters)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

KiNet = KiNet.to(device)


print(f"test results: {test_net(KiNet, test_dataloader)}")

raw_test_data = pd.read_csv( "./poggio_rusco/test_rusco_data.csv", dtype = {'xDate':np.float32,
                                                                           'yDate':np.float32,
                                                                           't2m_now':np.float32,
                                                                           't2m_next':np.float32} )


x0 = 5.159420000000011
iterative_model = [x0]



for indx, entry in raw_test_data.iterrows():

    giorno = entry["xDate"]
    mese = entry["yDate"]

    input_entry = np.array([giorno, mese, x0])

    tensor_input_entry = torch.as_tensor( input_entry, dtype=torch.float32).unsqueeze(0)

    pred_temp = KiNet(tensor_input_entry.to(device))
    iterative_model.append(pred_temp[0].cpu().detach().numpy())

    x0 = pred_temp[0].cpu().detach().numpy()



plt.plot(raw_test_data["t2m_now"].values)
plt.plot(iterative_model)
plt.grid()
plt.show()