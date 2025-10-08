
import torch 
from torch.nn import *
from h5py import File


#-----------------------
model_pth = r'weights.pth'
data_pth  = r'Data/electric_data.hdf5'
#-----------------------


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        layer_1_channels = 16
        layer_2_channels = 32
        layer_3_channels = 64

        self.conv = Sequential(
            Conv2d(1,                layer_1_channels, 3, 1, 1, bias = False), BatchNorm2d(layer_1_channels), SiLU(),
            Conv2d(layer_1_channels, layer_2_channels, 3, 1, 1, bias = False), BatchNorm2d(layer_2_channels), SiLU(),
            Conv2d(layer_2_channels, layer_3_channels, 3, 1, 1, bias = False), BatchNorm2d(layer_3_channels), SiLU(),

            Flatten(), Linear(55296, 96),
        )


    def forward(self, x):
     
        x = x.reshape(x.shape[0], 1, 9, 96)
        output = self.conv(x)
        return output 


#------------------------------------------------------------- Load Model and Weights
model = Net()
model.load_state_dict(torch.load(model_pth, weights_only=True, map_location=torch.device('cpu')))
model.eval()

#------------------------------------------------------------- Run Inference on Old Dataset
def run_inference_old_data():
    x = torch.from_numpy(File(data_pth, 'r')[f'features'][0, :, :]).float() # x ----> [dataset_length, 9, 96]
    # y = torch.from_numpy(File(data_pth, 'r')[f'target'][:, :]).float()    # Uncomment if you want to compare pred_y with y
    pred_y = model(x)

    return pred_y



print(run_inference_old_data())

