"""
@Author ：hhx
@Description ：Sample demo
"""

import glob
import torch
import matplotlib.pyplot as plt
import os
import tqdm
from utils import *
from models.TSSCD import Tsscd_FCN
from data_loader import MaskDataset
from torch.utils import data
from data_loader import load_data

plt.rc('font', family='Times New Roman', size=12)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def Datelist(freq):
    return pd.date_range('{}-{}'.format(2017, 12), '{}-{}'.format(2021, 11), freq=freq).strftime("%Y%m")


def testModel(test_dl, model, device):
    with torch.no_grad():
        test_tqdm = tqdm(iterable=test_dl, total=len(test_dl))
        for test_images, test_labels in test_tqdm:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            test_pred = model(test_images.float())
            test_pred = torch.argmax(input=test_pred, dim=1).cpu().numpy()

            for i in range(test_pred.shape[0]):
                pred = test_pred[i]
                predict_change_points = np.where((pred[1:] - pred[:-1]) != 0)[0]
                label = test_labels.cpu().numpy()[i]
                real_change_points = np.where((label[1:] - label[:-1]) != 0)[0]

                plt.figure(figsize=(12, 3))
                plt.subplot(121)
                plt.plot(Datelist('MS'), test_images[i].cpu().numpy().T)

                for cp in real_change_points:  # plot change point
                    plt.axvline(Datelist('MS')[cp], color='#4682B4', ls='--',
                                label=f"Real change date: {Datelist('MS')[cp]}")
                plt.legend()
                plt.xticks(Datelist('6M'), rotation=-45)
                plt.ylabel('Surface Reflectance')

                plt.subplot(122)
                plt.plot(Datelist('MS'), pred, label='Predict land cover change')
                plt.plot(Datelist('MS'), label, label='Real land cover change')

                for cp in predict_change_points:  # plot change point
                    plt.axvline(Datelist('MS')[cp], color='#D52445', ls='--',
                                label=f"Predict change date: {Datelist('MS')[cp]}")

                plt.xticks(Datelist('6M'), rotation=-45)
                plt.ylim(-1, 6)
                plt.yticks(range(6),
                           ['Water body', 'Woodland', 'Grassland', 'Bare soil', 'Impervious surface', 'Cropland'])
                plt.legend()
                plt.tight_layout()
                plt.show()



if __name__ == '__main__':

    test_ = np.load('dataset/test.npy').transpose(0, 2, 1)
    test_ds = MaskDataset(paths=test_, type='test')
    test_dl = data.DataLoader(dataset=test_ds, batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Tsscd_FCN(10, 6, [128, 256, 512, 1024])
    model = model.to(device=device)

    model_state_dict = torch.load(os.path.join('model_data', 'Best_TSSCD.pth'), map_location='cuda')
    model.load_state_dict(model_state_dict)

    model.eval()

    testModel(test_dl, model, device)
