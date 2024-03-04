"""
@Author ：hhx
@Description ：model training and validation
"""
import torch
import os
from data_loader import load_data
from torch import nn
from torch import optim
from config import Configs
from models.TSSCD import Tsscd_FCN
from utils import *
from metrics import Evaluator, SpatialChangeDetectScore, TemporalChangeDetectScore

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Diceloss(nn.Module):
    def __init__(self, smooth=1.):
        super(Diceloss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=0).sum(dim=0)
        loss = (1 - ((2. * intersection + self.smooth) / (
                pred.sum(dim=0).sum(dim=0) + target.sum(dim=0).sum(dim=0) + self.smooth)))
        return loss.mean()


def trainModel(model, configs):
    train_dl, test_dl = load_data(configs)  # Load data
    # print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))

    loss_fn = nn.CrossEntropyLoss()  # classification loss function
    loss_ch_noch = Diceloss()  # changed loss function

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7)

    # Start train
    early_stopping = EarlyStopping()
    best_acc = 0
    best_spatialscore = 0
    best_temporalscore = 0
    for epoch in range(100):
        train_tqdm = tqdm(iterable=train_dl, total=len(train_dl))
        train_tqdm.set_description_str(f'Train epoch: {epoch}')
        train_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        train_loss1_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        train_loss2_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        for train_images, train_labels in train_tqdm:
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            pred = model(train_images.float())

            # time series has changed or not
            pre_label = torch.argmax(input=pred, dim=1)
            pre_No_change = pre_label.max(dim=1).values == pre_label.min(dim=1).values
            label_No_change = train_labels.max(dim=1).values == train_labels.min(dim=1).values

            loss1 = loss_fn(pred, train_labels.long())
            loss2 = loss_ch_noch(pre_No_change, label_No_change)
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss1_sum = torch.cat([train_loss1_sum, torch.unsqueeze(input=loss1, dim=-1)], dim=-1)
                train_loss2_sum = torch.cat([train_loss2_sum, torch.unsqueeze(input=loss2, dim=-1)], dim=-1)
                train_loss_sum = torch.cat([train_loss_sum, torch.unsqueeze(input=loss, dim=-1)], dim=-1)
                train_tqdm.set_postfix(
                    {'train loss': train_loss_sum.mean().item(), 'train loss1': train_loss1_sum.mean().item(),
                     'train loss2': train_loss2_sum.mean().item()})
        train_tqdm.close()
        valid_loss_sum, best_acc, best_spatialscore, best_temporalscore = validModel(test_dl, model, device, configs,
                                                                                     True, best_acc,
                                                                                     best_spatialscore,
                                                                                     best_temporalscore)
        early_stopping(valid_loss_sum)
        if early_stopping.early_stop:
            break


def validModel(test_dl, model, device, configs, saveModel=True, best_acc=0, best_spatialscore=0, best_temporalscore=0):
    evaluator = Evaluator(configs.classes)
    loss_fn = nn.CrossEntropyLoss()
    loss_ch_noch = Diceloss()
    with torch.no_grad():
        valid_tqdm = tqdm(iterable=test_dl, total=len(test_dl))
        valid_tqdm.set_description_str('Valid : ')
        valid_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        evaluator.reset()
        spatialscore = SpatialChangeDetectScore()
        temporalscore = TemporalChangeDetectScore(error_rate=1)
        for valid_images, valid_labels in valid_tqdm:
            valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)
            valid_pred = model(valid_images.float())

            pre_label = torch.argmax(input=valid_pred, dim=1)
            pre_No_change = pre_label.max(dim=1).values == pre_label.min(dim=1).values
            label_No_change = valid_labels.max(dim=1).values == valid_labels.min(dim=1).values
            loss1 = loss_fn(valid_pred, valid_labels.long())
            loss2 = loss_ch_noch(pre_No_change, label_No_change)
            valid_loss = loss1 + loss2

            evaluator.add_batch(valid_labels.cpu().numpy(), torch.argmax(input=valid_pred, dim=1).cpu().numpy())

            valid_loss_sum = torch.cat([valid_loss_sum, torch.unsqueeze(input=valid_loss, dim=-1)], dim=-1)
            valid_tqdm.set_postfix({'valid loss': valid_loss_sum.mean().item()})

            predList = torch.argmax(input=valid_pred, dim=1).cpu().numpy()
            labelList = valid_labels.cpu().numpy()

            for pre, label in zip(predList, labelList):
                predata, prechangepoints, pretypes = FilteringSeries(pre, method='Majority')
                labdata, labchangepoints, labtypes = FilteringSeries(label)
                spatialscore.addValue(labchangepoints, prechangepoints)
                temporalscore.addValue(labchangepoints, prechangepoints)

        valid_tqdm.close()

        # Evaluation Accuracy
        Acc = evaluator.Pixel_Accuracy()
        Acc_class, Acc_mean = evaluator.Pixel_Accuracy_Class()
        print('OA:', round(Acc, 4))
        print('AA', round(Acc_mean, 4))
        F1 = evaluator.F1()
        print('F1:', round(F1, 4))
        Kappa = evaluator.Kappa()
        print('Kappa:', round(Kappa, 4))
        spatialscore.getScore()
        spatial_f1 = spatialscore.spatial_f1
        print('spatial_PA: ', round(spatialscore.spatial_pa_change, 4))
        print('spatial_UA: ', round(spatialscore.spatial_ua_change, 4))
        print('spatial_f1: ', round(spatial_f1, 4))
        temporalscore.getScore()
        temporal_f1 = temporalscore.temporal_f1
        print('temporal_PA: ', round(temporalscore.temporal_pa_change, 4))
        print('temporal_UA: ', round(temporalscore.temporal_ua_change, 4))
        print('temporal_f1: ', round(temporalscore.temporal_f1, 4))
        if saveModel:
            if not os.path.exists(os.path.join('model_data')):
                os.mkdir(os.path.join('model_data'))
            if Acc > best_acc and spatial_f1 > best_spatialscore and temporal_f1 > best_temporalscore:
                torch.save(model.state_dict(), os.path.join('model_data', 'Best_TSSCD_model.pth'))
                best_acc = Acc
                best_spatialscore = spatial_f1
                best_temporalscore = temporal_f1
            return valid_loss_sum.mean().item(), best_acc, best_spatialscore, best_temporalscore
        else:
            return


if __name__ == '__main__':
    configs = Configs()
    model = Tsscd_FCN(10, 6, [128, 256, 512, 1024])
    model = model.to(device=device)

    trainModel(model, configs)
