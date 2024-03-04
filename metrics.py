"""
@Author ：hhx
@Description ：classification and change detection metrics
"""

import numpy as np

eps = np.finfo(np.float32).eps.item()


class Evaluator(object):
    """语义分割类指标"""

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):  # OA
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):  # 召回率
        Acc_classes = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc_classes)
        return Acc_classes, Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def F1(self):
        precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        f1 = 2 * precision * recall / (precision + recall)
        f1 = np.nanmean(f1)
        return f1

    def Kappa(self):
        p_o = self.Pixel_Accuracy()
        pre = np.sum(self.confusion_matrix, axis=0)
        label = np.sum(self.confusion_matrix, axis=1)
        p_e = (pre * label).sum() / (self.confusion_matrix.sum() * self.confusion_matrix.sum())
        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class SpatialChangeDetectScore(object):
    def __init__(self):
        self.spatial_f1 = None
        self.spatial_ua_Nochange = None
        self.spatial_ua_change = None
        self.spatial_pa_Nochange = None
        self.spatial_pa_change = None
        self.PreChange_LabChange = eps
        self.PreNoChange_LabChange = eps
        self.PreChange_LabNoChange = eps
        self.PreNoChange_LabNoChange = eps

    def addValue(self, label, pre):
        if len(label) != 0 and len(pre) != 0:
            self.PreChange_LabChange += 1
        elif len(label) == 0 and len(pre) == 0:
            self.PreNoChange_LabNoChange += 1
        elif len(label) == 0 and len(pre) != 0:
            self.PreChange_LabNoChange += 1
        elif len(label) != 0 and len(pre) == 0:
            self.PreNoChange_LabChange += 1

    def getScore(self):
        self.spatial_ua_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreChange_LabNoChange)
        self.spatial_ua_Nochange = self.PreNoChange_LabNoChange / (
                self.PreNoChange_LabNoChange + self.PreNoChange_LabChange)

        self.spatial_pa_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreNoChange_LabChange)
        self.spatial_pa_Nochange = self.PreNoChange_LabNoChange / (
                self.PreNoChange_LabNoChange + self.PreChange_LabNoChange)
        self.spatial_f1 = 2 * self.spatial_pa_change * self.spatial_ua_change / (
                self.spatial_pa_change + self.spatial_ua_change)


class TemporalChangeDetectScore(object):
    def __init__(self, series_length=64, error_rate=0):
        self.temporal_f1 = None
        self.temporal_ua_Nochange = None
        self.temporal_ua_change = None
        self.temporal_pa_Nochange = None
        self.temporal_pa_change = None

        self.PreChange_LabChange = eps
        self.PreNoChange_LabChange = eps
        self.PreChange_LabNoChange = eps
        self.PreNoChange_LabNoChange = eps
        self.series_length = series_length
        self.error_rate = error_rate

    def addValue(self, label, pre):

        for lab in label:
            for p_index in range(len(pre)):
                if abs(pre[p_index] - lab) <= self.error_rate:
                    pre[p_index] = lab
        better_pre = list(set(pre))  # 去重
        hot_label = np.zeros(self.series_length)
        if len(label) != 0:
            hot_label[np.array(label)] = 1  # 标签
        hot_pre = np.zeros(self.series_length)
        if len(better_pre) != 0:
            hot_pre[np.array(better_pre)] = 1  # 预测
        self.hot_label = hot_label
        self.hot_pre = hot_pre
        self.PreChange_LabChange += np.where((hot_pre == 1) & (hot_label == 1))[0].shape[0]
        self.PreNoChange_LabChange += np.where((hot_pre != 1) & (hot_label == 1))[0].shape[0]
        self.PreChange_LabNoChange += np.where((hot_pre == 1) & (hot_label != 1))[0].shape[0]
        self.PreNoChange_LabNoChange += np.where((hot_pre != 1) & (hot_label != 1))[0].shape[0]

    def getScore(self):
        self.temporal_ua_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreChange_LabNoChange)
        self.temporal_ua_Nochange = self.PreNoChange_LabNoChange / (
                self.PreNoChange_LabNoChange + self.PreNoChange_LabChange)

        self.temporal_pa_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreNoChange_LabChange)
        self.temporal_pa_Nochange = self.PreNoChange_LabNoChange / (
                self.PreNoChange_LabNoChange + self.PreChange_LabNoChange)
        self.temporal_f1 = 2 * self.temporal_pa_change * self.temporal_ua_change / (
                self.temporal_pa_change + self.temporal_ua_change)
