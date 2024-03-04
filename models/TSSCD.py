from torch import nn
import torch


class Tsscd_FCN(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(Tsscd_FCN, self).__init__()
        self.out_channels = out_channels
        self.config = config
        c1, c2, c3, c4 = config
        # 第一层卷积
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/2
        )
        # 第二层卷积
        self.layer2 = nn.Sequential(
            nn.Conv1d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/4
        )

        # 第三层卷积
        self.layer3 = nn.Sequential(
            nn.Conv1d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/8
        )

        # 第四层卷积
        self.layer4 = nn.Sequential(
            nn.Conv1d(c3, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c4, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True)  # Downsampling 1/16
        )

        # 第六层使用卷积层取代FC层
        self.score_1 = nn.Conv1d(c4, out_channels, 1)
        self.score_2 = nn.Conv1d(c3, out_channels, 1)
        self.score_3 = nn.Conv1d(c2, out_channels, 1)

        # 第七层反卷积
        self.upsampling_2x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False)
        self.upsampling_4x = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1, bias=False)
        self.upsampling_8x = nn.ConvTranspose1d(out_channels, out_channels, 6, 4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        self.s1 = self.layer2(h)
        self.s2 = self.layer3(self.s1)  # 1/8
        self.s3 = self.layer4(self.s2)  # 1/16
        s3 = self.score_1(self.s3)
        s3 = self.upsampling_2x(s3)
        s2 = self.score_2(self.s2)
        s2 += s3
        s2 = self.upsampling_4x(s2)
        s1 = self.score_3(self.s1)
        score = s1 + s2
        score = self.upsampling_8x(score)
        return score
