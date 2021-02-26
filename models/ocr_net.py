from torch import nn
import torch
from einops import rearrange


class BidirectionalLSTM(nn.Module):
    """
    双向LSTM网络层。
    参数：
        input_size：输入特征尺寸
        hidden_size:隐藏层特征尺寸
        output_size：输出特征尺寸
    形状：
        input：（S,N,V）序列、批次、特征尺寸
        output：同输入
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        output = self.fc(recurrent)
        return output

class PositionEmbedding(nn.Module):

    def __init__(self, fm_size, head_channels, is_abs=True):
        super(PositionEmbedding, self).__init__()
        fm_size = (fm_size, fm_size) if not isinstance(fm_size,
                                                       tuple) else fm_size
        height, weight = fm_size
        scale = head_channels ** -0.5
        self.f_size = fm_size
        self.is_abs = is_abs
        if is_abs:
            self.height = nn.Parameter(torch.randn(height, 1, head_channels) * scale)
            self.weight = nn.Parameter(torch.randn(1, weight, head_channels) * scale)
        else:
            self.height = nn.Parameter(torch.randn(height * 2 - 1, head_channels) * scale)
            self.weight = nn.Parameter(torch.randn(weight * 2 - 1, head_channels) * scale)

    def forward(self, q):
        '''q b,h,s,v'''
        if self.is_abs:
            emb = self.height + self.weight
            h, w, c = emb.shape
            emb = emb.reshape(h * w, c)
            return torch.matmul(q, emb.transpose(0, 1))
        else:
            height, weight = self.f_size
            n, h, s, v = q.shape
            q = q.reshape(n, h, height, weight, v)
            r_w = self.relative(q, self.weight)
            n, h, x, i, y, j = r_w.shape
            r_w = r_w.permute(0, 1, 2, 4, 3, 5).reshape(n, h, x * y, i * j)

            q = q.permute(0, 1, 3, 2, 4)
            r_h = self.relative(q, self.height)
            r_h = r_h.permute(0, 1, 2, 4, 3, 5).reshape(n, h, x * y, i * j)

            return r_w + r_h

    def relative(self, q, rel_k):
        temp = torch.matmul(q, rel_k.transpose(0, 1))
        n, h, x, y, r = temp.shape
        temp = temp.reshape(n, h * x, y, r)
        temp = self.to_abs(temp).reshape(n, h, x, y, y)
        temp = temp.unsqueeze(dim=3)
        expand_shape = [-1] * len(temp.shape)
        expand_shape[3] = x
        return temp.expand(*expand_shape)

    @staticmethod
    def to_abs(input_):
        b, h, l, _ = input_.shape
        dd = {'device': input_.device, 'dtype': input_.dtype}
        col_pad = torch.zeros((b, h, l, 1), **dd)
        input_ = torch.cat((input_, col_pad), dim=3)
        _b, _h, _l, _c = input_.shape
        flat_x = input_.reshape(_b, _h, _l * _c)
        flat_pad = torch.zeros((b, h, l - 1), **dd)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
        final_x = final_x[:, :, :l, (l - 1):]
        return final_x


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, channels, feature_map_size, num_head=4, head_channels=128, is_abs=True):
        """
        图像多头自注意力
        :param channels: 通道数
        :param feature_map_size: 输入特诊图的尺寸
        :param num_head: 多头注意力的头数
        :param head_channels: 每个头的通道数
        :param is_abs: 是否使用绝对位置编码
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.channels = channels

        self.num_head = num_head
        self.scale = head_channels ** -0.5
        self.to_qkv = nn.Conv2d(self.channels, num_head * head_channels * 3, 1, bias=False)
        self.position = PositionEmbedding(feature_map_size, head_channels, is_abs)
        self.out_linear = nn.Conv2d(num_head * head_channels, self.channels, 1, bias=False)

    def forward(self, x):
        '''分出q、k、v'''
        n, _, height, weight = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        n, c, h, w = q.shape
        '''n,head v ,x,y  ---> n,head,s,v'''
        q = q.reshape(n, self.num_head, c // self.num_head, h, w).permute(0, 1, 3, 4, 2).reshape(n, self.num_head,
                                                                                                 h * w,
                                                                                                 c // self.num_head)
        k = k.reshape(n, self.num_head, c // self.num_head, h, w).permute(0, 1, 3, 4, 2).reshape(n, self.num_head,
                                                                                                 h * w,
                                                                                                 c // self.num_head)
        v = v.reshape(n, self.num_head, c // self.num_head, h, w).permute(0, 1, 3, 4, 2).reshape(n, self.num_head,
                                                                                                 h * w,
                                                                                                 c // self.num_head)

        qk = torch.matmul(q, k.transpose(-1, -2))  # n,h,s,v ===> n,h,s,s
        qk = qk * self.scale

        qr = self.position(q)

        attention = torch.matmul(torch.softmax(qk + qr, dim=-1), v)
        n, h, s, v = attention.shape

        attention = attention.permute(0, 1, 3, 2).reshape(n, h * v, height, weight)
        return self.out_linear(attention)


class DownSample(nn.Module):
    """下采样"""

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, input_):
        return self.layer(torch.cat(
            (input_[:, :, ::2, ::2], input_[:, :, ::2, 1::2], input_[:, :, 1::2, ::2], input_[:, :, 1::2, 1::2]), 1
        ))


class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, input_):
        return self.layer(input_) + input_


class BottleBlock(nn.Module):
    def __init__(
            self,
            *,
            channels,
            fmap_size,
            heads=8,
            dim_head=128,
            is_abs=True,
            activation=nn.LeakyReLU(0.1)
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 2*channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*channels),
            activation,
            MultiHeadSelfAttention(2*channels, fmap_size, heads, dim_head),
            nn.Conv2d(2*channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.activation = activation

    def forward(self, x):
        x = self.net(x)+x
        return self.activation(x)


class OcrNet(nn.Module):

    def __init__(self, num_class):
        super(OcrNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(3),
            DownSample(3, 64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            ResBlock(128),
            ResBlock(128),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            BottleBlock(channels=256, fmap_size=(6,18)),
            BottleBlock(channels=256, fmap_size=(6, 18)),
            BottleBlock(channels=256, fmap_size=(6, 18)),
            nn.Conv2d(256, 512, 3, (2, 1), 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, 1, (0,1), bias=False),
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512,256,128),
            BidirectionalLSTM(128,128,num_class)
        )

    def forward(self, input_):
        '''input_ of shape (3,48,144)'''
        input_ = self.cnn(input_)
        n, c, h, w = input_.shape  # n,c,h,w
        assert h == 1
        input_ = rearrange(input_, 'n c h w -> w n (c h)')
        return self.rnn(input_)


if __name__ == '__main__':
    m = OcrNet(70)
    x = torch.randn(36, 3, 48, 144)
    print(m(x).shape)
