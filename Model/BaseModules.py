import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchsummary import summary

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class NormReLU(nn.Module):
    def __init__(self):
        super(NormReLU, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(x)
        max_value = torch.abs(out).max(1, keepdim=True)[0] + 1e-5
        out = out / max_value
        return out

class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()
    def forward(self, x):
        return torch.tanh(x) * torch.sigmoid(x)

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        reshaped_input = input_seq.permute(0, 2, 1) # [b, C, L] -> [b, L, C]
        reshaped_input = reshaped_input.contiguous().view(-1, reshaped_input.size(-1))

        output = self.module(reshaped_input)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1)).permute(0, 2, 1)
        else:
            # (timesteps, samples, output_size)
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output

class TCN_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=None,
                 stride=1, dilation=1, causal=False,
                 dropout=None, activation='relu'):
        super(TCN_Block, self).__init__()
        if padding == None:
            padding = kernel_size // 2
        self.conv = weight_norm(nn.Conv1d(in_channel, out_channel, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding) if causal else None
        self.dropout = nn.Dropout(dropout) if dropout!=None else None

        if activation == 'norm_relu':
            self.activation = NormReLU()
        elif activation == 'wave_net':
            self.activation = WaveNet()
        else:
            self.activation = {'relu':nn.ReLU()}[activation]

    def forward(self, x):
        out = self.conv(x)
        if self.chomp != None:
            out = self.chomp(out)
        if self.dropout != None:
            out = self.dropout(out)
        out = self.activation(out)
        return out