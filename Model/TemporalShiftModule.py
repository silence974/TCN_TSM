import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalShift(nn.Module):
    def __init__(self, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            raise NotImplementedError
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
            out[:, :, 2*fold:] = x[:, :, 2*fold:]
        return out.view(nt, c, h, w)

if __name__ == '__main__':
    tsm1 = TemporalShift(n_segment=3, n_div=3, inplace=False)

    with torch.no_grad():
        x = torch.rand(1*3, 3, 1, 1)
        print(x)
        y = tsm1(x)
        print(y)
