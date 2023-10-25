from torch import Tensor, nn

class ResBlock3d(nn.Module):
    def __init__(
            self, 
            n_in: int = 1, 
            n_out: int = 24, 
            stride: int = 1,
            d_model: int = 64,
        ):

        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3,
                               stride=stride, padding=1)
        self.norm1 = nn.BatchNorm3d(n_out, momentum=0.1)
        # TODO: nn.InstanceNorm3d
        # self.norm1 = nn.InstanceNorm3d(n_out)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm1 = nn.GroupNorm(2, n_out)


        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(n_out, momentum=0.1)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm2 = nn.GroupNorm(2, n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out, momentum=0.1),
                # nn.LayerNorm(d_model),
                # nn.GroupNorm(2, n_out),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)
        return out