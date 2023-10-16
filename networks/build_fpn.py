from torch import nn, Tensor
from typing import Optional, Union, Dict, List
from monai.networks.blocks.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool


class BackboneWithFPN(nn.Module):
    """
    Adds an FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.

    Same code as https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/backbone_utils.py
    Except that this class uses spatial_dims

    Args:
        backbone: backbone network
        return_layers: a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list: number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels: number of channels in the FPN.
        spatial_dims: 2D or 3D images
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels_list: List[int],
        out_channels: int,
        spatial_dims: Union[int, None] = None,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ) -> None:
        super().__init__()

        # if spatial_dims is not specified, try to find it from backbone.
        if spatial_dims is None:
            if hasattr(backbone, "spatial_dims") and isinstance(backbone.spatial_dims, int):
                spatial_dims = backbone.spatial_dims
            elif isinstance(backbone.conv1, nn.Conv2d):
                spatial_dims = 2
            elif isinstance(backbone.conv1, nn.Conv3d):
                spatial_dims = 3
            else:
                raise ValueError("Could not find spatial_dims of backbone, please specify it.")

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool(spatial_dims)

        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            spatial_dims=spatial_dims,
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Computes the resulted feature maps of the network.

        Args:
            x: input images

        Returns:
            feature maps after FPN layers. They are ordered from highest resolution first.
        """
        x = self.body(x)  # backbone
        y: Dict[str, Tensor] = self.fpn(x)  # FPN
        return y


