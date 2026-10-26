"""Inference-only membrane architecture; no trainer, metrics or loss objects."""
import torch
from monai.networks.nets import DynUNet


class _DirectDeepSupervision(DynUNet):
    def forward(self, x):
        out = self.output_block(self.skip_layers(x))
        return [out, *self.heads] if self.deep_supervision else out


class MembraneNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model = _DirectDeepSupervision(
            spatial_dims=3, in_channels=1, out_channels=1,
            kernel_size=(3, 3, 3, 3, 3, 3), strides=(1, 2, 2, 2, 2, 2),
            upsample_kernel_size=(1, 2, 2, 2, 2, 2),
            filters=[32, 64, 128, 256, 512, 1024], res_block=True,
            deep_supervision=True, deep_supr_num=2)
        self.target_shape = (160, 160, 160)
        self.rescale_patches = False

    def forward(self, x):
        if self.rescale_patches:
            raise ValueError('Greco membrane inference does not rescale patches.')
        # Preserve the former wrapper's exact stack/layout operations.
        inputs = torch.stack([sample[0].unsqueeze(0) for sample in x])
        predicted = self._model(inputs)[0]
        return [torch.stack([sample[0].unsqueeze(0) for sample in predicted])]
