# Third-party references and adapted components
# Greco PyTorch implementation of the residual 3D U-Net topology used by CSBDeep/CARE. This attribution identifies the network reference; the PyTorch implementation and Greco integration are Greco contributions.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/csbdeep/LICENSE.txt
#
# Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
# Full license conditions and disclaimers are retained at the paths above.
#

"""TinyCARE's residual 12x2 3D U-Net, with the established layer topology."""
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import ARCHITECTURE, FORMAT


class TinyCARE(nn.Module):
    def __init__(self):
        super().__init__()
        # The middle and final decoder convolutions reduce channels. A generic
        # U-Net with two equal-width convolutions per level is NOT this model.
        channels = [
            ('down_level_0_no_0', 1, 12), ('down_level_0_no_1', 12, 12),
            ('down_level_1_no_0', 12, 24), ('down_level_1_no_1', 24, 24),
            ('middle_0', 24, 48), ('middle_2', 48, 24),
            ('up_level_1_no_0', 48, 24), ('up_level_1_no_2', 24, 12),
            ('up_level_0_no_0', 24, 12), ('up_level_0_no_2', 12, 12),
        ]
        self.convs = nn.ModuleDict(OrderedDict(
            (name, nn.Conv3d(cin, cout, 3, padding=1)) for name, cin, cout in channels))
        self.output = nn.Conv3d(12, 1, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(n % 4 for n in x.shape[2:]):
            raise ValueError('TinyCARE expects N,1,Z,Y,X with spatial sizes divisible by four.')
        def conv(name, value):
            return F.relu(self.convs[name](value))
        skip0 = conv('down_level_0_no_1', conv('down_level_0_no_0', x))
        skip1 = conv('down_level_1_no_1', conv('down_level_1_no_0', F.max_pool3d(skip0, 2)))
        value = conv('middle_2', conv('middle_0', F.max_pool3d(skip1, 2)))
        value = torch.cat((F.interpolate(value, scale_factor=2, mode='nearest'), skip1), dim=1)
        value = conv('up_level_1_no_2', conv('up_level_1_no_0', value))
        value = torch.cat((F.interpolate(value, scale_factor=2, mode='nearest'), skip0), dim=1)
        value = conv('up_level_0_no_2', conv('up_level_0_no_0', value))
        return self.output(value) + x


def save_weights(model, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(format=FORMAT, architecture=ARCHITECTURE,
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), temporary)
    temporary.replace(path)


def load_weights(model, path):
    path = Path(path)
    if path.suffix.lower() != '.pt':
        raise ValueError('TinyCARE checkpoints must use the current .pt format.')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or data.get('format') != FORMAT or data.get('architecture') != ARCHITECTURE:
        raise ValueError('Unsupported TinyCARE checkpoint architecture or format.')
    state = data.get('state_dict')
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or not v.is_floating_point()
                                         or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Invalid or nonfinite TinyCARE checkpoint tensors.')
    model.load_state_dict(state, strict=True)


