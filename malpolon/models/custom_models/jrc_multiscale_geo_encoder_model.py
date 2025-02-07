"""This module provides a model to align features from multiscale geo-tagged data.

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>

License: GPLv3
Python version: 3.12.9
"""

class MultiScaleGeoEncoder(nn.Module):
    # Need to add 4 encoders:
    # 1. Pl@ntNet encoder (species scale)
    # 2. MME encoder (satellite scale)
    # 3. Any ResNet encoder (LUCAS scale)
    # 4. GeoCLIP encoder (GPS encoding)
    def __init__(self):
        pass

    def forward(self, x):
        pass