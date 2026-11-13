"""Reference backend uses the upstream 16 gate relaxations, with saved wiring."""
import numpy as np
import torch
from torch import nn
from .vendor.difflogic_functional import bin_op_s

UPSTREAM_COMMIT = '469702c01ff0bfac9cdc6a395134252e11a56bd8'


class LogicLayer(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        if inputs < 2 or outputs * 2 < inputs:
            raise ValueError('Require inputs >= 2 and outputs*2 >= inputs')
        self.weights = nn.Parameter(torch.randn(outputs, 16))
        # Same random wiring construction as upstream, registered for checkpoints.
        c = torch.randperm(2 * outputs) % inputs
        c = torch.randperm(inputs)[c].reshape(2, outputs)
        self.register_buffer('wiring', c)

    def forward(self, x):
        w = self.weights.softmax(-1) if self.training else nn.functional.one_hot(
            self.weights.argmax(-1), 16).to(x.dtype)
        return bin_op_s(x[:, self.wiring[0]], x[:, self.wiring[1]], w)


class LogicHead(nn.Module):
    def __init__(self, concepts, classes, width=64, depth=2, tau=4.,
                 backend='reference', device='cpu'):
        super().__init__()
        if width % classes or width * 2 < concepts or depth < 1 or tau <= 0:
            raise ValueError('width must divide into classes, 2*width >= concepts; depth,tau positive')
        self.classes, self.tau, self.backend = classes, tau, backend
        sizes = [concepts] + [width] * depth
        layers = []
        for a, b in zip(sizes, sizes[1:]):
            if backend == 'upstream':
                try:
                    from difflogic import LogicLayer as OfficialLayer
                except ImportError as exc:
                    raise RuntimeError('Upstream requires a built difflogic CUDA extension; see README') from exc
                layer = OfficialLayer(a, b, device=device,
                    implementation='cuda' if str(device).startswith('cuda') else 'python')
                # Upstream indices are not included in state_dict. Register aliases;
                # checkpoint loading must also restore the indices tuple (below).
                layer.register_buffer('saved_a', layer.indices[0])
                layer.register_buffer('saved_b', layer.indices[1])
            else:
                layer = LogicLayer(a, b)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, c):
        for layer in self.layers:
            c = layer(c)
        return c.reshape(len(c), self.classes, -1).sum(-1) / self.tau

    def restore_wiring(self):
        for layer in self.layers:
            if self.backend == 'upstream':
                layer.indices = (layer.saved_a, layer.saved_b)
                if layer.implementation == 'cuda':
                    # CUDA backward adjacency depends on saved wiring as well.
                    adj = [[] for _ in range(layer.in_dim)]
                    for j, (a, b) in enumerate(zip(layer.saved_a.tolist(), layer.saved_b.tolist())):
                        adj[a].append(j); adj[b].append(j)
                    layer.given_x_indices_of_y_start = torch.tensor(
                        np.cumsum([0] + [len(v) for v in adj]), device=layer.weights.device)
                    layer.given_x_indices_of_y = torch.tensor(
                        [j for v in adj for j in v], device=layer.weights.device, dtype=torch.int64)

    def export(self, names):
        layers = []
        for layer in self.layers:
            wiring = layer.wiring if self.backend == 'reference' else torch.stack(layer.indices)
            layers.append({'a': wiring[0].tolist(), 'b': wiring[1].tolist(),
                           'gate': layer.weights.detach().argmax(-1).tolist()})
        return {'concept_names': list(names), 'classes': self.classes,
                'layers': layers, 'aggregation': 'equal_group_sum',
                'tie_break': 'lowest_class_index', 'tau': self.tau}


class Model(nn.Module):
    def __init__(self, input_shape, concepts, classes, width=64, depth=2,
                 tau=4., backend='reference', device='cpu'):
        super().__init__()
        if len(input_shape) == 1:
            self.reader = nn.Sequential(nn.Linear(input_shape[0], concepts), nn.Sigmoid())
        elif len(input_shape) == 3:
            self.reader = nn.Sequential(nn.Conv2d(input_shape[0], 8, 3, padding=1),
                nn.ReLU(), nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(),
                nn.Linear(8*8*8, 32), nn.ReLU(), nn.Linear(32, concepts), nn.Sigmoid())
        else:
            raise ValueError('Expected feature vectors or CHW images')
        self.head = LogicHead(concepts, classes, width, depth, tau, backend, device)

    def forward(self, x, hard=False, ste=False):
        q = self.reader(x)
        if hard:
            c = (q >= .5).to(q.dtype)
        elif ste:
            c = q + ((q >= .5).to(q.dtype) - q).detach()
        else:
            c = q
        return q, self.head(c)
