from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear


class HGTEncoder(nn.Module):
    """Heterogeneous Graph Transformer encoder."""

    def __init__(
        self,
        metadata,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        heads: int = 2,
        layers: int = 2,
    ) -> None:
        super().__init__()
        node_types, _ = metadata
        self.input_projection = nn.ModuleDict(
            {node_type: Linear(input_dim, hidden_dim) for node_type in node_types}
        )
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(layers)
            ]
        )
        self.output_projection = nn.ModuleDict(
            {node_type: Linear(hidden_dim, output_dim) for node_type in node_types}
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.input_projection[node_type](x).relu()
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: value.relu() for key, value in x_dict.items()}
        return {
            node_type: self.output_projection[node_type](x)
            for node_type, x in x_dict.items()
        }


class DotProductDecoder(nn.Module):
    """Scores candidate paper-concept links."""

    def forward(self, paper_z, concept_z, edge_label_index):
        paper_index, concept_index = edge_label_index
        return (paper_z[paper_index] * concept_z[concept_index]).sum(dim=-1)


class HGTLinkPredictor(nn.Module):
    def __init__(self, metadata, **encoder_kwargs) -> None:
        super().__init__()
        self.encoder = HGTEncoder(metadata, **encoder_kwargs)
        self.decoder = DotProductDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(
            z_dict["paper"],
            z_dict["concept"],
            edge_label_index,
        )
