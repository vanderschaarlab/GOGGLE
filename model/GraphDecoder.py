from .RGCNConv import RGCNConv
from torch import nn
from dgl.nn import SAGEConv


class GraphDecoderHomo(nn.Module):
    def __init__(self, decoder_dim, decoder_l, device):
        super(GraphDecoderHomo, self).__init__()
        decoder = nn.ModuleList([])

        for i in range(decoder_l):
            if i == decoder_l-1:
                decoder.append(
                    SAGEConv(decoder_dim, 1, aggregator_type='mean', bias=True))
            else:
                decoder_dim_ = int(decoder_dim/2)
                # decoder_dim_ = int(decoder_dim)
                # decoder.append(SAGEConv(decoder_dim, decoder_dim_,
                #                aggregator_type='mean', bias=True, activation=nn.Tanh(), norm=nn.BatchNorm1d(decoder_dim_)))
                decoder.append(SAGEConv(decoder_dim, decoder_dim_,
                               aggregator_type='mean', bias=True, activation=nn.Tanh()))
                decoder_dim = decoder_dim_

        self.decoder = nn.Sequential(*decoder)

    def forward(self, graph_input, b_size):

        b_z, b_adj, b_edge_weight = graph_input

        for layer in self.decoder:
            b_z = layer(b_adj, feat=b_z, edge_weight=b_edge_weight)

        x_hat = b_z.reshape(b_size, -1)

        return x_hat


class GraphDecoderHet(nn.Module):
    def __init__(self, decoder_dim, decoder_l, n_edge_types, device):
        super(GraphDecoderHet, self).__init__()
        decoder = nn.ModuleList([])

        for i in range(decoder_l):
            if i == decoder_l-1:
                decoder.append(
                    RGCNConv(decoder_dim, 1, num_relations=n_edge_types+1, root_weight=False))
            else:
                decoder_dim_ = int(decoder_dim/2)
                decoder.append(
                    RGCNConv(decoder_dim, decoder_dim_, num_relations=n_edge_types+1, root_weight=False))
                decoder.append(nn.ReLU())
                decoder_dim = decoder_dim_

        self.decoder = nn.Sequential(*decoder)

    def forward(self, graph_input, b_size):

        b_z, b_edge_index, b_edge_weights, b_edge_types = graph_input

        h = b_z
        for layer in self.decoder:
            if not isinstance(layer, nn.ReLU):
                h = layer(h, b_edge_index, b_edge_types, b_edge_weights)
            else:
                h = layer(h)

        x_hat = h.reshape(b_size, -1)

        return x_hat
