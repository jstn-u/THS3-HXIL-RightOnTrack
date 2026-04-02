import torch

class Config:
    data_folder = './data'

    sample_fraction = 0.20
    n_iterations = 1
    n_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    lstm_learning_rate = 0.0005
    residual_learning_rate = 0.0005
    weight_decay = 1e-5
    lstm_weight_decay = 1e-6
    lr_scheduler_factor = 0.5
    lr_scheduler_patience = 5
    early_stopping_patience = 50

    n_heads = 3
    node_embed_dim = 32
    gat_hidden = 32
    lstm_hidden = 64
    historical_dim = 16
    dropout = 0.3

    mtl_lambda = 0.5

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def haversine_meters(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r
