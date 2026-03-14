import utils

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json


class MySet(Dataset):
    def __init__(self, input_file, data_ratio=1.0, kernel_size=3):
        with open('./data/' + input_file, 'r') as f:
            self.content = [json.loads(line) for line in f]

        if data_ratio < 1.0:
            n = int(len(self.content) * data_ratio)
            self.content = self.content[:n]

        # remove trajectories shorter than kernel_size
        original_len = len(self.content)
        self.content = [x for x in self.content if len(x['lngs']) >= kernel_size]

        print("Filtered {} short trips".format(original_len - len(self.content)))

        self.lengths = [len(x['lngs']) for x in self.content]

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)


def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = [
    'lngs','lats','states','time_gap','dist_gap',
    'arrival_delay','departure_delay','speed','is_peak_hour',
    'temperature','apparent_temperature','precipitation',
    'rain','snowfall','windspeed','windgust','winddirection'
    ]

    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    for key in traj_attrs:
        seqs = [item[key] for item in data]
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate(seqs)

        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = utils.normalize(padded, key)

        traj[key] = torch.from_numpy(padded).float()

    traj['lens'] = lens.tolist()
    return attr, traj


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100
        chunks = (self.count + chunk_size - 1) // chunk_size

        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size:(i + 1) * chunk_size]
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size:(i + 1) * chunk_size] = partial_indices

        batches = (self.count + self.batch_size - 1) // self.batch_size
        for i in range(batches):
            yield self.indices[i * self.batch_size:(i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_loader(input_file, batch_size, data_ratio=1.0, kernel_size=3):
    dataset = MySet(
        input_file=input_file,
        data_ratio=data_ratio,
        kernel_size=kernel_size
    )

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=0,
        batch_sampler=batch_sampler,
        pin_memory=False
    )

    return data_loader