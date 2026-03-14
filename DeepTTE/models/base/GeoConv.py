import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter

        self.build()

    def build(self):
        # embedding for state
        self.state_em = nn.Embedding(3, 2)

        self.process_coords = nn.Linear(16, 16)

        self.conv = nn.Conv1d(
            in_channels=16,
            out_channels=self.num_filter,
            kernel_size=self.kernel_size
        )

    def forward(self, traj, config):

        lngs = torch.unsqueeze(traj['lngs'], dim=2)
        lats = torch.unsqueeze(traj['lats'], dim=2)

        states = self.state_em(traj['states'].long())

        arrival_delay = torch.unsqueeze(traj['arrival_delay'], dim=2)
        departure_delay = torch.unsqueeze(traj['departure_delay'], dim=2)
        speed = torch.unsqueeze(traj['speed'], dim=2)
        peak = torch.unsqueeze(traj['is_peak_hour'], dim=2)

        temperature = torch.unsqueeze(traj['temperature'], dim=2)
        apparent_temperature = torch.unsqueeze(traj['apparent_temperature'], dim=2)
        precipitation = torch.unsqueeze(traj['precipitation'], dim=2)
        rain = torch.unsqueeze(traj['rain'], dim=2)
        snowfall = torch.unsqueeze(traj['snowfall'], dim=2)

        windspeed = torch.unsqueeze(traj['windspeed'], dim=2)
        windgust = torch.unsqueeze(traj['windgust'], dim=2)
        winddirection = torch.unsqueeze(traj['winddirection'], dim=2)

        # concatenate all trajectory features
        locs = torch.cat((
            lngs,
            lats,
            states,
            arrival_delay,
            departure_delay,
            speed,
            peak,
            temperature,
            apparent_temperature,
            precipitation,
            rain,
            snowfall,
            windspeed,
            windgust,
            winddirection
        ), dim=2)

        # map the features into 16-dim vector
        locs = torch.tanh(self.process_coords(locs))

        # conv expects (batch, channels, seq_len)
        locs = locs.permute(0, 2, 1)

        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        # calculate local distance features
        local_dist = utils.get_local_seq(
            traj['dist_gap'],
            self.kernel_size,
            config['dist_gap_mean'],
            config['dist_gap_std']
        )

        local_dist = torch.unsqueeze(local_dist, dim=2)

        conv_locs = torch.cat((conv_locs, local_dist), dim=2)

        return conv_locs