import torch
import torch.nn as nn


#Example code
#input_size = 6 because we have 6 features (station_arrival, station_departure, bus_line, current_time, day (mon-sun), month)
rnn= nn.RNN(input_size=6, hidden_size=1, num_layers = 1, bias = False, batch_first=True)
