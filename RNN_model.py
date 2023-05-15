import torch
import torch.nn as nn


#Example code
#input_size = 6 because we have 6 features 
# (station_arrival, station_departure, bus_line, current_time, day (mon-sun), month)

trainldr = [] #dummy variable for now, get through the input formatting stuff
numepoch = 100 #dummy value, probably needs to be tweaked
inpdim = 6 #number of input features
hidendim = 80 #also a dummy value for now, probably needs to be tweaked
layerdim = 1 #number of hidden layers, 1 for now, probably needs to be tweaked
outpdim = 3 #number of input variable
rnn = nn.RNN(inpdim, hidendim, layerdim, outpdim)
criter= nn.CrossEntropyLoss()
l_r = 0.01

optim = torch.optim.SGD(rnn.parameters(), lr=l_r)  
list(rnn.parameters())[0].size()
seqdim = 28  

itr = 0
for epoch in range(numepoch):
    for x, (imgs, lbls) in enumerate(trainldr):
        rnn.train()
        imgs = imgs.view(-1, seqdim, inpdim).requires_grad_()
        optim.zero_grad()
        outps = rnn(imgs)
        loss = criter(outps, lbls)
        loss.backward()

        optim.step()

        itr += 1

        if itr % 500 == 0:
            rnn.eval()       
            crrct = 0
            ttl = 0
            for imgs, lbls in testldr:
                imgs = imgs.view(-1, seqdim, inpdim)

                outps = rnn(imgs)
                _, predicted = torch.max(outps.data, 1)

                ttl += lbls.size(0)

                crrct += (predicted == lbls).sum()

            accu = 100 * crrct / ttl

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accu))
