from stuff import *
import torch
import utils
import matplotlib.pyplot as plt

context_length = 100

data = SpeechDataset(context_length)
train_dataloader = DataLoader(data, batch_size=32, shuffle=True)

error = nn.L1Loss()

model = Full(context_length)

optimizer = torch.optim.Adam(model.parameters())
print(model)
losses = []

for e in range(10):
    for truth in train_dataloader:
        
        outputs = model(torch.clone(truth))

        loss = error(outputs, truth)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for i in range(10):
    test = data[i]
    model.eval()
    with torch.no_grad():
        result = model(test)
        utils.plot_waveform(test, file=f"{i}.png")
        utils.plot_waveform(result, file=f"{i}.png")
        plt.clf()

ax = plt.subplot()
ax.plot([i for i in range(len(losses))], losses )
plt.savefig("matplotlib.png")
