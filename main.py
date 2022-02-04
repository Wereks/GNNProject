import numpy as np
import torch
import matplotlib.pyplot as plt


from preprocessing import prepare_data, pad_data, MolDataset
from model import GCN


def fun_ds(x):
    return torch.Tensor([x['docking_score']])

def fun_pic(x):
    return torch.Tensor([x['pIC50']])

np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set, test_set, valid_set = prepare_data()

train_set = MolDataset(pad_data(train_set), target_transform=fun_ds)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True, drop_last=True)

test_set = MolDataset(pad_data(test_set), target_transform=fun_ds)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=True, drop_last=True)

valid_set = MolDataset(pad_data(valid_set), target_transform=fun_ds)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=len(valid_set), shuffle=True, drop_last=True)


model = GCN((125, 10)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

epoches_t = []
epoches_v = []
for epoch in range(38):
    epoch_losses = []
    epoch_losses_v = []
    for step, (graph, label) in enumerate(train_loader):
        nodes, adj_mat = graph
        nodes = nodes.to(device)
        adj_mat = adj_mat.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        out = model((nodes, adj_mat))
        loss = torch.nn.functional.mse_loss(out, label)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    with torch.no_grad():
        model.eval()

        for step, (graph, label) in enumerate(valid_loader):
            nodes, adj_mat = graph
            nodes = nodes.to(device)
            adj_mat = adj_mat.to(device)
            label = label.to(device)

            out = model((nodes, adj_mat))
            loss = torch.nn.functional.mse_loss(out, label)
            epoch_losses_v.append(loss.item())

    epoches_v.append(np.mean(epoch_losses_v))
    epoches_t.append(np.mean(epoch_losses))
    print(f'Epoch: {epoch}  |  train loss: {np.mean(epoch_losses):.4f}')



MolCLR_loss_test = [95.85225677490234, 13.150615692138672, 1.4426742792129517, 1.6860735416412354, 1.5793769359588623, 1.521316409111023, 1.323847770690918, 1.4447743892669678, 1.166344165802002, 1.2128026485443115, 0.5241248607635498, 0.9986566305160522, 1.0756027698516846, 0.9030214548110962, 1.0242723226547241, 0.9052048325538635, 0.9035671949386597, 0.7611069679260254, 0.7376093864440918, 0.6198439598083496, 0.8336885571479797, 1.0978338718414307, 0.8715410232543945, 0.9500739574432373, 0.8448262214660645, 1.0601234436035156, 0.785558819770813, 0.9796552658081055, 0.4510122537612915, 0.8150068521499634, 0.7652606964111328, 0.5451418161392212, 0.6572586297988892, 0.6369725465774536, 0.6777887344360352, 0.7321267127990723, 1.145585298538208, 0.9610881805419922]
MOLCLR_loss_valid = [25.923425737595718, 1.9473596613928181, 1.6042804654860339, 1.470293333988316, 1.3168032785125126, 1.24091184612931, 1.3772845287986148, 1.1632722908297912, 1.2347208285173834, 1.2596956838835154, 1.1092856782951102, 1.1268217839942074, 1.1551775766524257, 0.9743193032725758, 0.9958229057046751, 0.9877372245914888, 0.8601579350351498, 0.8774889020730328, 0.828455167100919, 0.791781240346416, 0.8226735205050336, 0.8384229793453848, 0.7852543568769038, 0.8646701329591259, 0.765149810851015, 0.8148021859838473, 0.7240941299507949, 0.7762755258193869, 0.7881924324477745, 0.7370529664273293, 0.7177708918685155, 0.6918446269651123, 0.6874250595932765, 0.7059061809486111, 0.6700840021995519, 0.67129145631727, 0.6691540155979182, 0.678610074796424]


model.eval()

for step, (graph, label) in enumerate(test_loader):
    nodes, adj_mat = graph
    nodes = nodes.to(device)
    adj_mat = adj_mat.to(device)
    label = label.to(device)

    out = model((nodes, adj_mat))
    errors_a = ((out - label) / label).abs()
    acc = (errors_a < 0.1).float().mean()
    error_a = errors_a.mean()
    error_b = ((out - label) ** 2).mean()

    print(f'Accuracy for 10% error: {acc:.2f}, approximate error: {error_a:.2f}, mse: {error_b}')

plt.plot(MolCLR_loss_test, label="MolCLR_training")
plt.plot(MOLCLR_loss_valid, label="MolCLR_validate")
plt.plot(epoches_t, label="training")
plt.plot(epoches_v, label="validate")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 5)
plt.show()
