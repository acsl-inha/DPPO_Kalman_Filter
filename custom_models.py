import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm1d):
        m.eval()


class FcLayer(nn.Module):                                       # define fully connected class for model load
    def __init__(self, in_nodes, nodes):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(in_nodes, nodes)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.fc(x)
        out = self.act(out)
        return out


class WaveNET(nn.Module):                                       # define network class for model load
    def __init__(self, block, planes, nodes, num_classes=3):
        super(WaveNET, self).__init__()
        self.in_nodes = 5

        self.layer1 = self.make_layer(block, planes[0], nodes[0])
        self.layer2 = self.make_layer(block, planes[1], nodes[1])
        self.layer3 = self.make_layer(block, planes[2], nodes[2])

        self.fin_fc = nn.Linear(self.in_nodes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def make_layer(self, block, planes, nodes):

        layers = [block(self.in_nodes, nodes)]
        self.in_nodes = nodes
        for _ in range(1, planes):
            layers.append(block(self.in_nodes, nodes))

        return nn.Sequential(*layers)

    def forward_impl(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fin_fc(x)

        return x

    def forward(self, x):
        return self.forward_impl(x)


class FcLayerBn(nn.Module):                                       # define fully connected class for model load
    def __init__(self, in_nodes, nodes):
        super(FcLayerBn, self).__init__()
        self.fc = nn.Linear(in_nodes, nodes)
        self.bn1 = nn.BatchNorm1d(nodes)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.fc(x)
        out = self.bn1(out)
        out = self.act(out)
        return out


class WaveResNET(nn.Module):                                       # define network class for model load
    def __init__(self, block, planes, nodes, out_nodes=5, in_nodes=500):
        super(WaveResNET, self).__init__()
        self.in_nodes = in_nodes

        self.down_sample1 = self.down_sample(nodes[0])
        self.layer1 = self.make_layer(block, planes[0], nodes[0])
        self.down_sample2 = self.down_sample(nodes[1])
        self.layer2 = self.make_layer(block, planes[1], nodes[1])
        self.down_sample3 = self.down_sample(nodes[2])
        self.layer3 = self.make_layer(block, planes[2], nodes[2])

        self.fin_fc = nn.Linear(self.in_nodes, out_nodes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def make_layer(self, block, planes, nodes):

        layers = [block(self.in_nodes, nodes)]
        self.in_nodes = nodes
        for _ in range(1, planes):
            layers.append(block(self.in_nodes, nodes))

        return nn.Sequential(*layers)

    def down_sample(self, nodes):
        return nn.Sequential(nn.Linear(self.in_nodes, nodes), nn.BatchNorm1d(nodes))

    def forward_impl(self, x):
        identity = self.down_sample1(x)
        x = self.layer1(x)
        x = x.clone() + identity
        identity = self.down_sample2(x)
        x = self.layer2(x)
        x = x.clone() + identity
        identity = self.down_sample3(x)
        x = self.layer3(x)
        x = x.clone() + identity
        x = self.fin_fc(x)

        return x

    def forward(self, x):
        return self.forward_impl(x)


class CombinedModel(nn.Module):                                 # define radar, command combined model
    def __init__(self, model_radar, model_cmd, mean_cmd, std_cmd):
        super(CombinedModel, self).__init__()
        self.model_radar = model_radar
        self.model_cmd = model_cmd
        self.mean_cmd = torch.Tensor(mean_cmd).requires_grad_(False).to(device)
        self.std_cmd = torch.Tensor(std_cmd).requires_grad_(False).to(device)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model_radar(x)
        x = (x - self.mean_cmd)/self.std_cmd
        x = self.model_cmd(x)
        x = self.soft_max(x)
        return x
