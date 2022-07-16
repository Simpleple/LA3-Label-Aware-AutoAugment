import torch.nn as nn
import torch
from collections import OrderedDict
from torch.nn import Parameter


class Neural_Predictor_Model(nn.Module):
    def __init__(self, input_dims=25, num_layers=10, layer_width=20):
        super(Neural_Predictor_Model, self).__init__()

        layer_lst = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_lst.append(('dense'+str(layer_idx), nn.Linear(input_dims, layer_width)))
            else:
                layer_lst.append(('dense'+str(layer_idx), nn.Linear(layer_width, layer_width)))

            layer_lst.append(('relu'+str(layer_idx), nn.ReLU()))

        layer_lst.append(('output', nn.Linear(layer_width, 1)))

        self.neural_predictor_model = nn.Sequential(OrderedDict(layer_lst))

    def forward(self, x):
        y_pred = self.neural_predictor_model(x)
        y_pred = torch.squeeze(y_pred, 1)
        return y_pred


class Neural_Predictor_Class_Model(nn.Module):
    def __init__(self, input_dims=49, num_layers=3, layer_width=100):
        super(Neural_Predictor_Class_Model, self).__init__()

        layer_lst = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer_lst.append(('dense'+str(layer_idx), nn.Linear(input_dims, layer_width)))
            else:
                layer_lst.append(('dense'+str(layer_idx), nn.Linear(layer_width, layer_width)))

            layer_lst.append(('relu'+str(layer_idx), nn.ReLU()))

        layer_lst.append(('output', nn.Linear(layer_width, 1)))

        self.neural_predictor_model = nn.Sequential(OrderedDict(layer_lst))

    def forward(self, x):
        y_pred = self.neural_predictor_model(x)
        y_pred = torch.squeeze(y_pred, 1)
        return y_pred


class Neural_Predictor_Embedding_Model(nn.Module):
    def __init__(self, num_augs=25, num_mags=10, embed_dim=10, linear_hidden=10, num_linears=3):
        super(Neural_Predictor_Embedding_Model, self).__init__()

        self.aug_embedding = nn.Embedding(num_augs, embed_dim)
        self.mag_embedding = nn.Embedding(num_mags, embed_dim)

        linear_layers = []
        for i in range(num_linears):
            if i == 0:
                linear_layers.append(('linear' + str(i), nn.Linear(embed_dim * 4, linear_hidden)))
            else:
                linear_layers.append(('linear' + str(i), nn.Linear(linear_hidden, linear_hidden)))
            linear_layers.append(('relu' + str(i), nn.ReLU()))
        linear_layers.append(('output', nn.Linear(linear_hidden, 1)))

        self.linear_model = nn.Sequential(OrderedDict(linear_layers))

    def forward(self, x):
        aug1_emb = self.aug_embedding(x[:,0])
        mag1_emb = self.aug_embedding(x[:,1])

        aug2_emb = self.aug_embedding(x[:,2])
        mag2_emb = self.aug_embedding(x[:,3])

        y = self.linear_model(torch.cat((aug1_emb, mag1_emb, aug2_emb, mag2_emb), dim=1))

        return y


class Neural_Predictor_Embedding_Class_Model(nn.Module):
    def __init__(self, num_augs=25, num_mags=10, num_cls=100, embed_dim=100, linear_hidden=100, num_linears=3):
        super(Neural_Predictor_Embedding_Class_Model, self).__init__()

        self.aug_embedding = nn.Embedding(num_augs, embed_dim)
        self.mag_embedding = nn.Embedding(num_mags, embed_dim)
        self.cls_embedding = nn.Embedding(num_cls, embed_dim)

        linear_layers = []
        for i in range(num_linears):
            if i == 0:
                linear_layers.append(('linear' + str(i), nn.Linear(embed_dim * 3, linear_hidden)))
            else:
                linear_layers.append(('linear' + str(i), nn.Linear(linear_hidden, linear_hidden)))
            linear_layers.append(('relu' + str(i), nn.ReLU()))
        linear_layers.append(('output', nn.Linear(linear_hidden, 1)))

        self.linear_model = nn.Sequential(OrderedDict(linear_layers))

    def forward(self, x):
        h = self.get_embedding(x)

        y = self.linear_model(h)

        return y

    def get_embedding(self, x):
        cls_emb = self.cls_embedding(x[:,4])

        aug1_emb = self.aug_embedding(x[:,0])
        mag1_emb = self.mag_embedding(x[:,1])

        aug2_emb = self.aug_embedding(x[:,2])
        mag2_emb = self.mag_embedding(x[:,3])

        aug1_emb = torch.cat((aug1_emb, mag1_emb), dim=1)
        aug2_emb = torch.cat((aug2_emb, mag2_emb), dim=1)
        aug_emb = (aug1_emb + aug2_emb) / 2

        return torch.cat((aug_emb, cls_emb), dim=1)
        # cls_emb = self.cls_embedding(x[:,4])
        # return torch.cat((aug1_emb, mag1_emb, aug2_emb, mag2_emb, cls_emb), dim=1)

    def get_hidden(self, x):
        hidden = self.get_embedding(x)
        for layer in self.linear_model[:-1]:
            hidden = layer(hidden)
        return hidden


class Neural_Predictor_Embedding_Class_Op_Model(nn.Module):
    def __init__(self, num_augs=16, num_mags=10, num_cls=100, embed_dim=100, linear_hidden=100, num_linears=3):
        super(Neural_Predictor_Embedding_Class_Op_Model, self).__init__()

        self.aug_embedding = nn.Embedding(num_augs, embed_dim, padding_idx=num_augs-1)
        self.cls_embedding = nn.Embedding(num_cls, embed_dim)

        linear_layers = []
        for i in range(num_linears):
            if i == 0:
                linear_layers.append(('linear' + str(i), nn.Linear(embed_dim * 2, linear_hidden)))
            else:
                linear_layers.append(('linear' + str(i), nn.Linear(linear_hidden, linear_hidden)))
            linear_layers.append(('relu' + str(i), nn.ReLU()))
        linear_layers.append(('output', nn.Linear(linear_hidden, 1)))

        self.linear_model = nn.Sequential(OrderedDict(linear_layers))

    def forward(self, x):
        h = self.get_embedding(x)

        y = self.linear_model(h)

        return y

    def get_embedding(self, x):
        cls_emb = self.cls_embedding(x[:,3])

        aug1_emb = self.aug_embedding(x[:,0])
        aug2_emb = self.aug_embedding(x[:,1])
        aug3_emb = self.aug_embedding(x[:,2])

        # aug_emb = torch.cat((aug1_emb, aug2_emb, aug3_emb), dim=1)
        aug_emb = (aug1_emb + aug2_emb + aug3_emb) #/ 3

        return torch.cat((aug_emb, cls_emb), dim=1)
        # cls_emb = self.cls_embedding(x[:,4])
        # return torch.cat((aug1_emb, mag1_emb, aug2_emb, mag2_emb, cls_emb), dim=1)

    def get_hidden(self, x):
        hidden = self.get_embedding(x)
        for layer in self.linear_model[:-1]:
            hidden = layer(hidden)
        return hidden


class Neural_Predictor_Embedding_Class_Op_Pair_Model(nn.Module):
    def __init__(self, num_augs=16, num_mags=10, num_cls=100, embed_dim=100, linear_hidden=100, num_linears=3):
        super(Neural_Predictor_Embedding_Class_Op_Pair_Model, self).__init__()

        self.aug_embedding = nn.Embedding(num_augs, embed_dim, padding_idx=num_augs-1)
        self.cls_embedding = nn.Embedding(num_cls, embed_dim)

        linear_layers = []
        for i in range(num_linears):
            if i == 0:
                linear_layers.append(('linear' + str(i), nn.Linear(embed_dim * 2, linear_hidden)))
            else:
                linear_layers.append(('linear' + str(i), nn.Linear(linear_hidden, linear_hidden)))
            linear_layers.append(('relu' + str(i), nn.ReLU()))
        linear_layers.append(('output', nn.Linear(linear_hidden, 1)))

        self.linear_model = nn.Sequential(OrderedDict(linear_layers))

    def forward(self, x):
        h = self.get_embedding(x)

        y = self.linear_model(h)

        return y

    def get_embedding(self, x):
        cls_emb = self.cls_embedding(x[:,2])

        aug1_emb = self.aug_embedding(x[:,0])
        aug2_emb = self.aug_embedding(x[:,1])

        # aug_emb = torch.cat((aug1_emb, aug2_emb, aug3_emb), dim=1)
        aug_emb = (aug1_emb + aug2_emb) #/ 3

        return torch.cat((aug_emb, cls_emb), dim=1)
        # cls_emb = self.cls_embedding(x[:,4])
        # return torch.cat((aug1_emb, mag1_emb, aug2_emb, mag2_emb, cls_emb), dim=1)

    def get_hidden(self, x):
        hidden = self.get_embedding(x)
        for layer in self.linear_model[:-1]:
            hidden = layer(hidden)
        return hidden


class Neural_Predictor_Embedding_Op_Model(nn.Module):
    def __init__(self, num_augs=16, num_mags=10, num_cls=100, embed_dim=100, linear_hidden=100, num_linears=3):
        super(Neural_Predictor_Embedding_Op_Model, self).__init__()

        self.aug_embedding = nn.Embedding(num_augs, embed_dim, padding_idx=num_augs-1)
        self.cls_embedding = nn.Embedding(num_cls, embed_dim)

        linear_layers = []
        for i in range(num_linears):
            if i == 0:
                linear_layers.append(('linear' + str(i), nn.Linear(embed_dim, linear_hidden)))
            else:
                linear_layers.append(('linear' + str(i), nn.Linear(linear_hidden, linear_hidden)))
            linear_layers.append(('relu' + str(i), nn.ReLU()))
        linear_layers.append(('output', nn.Linear(linear_hidden, 1)))

        self.linear_model = nn.Sequential(OrderedDict(linear_layers))

    def forward(self, x):
        h = self.get_embedding(x)

        y = self.linear_model(h)

        return y

    def get_embedding(self, x):
        aug1_emb = self.aug_embedding(x[:,0])
        aug2_emb = self.aug_embedding(x[:,1])
        aug3_emb = self.aug_embedding(x[:,2])

        # aug_emb = torch.cat((aug1_emb, aug2_emb, aug3_emb), dim=1)
        aug_emb = (aug1_emb + aug2_emb + aug3_emb) / 3

        return aug_emb
        # cls_emb = self.cls_embedding(x[:,4])
        # return torch.cat((aug1_emb, mag1_emb, aug2_emb, mag2_emb, cls_emb), dim=1)

    def get_hidden(self, x):
        hidden = self.get_embedding(x)
        for layer in self.linear_model[:-1]:
            hidden = layer(hidden)
        return hidden


if __name__ == '__main__':
    model = Neural_Predictor_Embedding_Class_Op_Model()
    x = torch.Tensor([[15, 15, 15, 0]]).long()
    print(model.get_embedding(x))