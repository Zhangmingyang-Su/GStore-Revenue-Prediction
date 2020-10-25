import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    def __init__(self, x_data, y_data, cat_cols1, cat_cols2, num_cols):

        """
        data: pandas data frame;
        cat_cols: list of string, the names of the categorical columns in the data, will be passed through the embedding layers;
        num_cols: list of string
        y_data: the target
        """
        self.n = x_data.shape[0]
        self.y = y_data.astype(np.float32).values.reshape(-1, 1)

        self.cat_cols1 = cat_cols1
        self.cat_cols2 = cat_cols2
        self.num_cols = num_cols

        self.num_X = x_data[self.num_cols].astype(np.float32).values
        self.cat_X1 = x_data[self.cat_cols1].astype(np.int64).values
        self.cat_X2 = x_data[self.cat_cols2].astype(np.int64).values


    def print_data(self):
        return self.num_X, self.cat_X1, self.cat_X2, self.y

    def __len__(self):
        """
        total number of samples
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.num_X[idx], self.cat_X1[idx], self.cat_X2[idx]]


class FeedForwardNN(nn.Module):
    def __init__(self, emb_dims1, emb_dims2, no_of_num, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts):
        """
        emb_dims:           List of two element tuples;
        no_of_num:          Integer, the number of continuous features in the data;
        lin_layer_sizes:    List of integers. The size of each linear layer;
        output_size:        Integer, the size of the final output;
        emb_dropout:        Float, the dropout to be used after the embedding layers.
        lin_layer_dropouts: List of floats, the dropouts to be used after each linear layer.
        """
        super().__init__()

        # embedding layers
        self.emb_layers1 = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims1])
        self.emb_layers2 = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims2])

        # 计算各个emb参数数量，为后续Linear layer的输入做准备
        self.no_of_embs1 = sum([y for x, y in emb_dims1])
        self.no_of_embs2 = sum([y for x, y in emb_dims2])
        self.no_of_num = no_of_num

        # 分支1
        self.branch1 = nn.Linear(self.no_of_embs1, lin_layer_sizes[0])
        self.branch1_2 = nn.Linear(lin_layer_sizes[0], lin_layer_sizes[1])
        nn.init.kaiming_normal_(self.branch1.weight.data)
        nn.init.kaiming_normal_(self.branch1_2.weight.data)

        # 分支2
        self.branch2 = nn.Linear(self.no_of_embs2, lin_layer_sizes[0] * 2)
        self.branch2_2 = nn.Linear(lin_layer_sizes[0] * 2, lin_layer_sizes[1] * 2)
        nn.init.kaiming_normal_(self.branch2.weight.data)
        nn.init.kaiming_normal_(self.branch2_2.weight.data)

        # 主分支
        self.main_layer1 = nn.Linear(lin_layer_sizes[1] * 3 + self.no_of_num, lin_layer_sizes[2])
        self.main_layer2 = nn.Linear(lin_layer_sizes[2], lin_layer_sizes[3])

        # batch normal
        self.branch_bn_layers1 = nn.BatchNorm1d(lin_layer_sizes[0])
        self.branch_bn_layers2 = nn.BatchNorm1d(lin_layer_sizes[0] * 2)
        self.main_bn_layer = nn.BatchNorm1d(lin_layer_sizes[2])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.dropout_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

        # Output layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

    def forward(self, num_data, cat_data1, cat_data2):
        # embedding categorical feature and cat them together
        x1 = [emb_layer(torch.tensor(cat_data1[:, i])) for i, emb_layer in enumerate(self.emb_layers1)]
        x1 = torch.cat(x1, 1)

        x1 = self.emb_dropout_layer(F.relu(self.branch1(x1)))
        x1 = self.branch_bn_layers1(x1)
        x1 = self.dropout_layers[0](F.relu(self.branch1_2(x1)))

        x2 = [emb_layer(torch.tensor(cat_data2[:, i])) for i, emb_layer in enumerate(self.emb_layers2)]
        x2 = torch.cat(x2, 1)

        x2 = self.emb_dropout_layer(F.relu(self.branch2(x2)))
        x2 = self.branch_bn_layers2(x2)
        x2 = self.dropout_layers[0](F.relu(self.branch2_2(x2)))

        main = torch.cat([x1, x2, num_data], 1)

        main = self.dropout_layers[1](F.relu(self.main_layer1(main)))
        main = self.main_bn_layer(main)
        main = self.dropout_layers[2](F.relu(self.main_layer2(main)))

        out = self.output_layer(main)
        return out
