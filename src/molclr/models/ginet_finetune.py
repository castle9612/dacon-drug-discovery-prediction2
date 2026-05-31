import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# Define number of input categories (based on MolCLR)
NUM_ATOM_TYPE = 119
NUM_CHIRALITY = 3
NUM_BOND_TYPE = 5        # includes self-loop (type=4)
NUM_BOND_DIRECTION = 3   # 0, 1, 2

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # Add self-loop edge_attr: bond_type = 4 (self), direction = 0
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, activation='relu', dropout=0.0):
        super().__init__()
        
        # n_layers가 1 미만일 경우 예외 처리
        if n_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        # 활성화 함수 선택
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'softplus':
            act_fn = nn.Softplus
        elif activation == 'gelu': # GELU 와 같은 다른 활성화 함수도 고려해볼 수 있습니다.
            act_fn = nn.GELU
        else:
            raise ValueError("Unsupported activation function")

        layers = []
        in_dim = input_dim
        
        # n_layers가 1이면, hidden layer 없이 바로 출력
        if n_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # 첫 번째 레이어 (Input -> Hidden)
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) # BatchNorm 추가
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # 중간 히든 레이어들 (Hidden -> Hidden)
            # n_layers-2 만큼 반복
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim)) # BatchNorm 추가
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            # 마지막 출력 레이어 (Hidden -> Output)
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)



class GINet(nn.Module):
    def __init__(self, 
        task='regression', num_layer=5, emb_dim=300, 
        # feat_dim=512, 
        drop_ratio=0.0, pool='mean', pred_n_layer=10, pred_act='softplus',
        pred_dropout=0.3
    ):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        # self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pool}")

        # self.feat_lin = nn.Linear(emb_dim, feat_dim)

        out_dim = 1 if task == 'regression' else 2
        self.pred_n_layer = max(1, pred_n_layer)

        act_layer = nn.ReLU if pred_act == 'relu' else nn.Softplus

        self.pred_head = MLP(
            input_dim=emb_dim,
            hidden_dim=emb_dim // 2, # 히든 차원을 여기서 지정
            output_dim=out_dim,
            n_layers=pred_n_layer,
            activation=pred_act,
            dropout=pred_dropout  # 예측 헤드에도 드롭아웃 적용 가능
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for i in range(self.num_layer):
            h = self.gnns[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            if i == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.pred_head(h)
        return h, out

    def forward_features(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for i in range(self.num_layer):
            h = self.gnns[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            if i == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        return h

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            own_state[name].copy_(param)
