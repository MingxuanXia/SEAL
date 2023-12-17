import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class BottleNeckMLP(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim, num_classes, type1='bn', type2='wn'):
        super(BottleNeckMLP, self).__init__()
        assert type1 in ['bn', 'bn_relu', 'bn_relu_drop']
        # assert type2 in ['wn', 'linear']
        self.fc1 = nn.Linear(feature_dim, bottleneck_dim)
        self.fc1.apply(init_weights)

        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        if type2 == 'wn':
            self.fc2 = weightNorm(nn.Linear(bottleneck_dim, num_classes), name="weight")
            self.fc2.apply(init_weights)
        elif type2 == 'linear':
            self.fc2 = nn.Linear(bottleneck_dim, num_classes)
            self.fc2.apply(init_weights)
        else:
            self.fc2 = nn.Linear(bottleneck_dim, num_classes, bias=False)
            nn.init.xavier_normal_(self.fc2.weight)
        self.type1 = type1
        self.type2 = type2

    def forward(self, x):
        x = self.fc1(x)
        if 'bn' in self.type1:
            x = self.bn(x)
        if 'relu' in self.type1:
            x = self.relu(x)
        if 'drop' in self.type1:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

class BottleNeckMLP_Cont(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim, num_classes, low_dim, type1='bn', type2='wn'):
        super(BottleNeckMLP_Cont, self).__init__()
        assert type1 in ['bn', 'bn_relu', 'bn_relu_drop']
        # assert type2 in ['wn', 'linear']
        self.fc1 = nn.Linear(feature_dim, bottleneck_dim)
        self.fc1.apply(init_weights)

        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        if type2 == 'wn':
            self.fc2 = weightNorm(nn.Linear(bottleneck_dim, num_classes), name="weight")
            self.fc2.apply(init_weights)
        elif type2 == 'linear':
            self.fc2 = nn.Linear(bottleneck_dim, num_classes)
            self.fc2.apply(init_weights)
        else:
            self.fc2 = nn.Linear(bottleneck_dim, num_classes, bias=False)
            nn.init.xavier_normal_(self.fc2.weight)
        self.type1 = type1
        self.type2 = type2

        self.fc_cont = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, low_dim)
        )

    def forward(self, x):
        # contrastive
        feat_cont = self.fc_cont(x)

        # classifier
        logits = self.fc1(x)
        if 'bn' in self.type1:
            logits = self.bn(logits)
        if 'relu' in self.type1:
            logits = self.relu(logits)
        if 'drop' in self.type1:
            logits = self.dropout(logits)
        logits = self.fc2(logits)

        return logits, F.normalize(feat_cont, dim=1)


class SEAL(nn.Module):

    def __init__(self, net_q, net_k, cfg):
        super().__init__()
        
        self.encoder_q = net_q
        self.encoder_k = net_k  # momentum encoder

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(cfg.moco_queue, cfg.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(cfg.moco_queue))
        self.register_buffer("queue_rel", torch.zeros(cfg.moco_queue, dtype=torch.bool))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(cfg.num_classes, cfg.low_dim))
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, cfg):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * cfg.moco_m + param_q.data * (1. - cfg.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, is_rel, cfg):

        batch_size = is_rel.shape[0]
        ptr = int(self.queue_ptr)
        assert cfg.moco_queue % batch_size == 0  # for simplicity
        
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        self.queue_rel[ptr:ptr + batch_size] = is_rel

        ptr = (ptr + batch_size) % cfg.moco_queue  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle(self, x):
        # random shuffle index
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).cuda()
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def forward(self, img_q, img_k=None, Y_ori=None, is_rel_src=None, cfg=None, eval_only=False, is_rel=None, eval_nm=False):

        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        # for testing

        if eval_nm:
            with torch.no_grad():
                prototypes = self.prototypes.clone().detach()
                logits_prot = torch.mm(q, prototypes.t())
                score_prot = torch.softmax(logits_prot, dim=1)
            return output, score_prot

        batch_weight = is_rel_src.float()
        with torch.no_grad():  # no gradient 
            src_pred_cls = Y_ori
            tgt_pred_cls = torch.argmax(output, dim=1)
            pred_cls = batch_weight * src_pred_cls + (1 - batch_weight) * tgt_pred_cls
            pred_cls = pred_cls.long()

            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()
            logits_prot = torch.mm(q, prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)

            # update momentum prototypes with predicted classes
            for feat, label in zip(q, pred_cls):
                self.prototypes[label] = self.prototypes[label]*cfg.proto_m + (1-cfg.proto_m)*feat
            # normalize prototypes
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
            
            # compute key features 
            self._momentum_update_key_encoder(cfg)  # update the momentum encoder
            img_k, idx_unshuffle = self._batch_shuffle(img_k)  # # shuffle for making use of BN
            _, k = self.encoder_k(img_k)
            k = self._batch_unshuffle(k, idx_unshuffle)  # undo shuffle

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pred_cls_queue = torch.cat((pred_cls, pred_cls, self.queue_pseudo.clone().detach()), dim=0)
        is_rel_queue = torch.cat((is_rel, is_rel, self.queue_rel.clone().detach()), dim=0)
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pred_cls, is_rel, cfg)

        return output, features, pred_cls_queue, score_prot, is_rel_queue