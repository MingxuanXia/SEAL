import os
import argparse
import time
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from utils.util import AverageMeter, accuracy, TrackMeter
from utils.util import set_seed

from utils.config import Config, DictAction
from losses import build_loss
from builder import build_optimizer
from models.build import build_model
from models.head import SEAL
from utils.util import format_time
from builder import build_logger
from datasets import build_divm_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--load', type=str, help='Load init weights for fine-tune (default: None)')
    parser.add_argument('--cfgname', help='specify log_file; for debug use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override the config; e.g., --cfg-options port=10001 k1=a,b k2="[a,b]"'
                             'Note that the quotation marks are necessary and that no white space is allowed.')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        filename = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join(dirname, filename)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    if args.cfgname is not None:
        cfg.cfgname = args.cfgname
    else:
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # seed
    if args.seed != 0:
        cfg.seed = args.seed
    elif not hasattr(cfg, 'seed'):
        cfg.seed = 43
    set_seed(cfg.seed)

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    return cfg


def adjust_lr(optimizer, step, tot_steps, gamma=10, power=0.75):
    decay = (1 + gamma * step / tot_steps) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['init_lr'] * decay


def set_optimizer(model, cfg):
    base_params = [v for k, v in model.named_parameters() if 'fc' not in k]
    head_params = [v for k, v in model.named_parameters() if 'fc' in k]
    param_groups = [{'params': base_params, 'lr': cfg.lr * 0.1},
                    {'params': head_params, 'lr': cfg.lr}]
    optimizer = build_optimizer(cfg.optimizer, param_groups)
    for param_group in optimizer.param_groups:
        param_group['init_lr'] = param_group['lr']
    return optimizer


def set_model(cfg):
    model = build_model(cfg.tgt_model)
    model.fc = build_model(cfg.tgt_head)
    return model


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag


def test(test_loader, model, criterion, epoch, logger, writer, model2=None, is_src=True):
    """ test target """
    model.eval()
    if model2 is not None:
        model2.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    all_pred = []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if is_src:
                logits = model(images)
            else:
                logits = model(images, eval_only=True)
            if model2 is not None:
                logits2 = model2(images)
                logits = (logits + logits2) / 2
            loss = criterion(logits, labels)

            pred = F.softmax(logits, dim=1)
            all_pred.append(pred.detach())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    all_pred = torch.cat(all_pred)
    mean_ent = (-all_pred * torch.log(all_pred + 1e-5)).sum(dim=1).mean().item() / np.log(all_pred.size(0))

    # writer
    writer.add_scalar(f'Loss/divm_test', losses.avg, epoch)
    writer.add_scalar(f'Entropy/divm_test', mean_ent, epoch)
    writer.add_scalar(f'Acc/divm_test', top1.avg, epoch)

    # logger
    time2 = time.time()
    test_time = format_time(time2 - time1)
    logger.info(f'Test at epoch [{epoch}] - test_time: {test_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_entropy: {mean_ent:.3f}, '
                f'test_Acc@1: {top1.avg:.2f}')
    return top1.avg, mean_ent


def test_class_acc(test_loader, model, criterion, it, logger, writer, cfg, model2=None, is_src=True):
    """ test target """
    model.eval()
    if model2 is not None:
        model2.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    all_pred, all_labels = [], []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            all_labels.append(labels)
            bsz = labels.shape[0]

            # forward
            if is_src:
                logits = model(images)
            else:
                logits = model(images, eval_only=True)
            if model2 is not None:
                logits2 = model2(images)
                logits = (logits + logits2) / 2
            loss = criterion(logits, labels)

            pred = F.softmax(logits, dim=1)
            all_pred.append(pred.detach())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)
    mean_ent = (-all_pred * torch.log(all_pred + 1e-5)).sum(dim=1).mean().item() / np.log(all_pred.size(0))
    pred_max = all_pred.max(dim=1).indices

    # class-wise acc
    class_accs = []
    all_eq = pred_max == all_labels
    for c in range(cfg.num_classes):
        mask_c = all_labels == c
        acc_c = all_eq[mask_c].float().mean().item()
        class_accs.append(round(acc_c * 100, 2))
    avg_acc = round(sum(class_accs) / len(class_accs), 2)

    # writer
    writer.add_scalar(f'Loss/ft_tgt_test', losses.avg, it)
    writer.add_scalar(f'Entropy/ft_tgt_test', mean_ent, it)
    writer.add_scalar(f'Acc/ft_tgt_test', top1.avg, it)

    # logger
    time2 = time.time()
    test_time = format_time(time2 - time1)
    logger.info(f'Test at iter [{it}] - test_time: {test_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_entropy: {mean_ent:.3f}, '
                f'test_Acc@1: {top1.avg:.2f}')
    logger.info(f'per class acc: {str(class_accs)}, avg_acc: {avg_acc}')
    return top1.avg, mean_ent, pred_max


def pred_target(test_loader, model, epoch, logger, cfg, model2=None, is_src=True):
    """ get predictions for target samples """
    model.eval()
    if model2 is not None:
        model2.eval()

    all_psl = []
    all_labels = []
    all_pred = []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = images.shape[0]

            # forward
            if is_src:
                logits = model(images)
            else:
                logits = model(images, eval_only=True)
            if model2 is not None:
                output2 = model2(images)
                logits = (logits + output2) / 2

            psl = logits.max(dim=1).indices
            pred = F.softmax(logits, dim=1)

            if epoch == 0:
                src_idx = torch.sort(pred, dim=1, descending=True).indices
                for i in range(bsz):
                    pred[i, src_idx[i, cfg.topk:]] = \
                        (1.0 - pred[i, src_idx[i, :cfg.topk]].sum()) / (cfg.num_classes - cfg.topk)

            all_psl.append(psl)
            all_labels.append(labels)
            all_pred.append(pred.detach())
    all_psl = torch.cat(all_psl)
    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)
    psl_acc = (all_psl == all_labels).float().mean()

    # logger
    time2 = time.time()
    pred_time = format_time(time2 - time1)
    logger.info(f'Predict target at epoch [{epoch}]: psl_acc: {psl_acc:.2f}, time: {pred_time}')
    return all_psl, all_labels, all_pred


def warmup(warmup_loader, model, optimizer, epoch, logger, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    num_iters = len(warmup_loader)

    model.train()
    t1 = end = time.time()
    for batch_idx, (inputs, labels) in enumerate(warmup_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        # outputs = model(inputs)
        outputs, _ = model.module.encoder_q(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                        f'Batch time: {batch_time.avg:.2f}, '
                        f'lr: {lr:.6f}, '
                        f'loss: {losses.avg:.3f}')

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(f'Epoch [{epoch}] - train_time: {epoch_time}, '
                f'train_loss: {losses.avg:.3f}\n')


def dist_train(train_loader, true_labels, model, optimizer, epoch, logger, cfg, pred_mem, rel_stats, warmup):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()
    losses_ent = AverageMeter()
    losses_cont = AverageMeter()
    acc_prot = AverageMeter()

    num_iters = len(train_loader)

    model.train()
    t1 = end = time.time()
    for batch_idx, (images, labels, probs, indices) in enumerate(train_loader):
        imgs1, imgs2, Y_ori = images[0].cuda(), images[1].cuda(), labels.cuda()
        targets = pred_mem[indices, :]
        bsz = imgs1.shape[0]
        is_rel_src = rel_stats['is_rel_src'][indices]
        is_rel_flx = rel_stats['is_rel_flx'][indices]
        is_rel_tgt = rel_stats['is_rel_tgt'][indices]

        # forward
        logits, features_cont, pred_cls_queue, score_prot, is_rel_queue = \
            model(imgs1, img_k=imgs2, Y_ori=Y_ori, is_rel_src=is_rel_src, cfg=cfg, eval_only=False, is_rel=(is_rel_src.bool() + is_rel_flx.bool()))
        pred_tgt = F.softmax(logits, dim=1)

        # self-distilled loss
        loss_kl = nn.KLDivLoss(reduction='batchmean')(pred_tgt.log(), targets)

        # mutual information loss
        loss_entropy = (-pred_tgt * torch.log(pred_tgt + 1e-5)).sum(dim=1).mean()
        pred_mean = pred_tgt.mean(dim=0)
        loss_gentropy = torch.sum(-pred_mean * torch.log(pred_mean + 1e-5))
        loss_entropy -= loss_gentropy
        loss = loss_kl + loss_entropy

        # update metric
        losses.update(loss.item(), bsz)
        losses_kl.update(loss_kl.item(), bsz)
        losses_ent.update(loss_entropy.item(), bsz)

        # mixup loss
        if cfg.is_mu:
            alpha = 0.3
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(bsz).cuda()
            mixed_images = lam * imgs1 + (1 - lam) * imgs1[index, :]
            mixed_targets = (lam * pred_tgt + (1 - lam) * pred_tgt[index, :]).detach()

            update_batch_stats(model, False)
            mixed_logits, _ = model.module.encoder_q(mixed_images)
            update_batch_stats(model, True)
            mixed_pred_tgt = F.softmax(mixed_logits, dim=1)
            loss_mix_kl = nn.KLDivLoss(reduction='batchmean')(mixed_pred_tgt.log(), mixed_targets)
            loss += loss_mix_kl

        # contrastive loss
        if warmup:
            q = features_cont[:bsz]
            k = features_cont[bsz:bsz*2]
            queue = features_cont[bsz*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            cont_logits = torch.cat([l_pos, l_neg], dim=1)
            cont_logits /= cfg.temperature

            cont_labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss_cont = cfg.cont_weight * F.cross_entropy(cont_logits, cont_labels)
        else:
            pseudo_target_cont = pred_cls_queue.contiguous().view(-1, 1)
            batch_weight = (is_rel_src.bool() + is_rel_flx.bool()).float()

            mask_ur = torch.eq(pseudo_target_cont[:bsz], pseudo_target_cont[bsz:].T).float().cuda().detach()
            mask_re = copy.deepcopy(mask_ur)
            mask_re = batch_weight.unsqueeze(1).repeat(1, mask_re.shape[1]) * mask_re  # remove row-wise
            mask_re = is_rel_queue[bsz:].view(1, -1).repeat(mask_re.shape[0], 1) * mask_re  # remove column-wise
            if epoch >= cfg.knn_start and cfg.knn > 0:
                _, knn_index = torch.topk(features_cont[:bsz] @ features_cont[bsz:].T, k=cfg.knn, dim=-1, largest=True)
                mask_knn = torch.scatter(torch.zeros_like(mask_ur).cuda(), 1, knn_index, 1)
                mask_knn = (1 - batch_weight).unsqueeze(1).repeat(1, mask_knn.shape[1]) * mask_knn  # remove row-wise reliable
            else:
                mask_knn = torch.zeros_like(mask_ur).cuda()
            mask = mask_re + cfg.knn_weight * (mask_ur.bool() + mask_knn.bool()).float()
            mask.fill_diagonal_(1)
            mask[mask > 1] = 1
            mask = mask / mask.sum(dim=1, keepdim=True)

            sim = torch.exp(torch.mm(features_cont[:bsz], features_cont[bsz:].t()) / cfg.temperature) 
            sim_probs = sim / sim.sum(dim=1, keepdim=True)
            loss_cont_vec = - (torch.log(sim_probs + 1e-7) * mask).sum(dim=1)
            loss_cont = cfg.cont_weight * loss_cont_vec.mean()
        
        losses_cont.update(loss_cont.item(), bsz)
        loss += loss_cont

        compare = (torch.argmax(score_prot, dim=1) == true_labels[indices])
        acc_prot.update(sum(compare) / len(compare), bsz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                f'Batch time: {batch_time.avg:.2f}, '
                f'lr: {lr:.6f}, '
                f'loss_kl: {losses_kl.avg:.3f}, '
                f'loss_ent: {losses_ent.avg:.3f}, '
                f'loss_cont: {losses_cont.avg:.3f}, '
                f'acc_prot: {acc_prot.avg:.3f}, '
                f'distill_loss: {losses.avg:.3f}'
            )

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'loss_kl: {losses_kl.avg:.3f}, '
        f'loss_ent: {losses_ent.avg:.3f}, '
        f'loss_cont: {losses_cont.avg:.3f}, '
        f'acc_prot: {acc_prot.avg:.3f}, '
        f'distill_loss: {losses.avg:.3f}'
    )


def eval_train(eval_loader, model, rel_stats):
    model.eval()
    losses = []
    confs = []
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):  # shuffle=False
            inputs = inputs.cuda()  # weak transform
            targets = targets.cuda()

            outputs = model(inputs, eval_only=True)
            conf = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, targets)
            losses.append(loss)
            confs.append(conf)

    losses = torch.cat(losses)
    rel_stats['loss'] = losses
    confs = torch.cat(confs)
    rel_stats['conf'] = confs


def train(train_loader, model, criterion, optimizer, epoch, logger, cfg, rel_stats):
    batch_time = AverageMeter()
    losses_src = AverageMeter()
    losses_tgt = AverageMeter()
    losses_cont = AverageMeter()
    t1 = end = time.time()

    num_iters = len(train_loader)
    model.train()

    for batch_idx, (images, labels, probs, indices) in enumerate(train_loader):
        imgs1, imgs2, Y_ori, probs = images[0].cuda(), images[1].cuda(), labels.cuda(), probs.cuda()
        bsz = imgs1.shape[0]
        is_rel_src = rel_stats['is_rel_src'][indices]
        is_rel_flx = rel_stats['is_rel_flx'][indices]
        is_rel_tgt = rel_stats['is_rel_tgt'][indices]

        # forward
        logits, features_cont, pred_cls_queue, score_prot, is_rel_queue = \
            model(imgs1, img_k=imgs2, Y_ori=Y_ori, is_rel_src=is_rel_src, cfg=cfg, eval_only=False, is_rel=(is_rel_src.bool() + is_rel_flx.bool()))
        

        # nearest-centroid classification logits
        sp_temp_scale = score_prot ** (1 / cfg.temperature)
        pred_cont = sp_temp_scale / sp_temp_scale.sum(dim=1, keepdim=True)


        # constructing label graph
        pseudo_target_cont = pred_cls_queue.contiguous().view(-1, 1)
        batch_weight = (is_rel_src.bool() + is_rel_flx.bool()).float()
        mask_ur = torch.eq(pseudo_target_cont[:bsz], pseudo_target_cont[bsz:].T).float().cuda().detach()
        mask_re = copy.deepcopy(mask_ur)
        mask_re = batch_weight.unsqueeze(1).repeat(1, mask_re.shape[1]) * mask_re  # remove row-wise
        mask_re = is_rel_queue[bsz:].view(1, -1).repeat(mask_re.shape[0], 1) * mask_re  # remove column-wise
        if epoch >= cfg.knn_start and cfg.knn > 0:
            _, knn_index = torch.topk(features_cont[:bsz] @ features_cont[bsz:].T, k=cfg.knn, dim=-1, largest=True)
            mask_knn = torch.scatter(torch.zeros_like(mask_ur).cuda(), 1, knn_index, 1)
            mask_knn = (1 - batch_weight).unsqueeze(1).repeat(1, mask_knn.shape[1]) * mask_knn  # remove row-wise reliable
        else:
            mask_knn = torch.zeros_like(mask_ur).cuda()
        mask = mask_re + cfg.knn_weight * (mask_ur.bool() + mask_knn.bool()).float()
        mask.fill_diagonal_(1)
        mask[mask > 1] = 1
        mask = mask / mask.sum(dim=1, keepdim=True)
        # constructing feature graph
        sim = torch.exp(torch.mm(features_cont[:bsz], features_cont[bsz:].t()) / cfg.temperature) 
        sim_probs = sim / sim.sum(dim=1, keepdim=True)
        # contrastive loss
        loss_cont_vec = - (torch.log(sim_probs + 1e-7) * mask).sum(dim=1)
        loss_cont = cfg.cont_weight * loss_cont_vec.mean()
        losses_cont.update(loss_cont.item(), bsz)

        # easily-adaptable sample loss
        pred_cont_onehot = F.one_hot(torch.argmax(pred_cont, dim=1), cfg.num_classes).float().cuda().detach()
        pred_src_onehot = F.one_hot(Y_ori, cfg.num_classes).float().cuda().detach()
        z_src = pred_src_onehot
        loss_src = criterion(logits[is_rel_src], z_src[is_rel_src])
        losses_src.update(loss_src.item(), bsz)
        loss = loss_cont + loss_src

        # model-confident sample loss
        if sum(is_rel_flx) != 0:
            pred_tgt = rel_stats['conf'][indices]
            pconf, pconf_label = torch.max(pred_tgt, dim=1)
            pred_tgt_onehot = F.one_hot(pconf_label, cfg.num_classes).float().cuda().detach()
            z_tgt = pconf.view(-1, 1) * pred_tgt_onehot + (1 - pconf).view(-1, 1) * pred_cont
            loss_tgt = criterion(logits[is_rel_flx], z_tgt[is_rel_flx])
            losses_tgt.update(loss_tgt.item(), bsz)
            loss += loss_tgt
            is_ur = ~(is_rel_src + is_rel_flx)
        else:
            losses_tgt.update(0.0)
            is_ur = ~is_rel_src

        # under-adapted sample loss
        z_ur = pred_cont
        loss_ur = cfg.ur_weight * criterion(logits[is_ur], z_ur[is_ur])
        loss += loss_ur

        # mixup loss
        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        pseudo_label = z_src
        if sum(is_rel_flx) != 0:
            pseudo_label[is_rel_flx] = z_tgt[is_rel_flx]
        pseudo_label[is_ur] = z_ur[is_ur]
        if cfg.is_mu:
            idx = torch.randperm(bsz)
            imgs1_rand = imgs1[idx]
            pseudo_label_rand = pseudo_label[idx]
            img1_mix = l * imgs1 + (1 - l) * imgs1_rand      
            pseudo_label_mix = l * pseudo_label + (1 - l) * pseudo_label_rand
            logits_mix, _ = model.module.encoder_q(img1_mix)
            loss_mix = criterion(logits_mix, pseudo_label_mix)
            loss += loss_mix

        # consistency regularization loss
        if cfg.is_cr:
            logits_s, _ = model.module.encoder_q(imgs2)
            loss_cr = criterion(logits_s, pseudo_label)
            loss += loss_cr

        # diversity loss
        if cfg.div:
            logits_sm = F.softmax(logits, dim=1)
            pred_mean = logits_sm.mean(dim=0)
            loss_gentropy = -torch.sum(-pred_mean * torch.log(pred_mean + 1e-5))
            loss += loss_gentropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                f'Batch time: {batch_time.avg:.2f}, '
                f'lr: {lr:.6f}, '
                f'loss_src: {losses_src.avg:.3f}, '
                f'loss_tgt: {losses_tgt.avg:.3f}, '
                f'loss_cont: {losses_cont.avg:.3f}, '
            )

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'loss_src: {losses_src.avg:.3f}, '
        f'loss_tgt: {losses_tgt.avg:.3f}, '
        f'loss_cont: {losses_cont.avg:.3f}, '
    )

def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)
    cudnn.benchmark = True

    # write cfg
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # logger
    logger = build_logger(cfg.work_dir, cfgname=f'train')
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, f'tensorboard'))

    '''
    # -----------------------------------------
    # build model & optimizer
    # -----------------------------------------
    '''
    # build source model & load weights
    src_model = build_model(cfg.src_model)
    src_model = torch.nn.DataParallel(src_model).cuda()

    print(f'==> Loading checkpoint "{cfg.load}"')
    ckpt = torch.load(cfg.load, map_location='cuda')
    src_model.load_state_dict(ckpt['model_state'])

    # build target model
    net_q = set_model(cfg)
    net_k = set_model(cfg)
    model = SEAL(net_q, net_k, cfg)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = set_optimizer(model, cfg)

    train_criterion = build_loss(cfg.loss.train).cuda()
    test_criterion = build_loss(cfg.loss.test).cuda()
    print('==> Model built.')

    '''
    # -----------------------------------------
    # build dataset/dataloader
    # -----------------------------------------
    '''
    test_loader = build_divm_loader(cfg, mode='test')

    '''
    # -----------------------------------------
    # Predict target 
    # -----------------------------------------
    '''
    given_labels, true_labels, pred_mem = pred_target(test_loader, src_model, 0, logger, cfg)
    pred_probs, _ = torch.max(pred_mem, dim=1)
    warmup_loader = build_divm_loader(cfg, mode='warmup', psl=given_labels)
    train_loader = build_divm_loader(cfg, mode='train', probs=pred_probs.cpu(), psl=given_labels)
    eval_train_loader = build_divm_loader(cfg, mode='eval_train', psl=given_labels)

    '''
    # -----------------------------------------
    # Start target training
    # -----------------------------------------
    '''
    
    cfg.num_instances = given_labels.shape[0]
    rel_stats = {
        'loss': torch.zeros(cfg.num_instances).cuda(),  # sample loss
        'conf': torch.zeros([cfg.num_instances, cfg.num_classes]).cuda(),  # sample confidence
        'is_rel_src': torch.ones(cfg.num_instances).bool().cuda(),  # whether easily-adaptable
        'is_rel_tgt': torch.zeros(cfg.num_instances).bool().cuda(),  # whether with high confidence (fix threshold)
        'is_rel_flx': torch.zeros(cfg.num_instances).bool().cuda(),  # whether model-confident (flexible threshold)
        'cls2tau_ini': cfg.tau_conf * torch.ones(cfg.num_classes).cuda(),
        'cls2tau': cfg.tau_conf * torch.ones(cfg.num_classes).cuda()  # class-wise flexible threshold
    }
    
    print("==> Start training...")
    model.train()

    test_meter = TrackMeter()
    start_epoch = 1

    for epoch in range(start_epoch, cfg.epochs + 1):
        adjust_lr(optimizer, epoch, cfg.epochs, power=1.5)

        # momentum update pred_mem
        if epoch % cfg.pred_interval == 0:
            _, _, pred_t = pred_target(test_loader, model, epoch, logger, cfg, is_src=False)
            pred_mem = cfg.ema * pred_mem + (1 - cfg.ema) * pred_t
            model.train()
        
        if epoch <= cfg.warmup_epochs:
            if cfg.CE_warmup and epoch == 1:
                # warmup using source-predicted label
                warmup(warmup_loader, model, optimizer, epoch, logger, cfg)
            else:
                # warmpup using self-distilled loss + contrastive loss
                dist_train(train_loader, true_labels, model, optimizer, epoch, logger, cfg, pred_mem, rel_stats, warmup=True)
                # true_labels only used for monitoring accuracy
        
        else:
            if epoch > cfg.warmup_epochs + 1:
                logger.info(f'Start distill training at epoch [{epoch}]...')
                dist_train(train_loader, true_labels, model, optimizer, epoch, logger, cfg, pred_mem, rel_stats, warmup=False)
                # true_labels only used for monitoring accuracy

            # Separation
            logger.info(f'==> Start evaluation at epoch [{epoch}]...')
            t1 = time.time()
            eval_train(eval_train_loader, model, rel_stats)
            reliable_set_selection(cfg, rel_stats, given_labels, true_labels, logger)
            t2 = time.time()
            eval_time = format_time(t2 - t1)
            logger.info(f'==> Evaluation finished ({eval_time}).')

            # Alignment
            train(train_loader, model, train_criterion, optimizer, epoch, logger, cfg, rel_stats)
            # true_labels only used for monitoring accuracy

        if epoch % cfg.test_interval == 0 or epoch == cfg.epochs:
            if cfg.get('test_class_acc', False):
                test_acc, mean_ent, pred_max = \
                    test_class_acc(test_loader, model, test_criterion, epoch, logger, writer, cfg, is_src=False)
            else:
                test_acc, mean_ent = test(test_loader, model, test_criterion, epoch, logger, writer, is_src=False)
            test_meter.update(test_acc, idx=epoch)

    _, _, _ = test_class_acc(test_loader, model, test_criterion, epoch, logger, writer, cfg, is_src=False)
    
    # We print the best test_acc but use the last checkpoint for fine-tuning.
    logger.info(f'Best test_Acc@1: {test_meter.max_val:.2f} (epoch={test_meter.max_idx}).')

    # save last
    model_path = os.path.join(cfg.work_dir, f'last.pth')
    state_dict = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epochs': cfg.epochs
    }
    torch.save(state_dict, model_path)


def reliable_set_selection(cfg, rel_stats, src_pred_labels, true_labels, logger):
    loss, conf = rel_stats['loss'], rel_stats['conf']
    cls2tau, cls2tau_ini = rel_stats['cls2tau'], rel_stats['cls2tau_ini']
    tgt_pred_labels = torch.argmax(conf, dim=1)
    src_compare = (src_pred_labels == true_labels)
    tgt_compare = (tgt_pred_labels == true_labels)

    # select by loss
    is_rel_loss = torch.zeros(cfg.num_instances).bool().cuda()
    sorted_idx = torch.argsort(loss)
    chosen_num = int(cfg.num_instances * cfg.tau_loss)
    is_rel_loss[sorted_idx[:chosen_num]] = True
    src_compare_rel = src_compare[is_rel_loss]
    logger.info(f'Acc selected by loss: {int(src_compare_rel.sum())} / {len(src_compare_rel)} = {round(float(src_compare_rel.sum() / len(src_compare_rel)), 3)}')
    rel_stats['is_rel_src'] = is_rel_loss
    well_adapted_src = torch.zeros(cfg.num_classes).cuda()
    for j in range(cfg.num_classes):
        idx_src = torch.where(src_pred_labels[is_rel_loss] == j)[0]
        well_adapted_src[j] = len(idx_src)
    logger.info(f'Number of small loss samples pre-class:{well_adapted_src}')

    # select by conf
    pconf, pconf_label = torch.max(conf[~is_rel_loss].detach(), dim=1)
    idx_chosen = torch.where(pconf > cfg.tau_conf)[0]
    idx_chosen_flex = torch.where(pconf > cls2tau[pconf_label])[0]
    if len(idx_chosen) != 0:
        is_rel_conf = torch.zeros(cfg.num_instances).bool().cuda()
        idx_ur = torch.where(is_rel_loss == False)[0]
        is_rel_conf[idx_ur[idx_chosen]] = True
        tgt_compare_rel = tgt_compare[is_rel_conf]
        logger.info(f'Acc selected by high: {int(tgt_compare_rel.sum())} / {len(tgt_compare_rel)} = {round(float(tgt_compare_rel.sum() / len(tgt_compare_rel)), 3)}')
        rel_stats['is_rel_tgt'] = is_rel_conf

        is_rel_flex = torch.zeros(cfg.num_instances).bool().cuda()
        is_rel_flex[idx_ur[idx_chosen_flex]] = True
        flx_compare_rel = tgt_compare[is_rel_flex]
        logger.info(f'Acc selected by flex: {int(flx_compare_rel.sum())} / {len(flx_compare_rel)} = {round(float(flx_compare_rel.sum() / len(flx_compare_rel)), 3)}')
        rel_stats['is_rel_flx'] = is_rel_flex
    
    # update confidence threshold
    pconf_all, pconf_label_all = torch.max(conf.detach(), dim=1)
    idx_chosen_all = torch.where(pconf_all > cfg.tau_conf)[0]
    effects_tgt = torch.zeros(cfg.num_classes).cuda()
    effects_all = torch.zeros(cfg.num_classes).cuda()
    for j in range(cfg.num_classes):
        idx_cls = torch.where(pconf_label[idx_chosen] == j)[0]
        effects_tgt[j] = len(idx_cls)
        idx_cls_all = torch.where(pconf_label_all[idx_chosen_all] == j)[0]
        effects_all[j] = len(idx_cls_all)
    logger.info(f'Number of high-conf samples pre-class in tgt:{effects_tgt}')
    logger.info(f'Number of high-conf samples pre-class in all:{effects_all}')
    # scale and normalize the adaptation progress
    effects = effects_all ** cfg.tau_scale
    effects /= max(effects.clone())
    if cfg.upd_tau_trans == 1:  # linear
        cls2tau = effects * cls2tau_ini
    if cfg.upd_tau_trans == 2:  # convex
        cls2tau = (effects / (2 - effects)) * cls2tau_ini
    if cfg.upd_tau_trans == 3:  # concave
        cls2tau = (torch.log(effects + 1.) + 0.5)/(math.log(2) + 0.5) * cls2tau_ini
    if cfg.upd_tau_trans == 4:  # concave standard
        cls2tau = (torch.log(effects + 1.))/(math.log(2)) * cls2tau_ini
    cls2tau[cls2tau > cfg.tau_conf] = cfg.tau_conf
    logger.info(f'Updating class wise selection threshold:{cls2tau}')
    rel_stats['cls2tau'] = cls2tau


if __name__ == '__main__':
    main()
