"""
Multi-prompt FSL with explicit cross-template diversity (mp2).

Based on train_FSL_mp.py, adds:
  1. Margin-based text diversity: push inter-prompt cosine sim below margin
  2. Prototype diversity: different prompts -> different semantic prototypes (same class)
  3. Per-prompt CE: each template branch must classify (prevents useless orthogonality)
  4. Logging of mean inter-prompt similarity for monitoring

Loss = CE(sim_im) + CE(sim_text) + KD
     + pp_ce_weight * mean_i CE(sim_stack[..., i])
     + div_text_weight * text_diversity_loss
     + div_proto_weight * prototype_diversity_loss
     + optional fuse / legacy div_weight terms
"""

import os
import argparse
import math
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import clip
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from . import visformer
from .multi_prompt_learner import MultiPromptLearner
from .head_guidance import prompt_diversity_loss
from .utils import mean_confidence_interval
from data.dataloader import EpisodeSampler
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC


def sinkhorn_transport(sim, eps=0.1, max_iter=100, thresh=1e-2):
    wdist = 1.0 - sim
    m, n = sim.shape[-2], sim.shape[-1]
    lead = sim.shape[:-2]
    flat = math.prod(lead) if lead else 1
    wdist_f = wdist.reshape(flat, m, n)
    sim_f = sim.reshape(flat, m, n)
    xx = wdist_f.new_full((flat, m), 1.0 / m)
    yy = wdist_f.new_full((flat, n), 1.0 / n)
    with torch.no_grad():
        kk = torch.exp(-wdist_f / eps)
        r = torch.ones_like(xx)
        c = torch.ones_like(yy)
        for _ in range(max_iter):
            r0 = r
            r = xx / torch.matmul(kk, c.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
            c = yy / torch.matmul(kk.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
            if (r - r0).abs().mean() < thresh:
                break
        t = r.unsqueeze(-1) * c.unsqueeze(-2) * kk
    return (t * sim_f).sum(dim=(-2, -1)).reshape(lead)


def aggregate_over_prompts(sim, mode='mean', ot_eps=0.1, prompt_weight=None):
    if mode == 'mean':
        return sim.mean(dim=-1)
    if mode == 'max':
        return sim.max(dim=-1).values
    if mode == 'learnable':
        w = F.softmax(prompt_weight, dim=0).view(1, 1, -1)
        return (sim * w).sum(dim=-1)
    if mode == 'ot':
        q, way, n = sim.shape
        return sinkhorn_transport(sim.reshape(q * way, 1, n), eps=ot_eps).reshape(q, way)
    raise ValueError(f'unknown text_agg mode: {mode}')


def _off_diagonal_cosine_sim(feat):
    """feat: (..., N, D) -> mean off-diagonal cosine similarity."""
    t = F.normalize(feat, dim=-1)
    sim = torch.matmul(t, t.transpose(-1, -2))
    n = sim.shape[-1]
    if n <= 1:
        return sim.new_tensor(0.), sim.new_tensor(0.)
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    off = sim[..., ~eye].view(*sim.shape[:-2], n, n - 1)
    return off.mean(), off


def text_diversity_loss(text_features, margin=0.0):
    """
    Penalize high cosine similarity between prompt text embeddings (same sample).
    margin>0: only penalize sim above margin (hinge), avoids over-pushing already-diverse pairs.
    """
    _, off = _off_diagonal_cosine_sim(text_features)
    if margin > 0:
        return F.relu(off - margin).pow(2).mean()
    return (off ** 2).mean()


def prototype_diversity_loss(proto_sem):
    """
    proto_sem: (way, N, D)
    For each class, push N semantic prototypes (image+text) to be dissimilar.
    Stronger signal than text-only diversity: captures fused semantic directions.
    """
    _, off = _off_diagonal_cosine_sim(proto_sem)
    return (off ** 2).mean()


def adapted_text_diversity_loss(text_adapt, way, shot):
    """
    Decorrelate adaptor outputs across prompts, averaged per class.
    text_adapt: (way*shot, N, D)
    """
    _, n, _ = text_adapt.shape
    if n <= 1:
        return text_adapt.new_tensor(0.)
    per_class = text_adapt.view(way, shot, n, -1).mean(dim=1)
    return text_diversity_loss(per_class, margin=0.0)


def per_prompt_ce_loss(sim_stack, labels, temperature):
    """Each prompt logit slice must classify correctly."""
    _, _, n = sim_stack.shape
    if n <= 1:
        return F.cross_entropy(sim_stack.squeeze(-1) / temperature, labels)
    losses = [
        F.cross_entropy(sim_stack[:, :, i] / temperature, labels)
        for i in range(n)
    ]
    return torch.stack(losses).mean()


def mean_inter_prompt_similarity(text_features, proto_sem=None):
    """Monitor-only metrics in [0, 1]."""
    with torch.no_grad():
        _, text_off = _off_diagonal_cosine_sim(text_features)
        text_sim = text_off.mean().item()
        if proto_sem is None:
            return text_sim, 0.0
        _, proto_off = _off_diagonal_cosine_sim(proto_sem)
        return text_sim, proto_off.mean().item()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class adaptor(nn.Module):
    def __init__(self, text_dim, feature_dim):
        super().__init__()
        self.layer1 = nn.Linear(text_dim, feature_dim)
        self.layer2 = nn.Linear(feature_dim, feature_dim // 4)
        self.layer3 = nn.Linear(feature_dim // 4, feature_dim)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x1 = F.leaky_relu(self.layer2(x))
        return self.layer3(x1) + x


def normalize_text_features(text_features, eqnorm=True):
    if not eqnorm:
        return text_features
    avg_length = (text_features ** 2).sum(-1).sqrt().mean().item()
    return F.normalize(text_features, dim=-1) * avg_length


def encode_sample_text(prompt_learner, text_encoder, class_ids, eqnorm):
    prompts, tokenized = prompt_learner()
    flat = prompt_learner.class_prompt_indices(class_ids).reshape(-1)
    text = text_encoder(prompts[flat], tokenized[flat]).float()
    b, n = class_ids.shape[0], prompt_learner.n_prompts
    text = text.view(b, n, -1)
    return normalize_text_features(text, eqnorm)


def forward_episode(
    prompt_learner, text_encoder, student, sup, que, class_ids,
    way, shot, args, prompt_weight=None, class_offset=0,
):
    class_ids = class_ids + class_offset

    _, sup_im = student(sup)
    _, que_im = student(que)
    que_im = F.normalize(que_im, dim=-1)

    im_proto = sup_im.view(way, shot, -1).mean(dim=1)
    im_proto = F.normalize(im_proto, dim=-1)
    sim_im = que_im @ im_proto.t()

    text_feat = encode_sample_text(prompt_learner, text_encoder, class_ids, args.eqnorm)
    text_adapt = student.adaptor(text_feat)
    sup_sem = sup_im.unsqueeze(1) + text_adapt
    proto_sem = sup_sem.view(way, shot, prompt_learner.n_prompts, -1).mean(dim=1)
    proto_sem = F.normalize(proto_sem, dim=-1)

    sim_stack = torch.einsum('qd,wnd->qwn', que_im, proto_sem)
    sim_text = aggregate_over_prompts(
        sim_stack, mode=args.text_agg, ot_eps=args.ot_eps, prompt_weight=prompt_weight,
    )

    fusion_alpha = args.fusion_alpha
    if args.learnable_fusion_alpha and hasattr(student, 'fusion_alpha'):
        fusion_alpha = student.fusion_alpha.sigmoid() * args.fusion_alpha_max

    out = {
        'sim_im': sim_im,
        'sim_text': sim_text,
        'sim_stack': sim_stack,
        'text_features': text_feat,
        'text_adapt': text_adapt,
        'proto_sem': proto_sem,
    }
    for alpha in (0.2, 0.4, 0.6, 0.8, 1.0):
        out[f'sim_sum{int(alpha * 10)}'] = sim_text + alpha * sim_im
    out['sim_fuse'] = sim_text + fusion_alpha * sim_im
    return out


def compute_loss(out, labels, args, way, shot):
    sim_im, sim_text = out['sim_im'], out['sim_text']
    kd_loss = JS_div(sim_im / args.t, sim_text / args.t)
    loss = (
        F.cross_entropy(sim_im / args.t, labels)
        + F.cross_entropy(sim_text / args.t, labels)
        + args.KD * kd_loss
    )

    aux = {}
    if args.pp_ce_weight > 0:
        pp_ce = per_prompt_ce_loss(out['sim_stack'], labels, args.t)
        loss = loss + args.pp_ce_weight * pp_ce
        aux['pp_ce'] = pp_ce.item()

    if args.div_text_weight > 0:
        div_text = text_diversity_loss(out['text_features'], margin=args.div_margin)
        loss = loss + args.div_text_weight * div_text
        aux['div_text'] = div_text.item()

    if args.div_proto_weight > 0:
        div_proto = prototype_diversity_loss(out['proto_sem'])
        loss = loss + args.div_proto_weight * div_proto
        aux['div_proto'] = div_proto.item()

    if args.div_adapt_weight > 0:
        div_adapt = adapted_text_diversity_loss(out['text_adapt'], way, shot)
        loss = loss + args.div_adapt_weight * div_adapt
        aux['div_adapt'] = div_adapt.item()

    if args.div_weight > 0:
        div_legacy = prompt_diversity_loss(out['text_features'])
        loss = loss + args.div_weight * div_legacy
        aux['div_legacy'] = div_legacy.item()

    if args.fuse_weight > 0:
        loss = loss + args.fuse_weight * F.cross_entropy(out['sim_fuse'] / args.t, labels)

    return loss, aux


def JS_div(p_output, q_output):
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    p_output = F.softmax(p_output, dim=1)
    q_output = F.softmax(q_output, dim=1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


def build_adaptor(student, text_dim, feature_dim, args):
    if args.adaptor == 'linear':
        student.adaptor = nn.Linear(text_dim, feature_dim, bias=False)
    elif args.adaptor == 'mlp':
        student.adaptor = nn.Sequential(
            nn.Linear(text_dim, feature_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
        )
    else:
        student.adaptor = adaptor(text_dim, feature_dim)


def main(args):
    args.tensorboard_dir = f'tensorboard/{args.dataset}/{args.model}/{args.exp}/'
    args.checkpoint_dir = f'checkpoint/{args.dataset}/{args.model}/{args.exp}/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

    norm = transforms.Normalize(
        np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
    )
    train_aug = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm,
    ])
    if args.aug:
        train_aug = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm,
        ])
    if args.rand_aug:
        train_aug = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            RandAugmentMC(2, 10, args.image_size),
            transforms.ToTensor(), norm,
        ])
    test_aug = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.1)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), norm,
    ])

    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    n_episodes = args.train_episodes
    args.train_way = args.way if args.train_way == -1 else args.train_way
    if n_episodes == -1:
        n_episodes = int(len(train_dataset) / (args.train_way * (args.shot + 15)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=EpisodeSampler(
            train_dataset.dataset.targets, n_episodes, args.train_way, args.shot + 15, fix_seed=False,
        ),
        num_workers=8,
    )
    num_classes = len(train_dataset.dataset.classes)

    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=EpisodeSampler(
            test_dataset.dataset.targets, args.test_episodes, args.way, args.shot + 15,
        ),
        num_workers=6,
    )

    teacher, _ = clip.load('ViT-B/32', device=f'cuda:{args.gpu}')
    teacher.float()
    teacher.requires_grad_(False)
    teacher.eval()
    text_dim = 512

    train_classnames = [train_dataset.idx2text[idx] for idx in train_dataset.dataset.classes]
    test_classnames = [test_dataset.idx2text[idx] for idx in test_dataset.dataset.classes]
    all_classnames = train_classnames + test_classnames

    text_encoder = TextEncoder(teacher).cuda()
    prompt_learner = MultiPromptLearner(
        all_classnames, teacher,
        num_prompts=args.num_prompts,
        ctx_init=args.ctx_init,
        dataset_name=args.dataset,
        n_ctx=args.n_ctx,
    ).cuda()

    student = visformer.visformer_tiny(num_classes=num_classes, drop_rate=args.dropout)
    feature_dim = 192 if 2 <= args.stage < 3 else 384
    build_adaptor(student, text_dim, feature_dim, args)

    prompt_weight = None
    if args.text_agg == 'learnable':
        prompt_weight = nn.Parameter(torch.zeros(args.num_prompts))

    if args.learnable_fusion_alpha:
        student.fusion_alpha = nn.Parameter(torch.tensor(0.0))

    student = student.cuda(args.gpu)
    if prompt_weight is not None:
        prompt_weight = prompt_weight.cuda()

    optim_params_id = {id(p) for p in student.adaptor.parameters()}
    if args.learnable_fusion_alpha:
        optim_params_id.add(id(student.fusion_alpha))
    optim_head = [p for p in student.parameters() if id(p) in optim_params_id]
    optim_head += list(prompt_learner.parameters())
    if prompt_weight is not None:
        optim_head.append(prompt_weight)
    encoder_params = [p for p in student.parameters() if id(p) not in optim_params_id]

    optim = torch.optim.AdamW([
        {'params': optim_head, 'lr': args.lr},
        {'params': encoder_params, 'lr': args.encoder_lr},
    ], weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        args.init = args.resume
    if not args.init:
        raise ValueError('must provide pre-trained model')

    ckpt = torch.load(args.init, map_location=f'cuda:{args.gpu}')
    student.load_state_dict(ckpt['state_dict'], strict=False)
    print(f'Loaded init from {args.init}')

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        student.load_state_dict(ckpt['state_dict'], strict=False)
        if 'prompt_learner' in ckpt:
            prompt_learner.load_state_dict(ckpt['prompt_learner'])
        if 'prompt_weight' in ckpt and prompt_weight is not None:
            prompt_weight.data.copy_(ckpt['prompt_weight'])
        if 'optimizer' in ckpt:
            optim.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    print(
        f'[mp2 diversity] div_text={args.div_text_weight}, div_proto={args.div_proto_weight}, '
        f'div_adapt={args.div_adapt_weight}, margin={args.div_margin}, pp_ce={args.pp_ce_weight}'
    )

    if args.test:
        test(prompt_learner, text_encoder, student, test_loader, 0, args, prompt_weight)
        return

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=50)
    best_acc, best_epoch = 0.0, 0

    for epoch in range(start_epoch, args.epochs):
        train(prompt_learner, text_encoder, student, train_loader, optim, epoch, args, prompt_weight)
        acc = 0.0
        if (epoch + 1) % args.test_freq == 0:
            acc = test(prompt_learner, text_encoder, student, test_loader, epoch, args, prompt_weight)
        if args.sheduler == 'True' and (epoch + 1) % args.test_freq == 0:
            scheduler.step(acc)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
            'prompt_learner': prompt_learner.state_dict(),
        }
        if prompt_weight is not None:
            checkpoint['prompt_weight'] = prompt_weight.data
        torch.save(checkpoint, args.checkpoint_dir + 'checkpoint_epoch_latest.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_{epoch + 1:03d}.pth')
        if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(checkpoint, args.checkpoint_dir + 'checkpoint_epoch_best.pth')
        print(f'best_epoch: {best_epoch}, best_acc: {best_acc:.4f}')


def _episode_tensors(episode, args, way):
    image = episode[0].cuda(args.gpu)
    glabels = episode[1].cuda(args.gpu)
    labels = torch.arange(way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)
    image = image.view(way, args.shot + 15, *image.shape[1:])
    sup = image[:, :args.shot].contiguous().view(-1, *image.shape[2:])
    que = image[:, args.shot:].contiguous().view(-1, *image.shape[2:])
    class_ids = glabels.view(way, args.shot + 15)[:, :args.shot].contiguous().view(-1)
    return sup, que, class_ids, labels


def train(prompt_learner, text_encoder, student, train_loader, optim, epoch, args, prompt_weight):
    student.train()
    prompt_learner.train()
    meters = {k: 0.0 for k in [
        'loss', 'acc_im', 'acc_text', 'acc_fuse',
        'acc_sum2', 'acc_sum4', 'acc_sum6', 'acc_sum8', 'acc_sum10',
        'text_sim', 'proto_sim', 'pp_ce', 'div_text', 'div_proto',
    ]}

    for idx, episode in enumerate(train_loader):
        sup, que, class_ids, labels = _episode_tensors(episode, args, args.train_way)
        out = forward_episode(
            prompt_learner, text_encoder, student, sup, que, class_ids,
            args.train_way, args.shot, args, prompt_weight=prompt_weight, class_offset=0,
        )
        loss, aux = compute_loss(out, labels, args, args.train_way, args.shot)
        meters['loss'] += loss.item()

        text_sim, proto_sim = mean_inter_prompt_similarity(out['text_features'], out['proto_sem'])
        meters['text_sim'] += text_sim
        meters['proto_sim'] += proto_sim
        for k in ('pp_ce', 'div_text', 'div_proto'):
            if k in aux:
                meters[k] += aux[k]

        for name, key in [
            ('acc_im', 'sim_im'), ('acc_text', 'sim_text'), ('acc_fuse', 'sim_fuse'),
            ('acc_sum2', 'sim_sum2'), ('acc_sum4', 'sim_sum4'), ('acc_sum6', 'sim_sum6'),
            ('acc_sum8', 'sim_sum8'), ('acc_sum10', 'sim_sum10'),
        ]:
            _, pred = out[key].max(-1)
            meters[name] += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            n = idx + 1
            print(
                f'Train epoch: {epoch}, step: {idx:3d}, loss: {meters["loss"]/n:.4f}, '
                f'acc_text: {meters["acc_text"]/n*100:.2f}, acc_sum10: {meters["acc_sum10"]/n*100:.2f}, '
                f'text_sim: {meters["text_sim"]/n:.3f}, proto_sim: {meters["proto_sim"]/n:.3f}, '
                f'pp_ce: {meters["pp_ce"]/max(n if meters["pp_ce"] else 1, 1):.4f}, '
                f'N: {args.num_prompts}'
            )

    n = len(train_loader)
    args.logger.add_scalar('train/loss', meters['loss'] / n, epoch)
    args.logger.add_scalar('train/acc_text', meters['acc_text'] / n, epoch)
    args.logger.add_scalar('train/acc_sum10', meters['acc_sum10'] / n, epoch)
    args.logger.add_scalar('train/text_sim', meters['text_sim'] / n, epoch)
    args.logger.add_scalar('train/proto_sim', meters['proto_sim'] / n, epoch)
    if meters['div_text'] > 0:
        args.logger.add_scalar('train/div_text', meters['div_text'] / n, epoch)
    if meters['div_proto'] > 0:
        args.logger.add_scalar('train/div_proto', meters['div_proto'] / n, epoch)


@torch.no_grad()
def test(prompt_learner, text_encoder, student, test_loader, epoch, args, prompt_weight):
    student.eval()
    prompt_learner.eval()
    buckets = {k: [] for k in [
        'acc_im', 'acc_text', 'acc_fuse',
        'acc_sum2', 'acc_sum4', 'acc_sum6', 'acc_sum8', 'acc_sum10',
    ]}
    text_sims, proto_sims = [], []

    for episode in test_loader:
        sup, que, class_ids, labels = _episode_tensors(episode, args, args.way)
        out = forward_episode(
            prompt_learner, text_encoder, student, sup, que, class_ids,
            args.way, args.shot, args, prompt_weight=prompt_weight, class_offset=args.delta,
        )
        text_sim, proto_sim = mean_inter_prompt_similarity(out['text_features'], out['proto_sem'])
        text_sims.append(text_sim)
        proto_sims.append(proto_sim)

        for name, key in [
            ('acc_im', 'sim_im'), ('acc_text', 'sim_text'), ('acc_fuse', 'sim_fuse'),
            ('acc_sum2', 'sim_sum2'), ('acc_sum4', 'sim_sum4'), ('acc_sum6', 'sim_sum6'),
            ('acc_sum8', 'sim_sum8'), ('acc_sum10', 'sim_sum10'),
        ]:
            _, pred = out[key].max(-1)
            buckets[name].append(labels.eq(pred).sum().float().item() / labels.shape[0])

    for name in buckets:
        m, h = mean_confidence_interval(buckets[name])
        print(f'{name} Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')

    print(
        f'prompt_sim Test epoch: {epoch}, '
        f'text: {np.mean(text_sims):.3f}, proto: {np.mean(proto_sims):.3f}'
    )

    m, _ = mean_confidence_interval(buckets['acc_sum10'])
    args.logger.add_scalar('test/acc', m * 100, epoch)
    args.logger.add_scalar('test/text_sim', np.mean(text_sims), epoch)
    args.logger.add_scalar('test/proto_sim', np.mean(proto_sims), epoch)
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='fsl_mp2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true', default=True)
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--model', type=str, default='visformer-t')
    parser.add_argument('--eqnorm', action='store_true', default=True)
    parser.add_argument('--stage', type=float, default=3.2)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--encoder_lr', type=float, default=1e-6)
    parser.add_argument('--init', type=str,
                        default='checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--train_way', type=int, default=-1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--test_episodes', type=int, default=2000)
    parser.add_argument('--print_step', type=int, default=300)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--comment', type=str, default=' ')
    parser.add_argument('--sheduler', type=str, default='False')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--KD', type=float, default=1)
    parser.add_argument('--adaptor', type=str, default='mlp', choices=['linear', 'mlp', 'bottle'])
    # multi-prompt (same as mp)
    parser.add_argument('--num_prompts', type=int, default=4)
    parser.add_argument('--n_ctx', type=int, default=-1)
    parser.add_argument('--ctx_init', type=str, default='templates', choices=['single', 'templates'])
    parser.add_argument('--text_agg', type=str, default='mean',
                        choices=['mean', 'max', 'ot', 'learnable'])
    parser.add_argument('--ot_eps', type=float, default=0.1)
    parser.add_argument('--fuse_weight', type=float, default=0.0)
    parser.add_argument('--fusion_alpha', type=float, default=1.0)
    parser.add_argument('--learnable_fusion_alpha', action='store_true')
    parser.add_argument('--fusion_alpha_max', type=float, default=1.0)
    # mp2: cross-template diversity
    parser.add_argument('--div_weight', type=float, default=0.0,
                        help='legacy text diversity from head_guidance; usually use div_text_weight instead')
    parser.add_argument('--div_text_weight', type=float, default=0.05,
                        help='margin/hinge diversity on CLIP text embeddings across prompts')
    parser.add_argument('--div_proto_weight', type=float, default=0.05,
                        help='diversity on fused semantic prototypes (way, N, D)')
    parser.add_argument('--div_adapt_weight', type=float, default=0.0,
                        help='optional diversity on adaptor outputs per class')
    parser.add_argument('--div_margin', type=float, default=0.2,
                        help='hinge margin for text diversity; only penalize cos_sim > margin')
    parser.add_argument('--pp_ce_weight', type=float, default=0.3,
                        help='per-prompt CE; keeps each template useful for classification')

    args = parser.parse_args()
    from datetime import datetime
    args.exp = args.exp + args.dataset + str(datetime.now())[:18] + args.comment

    if args.dataset == 'FC100':
        args.delta = 60
    elif args.dataset == 'tieredImageNet':
        args.delta = 351
    else:
        args.delta = 64

    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)
