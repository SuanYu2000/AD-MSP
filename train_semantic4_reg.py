"""
train_semantic4_reg: train_semantic4.py + train_FSL_reg.py regularizers.

Semantic4: 4 fixed imaging-axis templates + MultiPromptLearner + mp2 diversity losses.
Reg (optional, weight=0 to disable):
  --supcon_weight   Episode SupCon on visual features
  --text_dropout_p  Prob. of visual-only step (no text fusion)
  --clip_kd_weight  Relational CLIP visual KD on support prototypes

Train:
  python train_semantic4_reg.py --gpu 0 --dataset miniImageNet \\
    --init checkpoint/miniImageNet/visformer-t/pre-train-1shot/checkpoint_epoch_580.pth \\
    --supcon_weight 0.3 --comment supcon

Test (fusion sweep):
  python train_semantic4_reg.py --gpu 0 --dataset miniImageNet --test \\
    --resume checkpoint/.../checkpoint_epoch_best.pth --shot 1
"""

import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import clip

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from lib import visformer
from lib.multi_prompt_learner import MultiPromptLearner
from lib.utils import mean_confidence_interval
from data.dataloader import EpisodeSampler
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC

from lib.train_FSL_mp2 import (
    TextEncoder,
    build_adaptor,
    aggregate_over_prompts,
    encode_sample_text,
    mean_inter_prompt_similarity,
    compute_loss as mp_compute_loss,
    _episode_tensors,
)

NUM_PROMPTS = 4
SEMANTIC4_TEMPLATES = [
    "a photo of a",
    "a close-up photo of a",
    "a cropped photo of a",
    "a photo of a in natural light",
]

_STUDENT_MEAN = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]], dtype=np.float32)
_STUDENT_STD = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]], dtype=np.float32)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class PromptWeightModule(nn.Module):
    def __init__(self, n_prompts):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_prompts))


def student_to_clip_input(x):
    device, dtype = x.device, x.dtype
    mean_s = torch.as_tensor(_STUDENT_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_s = torch.as_tensor(_STUDENT_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    mean_c = torch.as_tensor(_CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_c = torch.as_tensor(_CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return (x * std_s + mean_s - mean_c) / std_c


def supcon_loss(features, labels, temperature):
    features = F.normalize(features, dim=-1)
    n = features.shape[0]
    if n <= 1:
        return features.new_tensor(0.)
    sim = features @ features.t() / temperature
    logits_mask = 1.0 - torch.eye(n, device=features.device, dtype=features.dtype)
    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.t()).float() * logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1).clamp(min=1e-8)
    return -((pos_mask * log_prob).sum(dim=1) / pos_count).mean()


def relational_kd(student_proto, clip_proto):
    return F.mse_loss(student_proto @ student_proto.t(), clip_proto @ clip_proto.t())


@torch.no_grad()
def clip_support_prototypes(teacher, sup_images, way, shot):
    feat = teacher.encode_image(student_to_clip_input(sup_images)).float()
    proto = feat.view(way, shot, -1).mean(dim=1)
    return F.normalize(proto, dim=-1)


def forward_episode(
    prompt_learner, text_encoder, student, sup, que, class_ids,
    way, shot, args, prompt_weight=None, class_offset=0, use_text=True,
):
    class_ids = class_ids + class_offset
    _, sup_im = student(sup)
    _, que_im = student(que)
    que_im = F.normalize(que_im, dim=-1)

    im_proto = sup_im.view(way, shot, -1).mean(dim=1)
    im_proto = F.normalize(im_proto, dim=-1)
    sim_im = que_im @ im_proto.t()

    n_prompts = prompt_learner.n_prompts
    if use_text:
        text_feat = encode_sample_text(prompt_learner, text_encoder, class_ids, args.eqnorm)
        text_adapt = student.adaptor(text_feat)
        sup_sem = sup_im.unsqueeze(1) + text_adapt
        proto_sem = sup_sem.view(way, shot, n_prompts, -1).mean(dim=1)
    else:
        text_feat = encode_sample_text(prompt_learner, text_encoder, class_ids, args.eqnorm)
        text_adapt = student.adaptor(text_feat)
        proto_sem = im_proto.unsqueeze(1).expand(way, n_prompts, -1)

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
        'im_proto': im_proto,
        'sup_im': sup_im,
        'que_im': que_im,
        'use_text': use_text,
    }
    for alpha in (0.2, 0.4, 0.6, 0.8, 1.0):
        out[f'sim_sum{int(alpha * 10)}'] = sim_text + alpha * sim_im
    out['sim_fuse'] = sim_text + fusion_alpha * sim_im
    return out


def compute_loss(out, labels, teacher, sup, args, way, shot):
    use_text = out['use_text']
    if use_text:
        loss, aux = mp_compute_loss(out, labels, args, way, shot)
    else:
        loss = F.cross_entropy(out['sim_im'] / args.t, labels)
        aux = {}

    aux.setdefault('supcon', 0.0)
    aux.setdefault('clip_kd', 0.0)
    aux['text_drop'] = 0.0 if use_text else 1.0

    if args.supcon_weight > 0:
        sup_lbl = torch.arange(way, device=labels.device).repeat_interleave(shot)
        feat = torch.cat([out['sup_im'], out['que_im']], dim=0)
        ep_labels = torch.cat([sup_lbl, labels], dim=0)
        sc = supcon_loss(feat, ep_labels, args.supcon_t)
        loss = loss + args.supcon_weight * sc
        aux['supcon'] = sc.item()

    if args.clip_kd_weight > 0:
        clip_proto = clip_support_prototypes(teacher, sup, way, shot)
        rk = relational_kd(out['im_proto'], clip_proto)
        loss = loss + args.clip_kd_weight * rk
        aux['clip_kd'] = rk.item()

    return loss, aux


def _print_config(args):
    axes = ['global baseline', 'scale / detail', 'composition / crop', 'lighting']
    print('[semantic4_reg] 4 shared templates:')
    for i, t in enumerate(SEMANTIC4_TEMPLATES):
        print(f'  P{i}: "{t}"  ({axes[i]})')
    print(
        f'[semantic4_reg] div_text={args.div_text_weight}, div_proto={args.div_proto_weight}, '
        f'pp_ce={args.pp_ce_weight}, text_agg={args.text_agg}'
    )
    active = []
    if args.supcon_weight > 0:
        active.append(f'supcon={args.supcon_weight}')
    if args.text_dropout_p > 0:
        active.append(f'text_drop={args.text_dropout_p}')
    if args.clip_kd_weight > 0:
        active.append(f'clip_kd={args.clip_kd_weight}')
    print('Regularizers:', ', '.join(active) if active else 'none (baseline)')


# During training validation: only main metrics.
# With --test: also report learnable fuse + text+alpha*im fusion sweep.
_TRAIN_TEST_METRICS = [
    ('acc_im', 'sim_im'),
    ('acc_text', 'sim_text'),
    ('acc_sum10', 'sim_sum10'),
]

_FULL_TEST_METRICS = [
    ('acc_im', 'sim_im'),
    ('acc_text', 'sim_text'),
    ('acc_fuse', 'sim_fuse'),   # learnable / configured fusion_alpha
    ('acc_sum2', 'sim_sum2'),   # alpha=0.2
    ('acc_sum4', 'sim_sum4'),   # alpha=0.4
    ('acc_sum6', 'sim_sum6'),   # alpha=0.6
    ('acc_sum8', 'sim_sum8'),   # alpha=0.8
    ('acc_sum10', 'sim_sum10'), # alpha=1.0
]


def train(prompt_learner, text_encoder, teacher, student, train_loader, optim, epoch, args, prompt_weight):
    student.train()
    prompt_learner.train()
    meters = {k: 0.0 for k in [
        'loss', 'acc_text', 'acc_sum10', 'text_sim', 'proto_sim', 'pp_ce',
        'supcon', 'clip_kd', 'text_drop',
    ]}

    for idx, episode in enumerate(train_loader):
        sup, que, class_ids, labels = _episode_tensors(episode, args, args.train_way)
        use_text = not (args.text_dropout_p > 0 and random.random() < args.text_dropout_p)
        out = forward_episode(
            prompt_learner, text_encoder, student, sup, que, class_ids,
            args.train_way, args.shot, args, prompt_weight=prompt_weight,
            class_offset=0, use_text=use_text,
        )
        loss, aux = compute_loss(out, labels, teacher, sup, args, args.train_way, args.shot)
        meters['loss'] += loss.item()

        text_sim, proto_sim = mean_inter_prompt_similarity(out['text_features'], out['proto_sem'])
        meters['text_sim'] += text_sim
        meters['proto_sim'] += proto_sim
        for k in ('pp_ce', 'supcon', 'clip_kd', 'text_drop'):
            meters[k] += aux.get(k, 0.0)

        for name, key in [('acc_text', 'sim_text'), ('acc_sum10', 'sim_sum10')]:
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
                f'supcon: {meters["supcon"]/n:.4f}, clip_kd: {meters["clip_kd"]/n:.4f}, '
                f'text_drop: {meters["text_drop"]/n:.2f}, '
                f'text_sim: {meters["text_sim"]/n:.3f}, proto_sim: {meters["proto_sim"]/n:.3f}'
            )

    n = len(train_loader)
    args.logger.add_scalar('train/loss', meters['loss'] / n, epoch)
    args.logger.add_scalar('train/acc_sum10', meters['acc_sum10'] / n, epoch)
    args.logger.add_scalar('train/text_sim', meters['text_sim'] / n, epoch)
    args.logger.add_scalar('train/proto_sim', meters['proto_sim'] / n, epoch)
    if args.supcon_weight > 0:
        args.logger.add_scalar('train/supcon', meters['supcon'] / n, epoch)
    if args.clip_kd_weight > 0:
        args.logger.add_scalar('train/clip_kd', meters['clip_kd'] / n, epoch)
    if args.text_dropout_p > 0:
        args.logger.add_scalar('train/text_drop_rate', meters['text_drop'] / n, epoch)


@torch.no_grad()
def test(prompt_learner, text_encoder, student, test_loader, epoch, args, prompt_weight):
    student.eval()
    prompt_learner.eval()
    metric_pairs = _FULL_TEST_METRICS if args.test else _TRAIN_TEST_METRICS
    buckets = {name: [] for name, _ in metric_pairs}
    text_sims, proto_sims = [], []

    for episode in test_loader:
        sup, que, class_ids, labels = _episode_tensors(episode, args, args.way)
        out = forward_episode(
            prompt_learner, text_encoder, student, sup, que, class_ids,
            args.way, args.shot, args, prompt_weight=prompt_weight,
            class_offset=args.delta, use_text=True,
        )
        ts, ps = mean_inter_prompt_similarity(out['text_features'], out['proto_sem'])
        text_sims.append(ts)
        proto_sims.append(ps)
        for name, key in metric_pairs:
            _, pred = out[key].max(-1)
            buckets[name].append(labels.eq(pred).sum().float().item() / labels.shape[0])

    for name, _ in metric_pairs:
        m, h = mean_confidence_interval(buckets[name])
        print(f'{name} Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    print(f'prompt_sim Test epoch: {epoch}, text: {np.mean(text_sims):.3f}, proto: {np.mean(proto_sims):.3f}')

    m, _ = mean_confidence_interval(buckets['acc_sum10'])
    args.logger.add_scalar('test/acc', m * 100, epoch)
    if args.test:
        args.logger.add_scalar('test/text_sim', np.mean(text_sims), epoch)
        args.logger.add_scalar('test/proto_sim', np.mean(proto_sims), epoch)
        for name, _ in metric_pairs:
            if name in ('acc_fuse',) or name.startswith('acc_sum'):
                mm, _ = mean_confidence_interval(buckets[name])
                args.logger.add_scalar(f'test/{name}', mm * 100, epoch)
    return m


def main(args):
    args.num_prompts = NUM_PROMPTS
    device = torch.device(f'cuda:{args.gpu}')

    args.tensorboard_dir = f'tensorboard/{args.dataset}/{args.model}/{args.exp}/'
    args.checkpoint_dir = f'checkpoint/{args.dataset}/{args.model}/{args.exp}/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)
    _print_config(args)

    norm = transforms.Normalize(_STUDENT_MEAN, _STUDENT_STD)
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
    args.train_way = args.way if args.train_way == -1 else args.train_way
    n_episodes = args.train_episodes
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

    teacher, _ = clip.load('ViT-B/32', device=device)
    teacher.float()
    teacher.requires_grad_(False)
    teacher.eval()

    train_names = [train_dataset.idx2text[i] for i in train_dataset.dataset.classes]
    test_names = [test_dataset.idx2text[i] for i in test_dataset.dataset.classes]
    all_classnames = train_names + test_names

    text_encoder = TextEncoder(teacher).to(device)
    prompt_learner = MultiPromptLearner(
        all_classnames, teacher,
        num_prompts=NUM_PROMPTS,
        dataset_name=args.dataset,
        n_ctx=args.n_ctx,
        template_list=SEMANTIC4_TEMPLATES,
    ).to(device)

    student = visformer.visformer_tiny(num_classes=num_classes, drop_rate=args.dropout)
    feature_dim = 192 if 2 <= args.stage < 3 else 384
    build_adaptor(student, 512, feature_dim, args)

    prompt_weight_mod = None
    if args.text_agg == 'learnable':
        prompt_weight_mod = PromptWeightModule(NUM_PROMPTS).to(device)

    if args.learnable_fusion_alpha:
        student.fusion_alpha = nn.Parameter(torch.zeros(1, device=device))

    student = student.to(device)
    prompt_weight = prompt_weight_mod.weight if prompt_weight_mod is not None else None

    optim_head = list(student.adaptor.parameters())
    if args.learnable_fusion_alpha:
        optim_head.append(student.fusion_alpha)
    optim_head += list(prompt_learner.parameters())
    if prompt_weight_mod is not None:
        optim_head += list(prompt_weight_mod.parameters())
    head_ids = {id(p) for p in optim_head}
    encoder_params = [p for p in student.parameters() if id(p) not in head_ids]

    optim = torch.optim.AdamW([
        {'params': optim_head, 'lr': args.lr},
        {'params': encoder_params, 'lr': args.encoder_lr},
    ], weight_decay=args.weight_decay)

    if args.resume:
        args.init = args.resume
    if not args.init:
        raise ValueError('must provide pre-trained model')

    ckpt = torch.load(args.init, map_location=device)
    student.load_state_dict(ckpt['state_dict'], strict=False)
    print(f'Loaded init from {args.init}')

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt['state_dict'], strict=False)
        if 'prompt_learner' in ckpt:
            prompt_learner.load_state_dict(ckpt['prompt_learner'])
        if 'prompt_learner_n_ctx_list' in ckpt:
            prompt_learner.n_ctx_list = list(ckpt['prompt_learner_n_ctx_list'])
            prompt_learner.n_ctx = max(prompt_learner.n_ctx_list)
        if 'prompt_weight' in ckpt and prompt_weight_mod is not None:
            prompt_weight_mod.weight.data.copy_(ckpt['prompt_weight'])
        if 'optimizer' in ckpt:
            optim.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    if args.test:
        print(
            f'[test mode] way={args.way}, shot={args.shot}, episodes={args.test_episodes}, '
            f'fusion sweep: acc_fuse + acc_sum2/4/6/8/10'
        )
        test(prompt_learner, text_encoder, student, test_loader, 0, args, prompt_weight)
        return

    best_acc, best_epoch = 0.0, 0
    for epoch in range(start_epoch, args.epochs):
        train(prompt_learner, text_encoder, teacher, student, train_loader, optim, epoch, args, prompt_weight)
        acc = 0.0
        if (epoch + 1) % args.test_freq == 0:
            acc = test(prompt_learner, text_encoder, student, test_loader, epoch, args, prompt_weight)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
            'prompt_learner': prompt_learner.state_dict(),
            'prompt_learner_n_ctx_list': prompt_learner.n_ctx_list,
        }
        if prompt_weight_mod is not None:
            checkpoint['prompt_weight'] = prompt_weight_mod.weight.data
        torch.save(checkpoint, args.checkpoint_dir + 'checkpoint_epoch_latest.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_{epoch + 1:03d}.pth')
        if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(checkpoint, args.checkpoint_dir + 'checkpoint_epoch_best.pth')
        print(f'best_epoch: {best_epoch}, best_acc: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='semantic4 multi-prompt FSL + reg regularizers')
    parser.add_argument('--exp', type=str, default='semantic4_reg')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--image_size', type=int, default=224)
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
                        default='checkpoint/miniImageNet/visformer-t/pre-train-1shot/checkpoint_epoch_580.pth')
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
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--KD', type=float, default=1)
    parser.add_argument('--adaptor', type=str, default='mlp', choices=['linear', 'mlp', 'bottle'])
    parser.add_argument('--n_ctx', type=int, default=-1)
    parser.add_argument('--text_agg', type=str, default='learnable',
                        choices=['mean', 'max', 'ot', 'learnable'])
    parser.add_argument('--ot_eps', type=float, default=0.1)
    # mp2 diversity
    parser.add_argument('--div_text_weight', type=float, default=0.05)
    parser.add_argument('--div_proto_weight', type=float, default=0.05)
    parser.add_argument('--pp_ce_weight', type=float, default=0.3)
    parser.add_argument('--div_margin', type=float, default=0.2)
    parser.add_argument('--div_adapt_weight', type=float, default=0.0)
    parser.add_argument('--div_weight', type=float, default=0.0)
    parser.add_argument('--fuse_weight', type=float, default=0.0)
    parser.add_argument('--fusion_alpha', type=float, default=1.0)
    parser.add_argument('--learnable_fusion_alpha', action='store_true')
    parser.add_argument('--fusion_alpha_max', type=float, default=1.0)
    # train_FSL_reg regularizers
    parser.add_argument('--supcon_weight', type=float, default=0.0)
    parser.add_argument('--supcon_t', type=float, default=0.07)
    parser.add_argument('--text_dropout_p', type=float, default=0.0)
    parser.add_argument('--clip_kd_weight', type=float, default=0.0)

    args = parser.parse_args()
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
