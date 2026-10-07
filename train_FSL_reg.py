"""
train_FSL_reg: train_FSL.py + optional regularizers for novel-class generalization.

Each component is gated by its weight / probability (0 = off), so you can ablate one at a time
without maintaining multiple scripts:

  --supcon_weight   Episode supervised contrastive on visual features (SupCon).
  --text_dropout_p  Probability of training step without text fusion (visual-only branch).
  --clip_kd_weight  Relational KD: align support-prototype similarity structure with CLIP visual.

Suggested ablation order (same seed, same init):
  1) baseline:  all weights 0, text_dropout_p 0
  2) +SupCon:   --supcon_weight 0.1
  3) +dropout:  --text_dropout_p 0.3
  4) +CLIP-KD:  --clip_kd_weight 0.5
  5) combine only the components that helped in 2–4

Test (fusion sweep, only with --test):
  python train_FSL_reg.py --gpu 0 --dataset miniImageNet --test \\
    --resume checkpoint/.../checkpoint_epoch_best.pth --shot 1
  Reports acc for sim_text + alpha * sim_im, alpha in {0.2, 0.4, 0.6, 0.8, 1.0}.
"""

import os
import argparse
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from lib import visformer
from lib.utils import mean_confidence_interval
from data.dataloader import EpisodeSampler
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC

print(time.strftime('%S', time.localtime()))

# visformer normalization (train_FSL.py)
_STUDENT_MEAN = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]], dtype=np.float32)
_STUDENT_STD = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]], dtype=np.float32)
# CLIP ViT-B/32
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def student_to_clip_input(x):
    """Denormalize visformer stats, re-normalize with CLIP stats."""
    device, dtype = x.device, x.dtype
    mean_s = torch.as_tensor(_STUDENT_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_s = torch.as_tensor(_STUDENT_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    mean_c = torch.as_tensor(_CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_c = torch.as_tensor(_CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    raw = x * std_s + mean_s
    return (raw - mean_c) / std_c


def supcon_loss(features, labels, temperature):
    """Supervised contrastive loss within one episode (Khosla et al.)."""
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
    """Match pairwise cosine structure; no extra projection layer needed."""
    s = student_proto @ student_proto.t()
    c = clip_proto @ clip_proto.t()
    return F.mse_loss(s, c)


@torch.no_grad()
def clip_support_prototypes(teacher, sup_images, way, shot):
    clip_in = student_to_clip_input(sup_images)
    feat = teacher.encode_image(clip_in).float()
    proto = feat.view(way, shot, -1).mean(dim=1)
    return F.normalize(proto, dim=-1)


def JS_div(p_output, q_output):
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    p_output = F.softmax(p_output, dim=1)
    q_output = F.softmax(q_output, dim=1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


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


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        ctx_init = 'a photo of a'
        dtype = clip_model.dtype
        ctx_init = ctx_init.replace('_', ' ')
        n_ctx = len(ctx_init.split(' '))
        prompt = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        prompt_prefix = ctx_init
        print(f'Initial context: "{prompt_prefix}"')
        print(f'Number of context words (tokens): {n_ctx}')

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace('_', ' ') for name in classnames]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


def forward_episode(
    prompts, tokenized_prompts, text_encoder, student, teacher,
    sup, que, glabels, way, shot, args, use_text=True,
):
    """Shared train/test forward; returns logits and aux tensors for loss."""
    text_features = text_encoder(prompts[glabels], tokenized_prompts[glabels])
    avg_length = (text_features ** 2).sum(-1).sqrt().mean().item()
    text_features = F.normalize(text_features, dim=-1) * avg_length

    _, sup_im = student(sup)
    _, que_im = student(que)
    que_im = F.normalize(que_im, dim=-1)

    im_proto = sup_im.view(way, shot, -1).mean(dim=1)
    im_proto = F.normalize(im_proto, dim=-1)
    sim_im = que_im @ im_proto.t()

    if use_text:
        text_adapt = student.adaptor(text_features)
        sup_sem = sup_im + text_adapt
    else:
        sup_sem = sup_im

    proto_sem = sup_sem.view(way, shot, -1).mean(dim=1)
    proto_sem = F.normalize(proto_sem, dim=-1)
    sim_text = que_im @ proto_sem.t()

    out = {
        'sim_im': sim_im,
        'sim_text': sim_text,
        'que_im': que_im,
        'sup_im': sup_im,
        'im_proto': im_proto,
        'use_text': use_text,
    }
    for alpha in (0.2, 0.4, 0.6, 0.8, 1.0):
        out[f'sim_sum{int(alpha * 10)}'] = sim_text + alpha * sim_im
    return out


def compute_loss(out, labels, teacher, sup, way, shot, args):
    sim_im, sim_text = out['sim_im'], out['sim_text']
    use_text = out['use_text']

    loss = F.cross_entropy(sim_im / args.t, labels)
    aux = {'supcon': 0.0, 'clip_kd': 0.0, 'text_drop': 0.0 if use_text else 1.0}

    if use_text:
        kd_loss = JS_div(sim_im / args.t, sim_text / args.t)
        loss = loss + F.cross_entropy(sim_text / args.t, labels) + args.KD * kd_loss

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


def train(prompt_learner, prompts, tokenized_prompts, text_encoder, teacher, student,
          train_loader, optim, epoch, args):
    student.train()
    prompt_learner.train()

    meters = {k: 0.0 for k in [
        'loss', 'supcon', 'clip_kd', 'text_drop', 'acc_im', 'acc_text', 'acc_sum2',
    ]}

    for idx, episode in enumerate(train_loader):
        image = episode[0].cuda(args.gpu)
        glabels = episode[1].cuda(args.gpu)
        labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

        image = image.view(args.train_way, args.shot + 15, *image.shape[1:])
        sup = image[:, :args.shot].contiguous().view(-1, *image.shape[2:])
        que = image[:, args.shot:].contiguous().view(-1, *image.shape[2:])

        glabels = glabels.view(args.train_way, args.shot + 15)[:, :args.shot].contiguous().view(-1)

        use_text = not (args.text_dropout_p > 0 and random.random() < args.text_dropout_p)
        out = forward_episode(
            prompts, tokenized_prompts, text_encoder, student, teacher,
            sup, que, glabels, args.train_way, args.shot, args, use_text=use_text,
        )
        loss, aux = compute_loss(out, labels, teacher, sup, args.train_way, args.shot, args)

        meters['loss'] += loss.item()
        for k in ('supcon', 'clip_kd', 'text_drop'):
            meters[k] += aux[k]
        for name, key in [('acc_im', 'sim_im'), ('acc_text', 'sim_text'), ('acc_sum2', 'sim_sum2')]:
            _, pred = out[key].max(-1)
            meters[name] += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            n = idx + 1
            print(
                f'Train epoch: {epoch}, step: {idx:3d}, '
                f'loss: {meters["loss"] / n:.4f}, '
                f'supcon: {meters["supcon"] / n:.4f}, clip_kd: {meters["clip_kd"] / n:.4f}, '
                f'text_drop: {meters["text_drop"] / n:.2f}, '
                f'acc_im: {meters["acc_im"] * 100 / n:.2f}, '
                f'acc_text: {meters["acc_text"] * 100 / n:.2f}, '
                f'acc_sum2: {meters["acc_sum2"] * 100 / n:.2f}'
            )

    n = len(train_loader)
    args.logger.add_scalar('train/loss', meters['loss'] / n, epoch)
    args.logger.add_scalar('train/acc_sum2', meters['acc_sum2'] / n, epoch)
    if args.supcon_weight > 0:
        args.logger.add_scalar('train/supcon', meters['supcon'] / n, epoch)
    if args.clip_kd_weight > 0:
        args.logger.add_scalar('train/clip_kd', meters['clip_kd'] / n, epoch)
    if args.text_dropout_p > 0:
        args.logger.add_scalar('train/text_drop_rate', meters['text_drop'] / n, epoch)


# During training validation: only main metrics.
# With --test: also report text+alpha*im fusion sweep (alpha=0.2..1.0).
_TRAIN_TEST_METRICS = [
    ('acc_im', 'sim_im'),
    ('acc_text', 'sim_text'),
    ('acc_sum10', 'sim_sum10'),
]

_FULL_TEST_METRICS = [
    ('acc_im', 'sim_im'),
    ('acc_text', 'sim_text'),
    ('acc_sum2', 'sim_sum2'),   # alpha=0.2
    ('acc_sum4', 'sim_sum4'),   # alpha=0.4
    ('acc_sum6', 'sim_sum6'),   # alpha=0.6
    ('acc_sum8', 'sim_sum8'),   # alpha=0.8
    ('acc_sum10', 'sim_sum10'), # alpha=1.0
]


def test(prompt_learner, prompts, tokenized_prompts, text_encoder, student, test_loader, epoch, args):
    student.eval()
    prompt_learner.eval()
    metric_pairs = _FULL_TEST_METRICS if args.test else _TRAIN_TEST_METRICS
    buckets = {name: [] for name, _ in metric_pairs}

    with torch.no_grad():
        for episode in test_loader:
            image = episode[0].cuda(args.gpu)
            glabels = episode[1].cuda(args.gpu)
            labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

            image = image.view(args.way, args.shot + 15, *image.shape[1:])
            sup = image[:, :args.shot].contiguous().view(-1, *image.shape[2:])
            que = image[:, args.shot:].contiguous().view(-1, *image.shape[2:])

            glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot].contiguous().view(-1)
            idx = glabels + args.delta

            out = forward_episode(
                prompts, tokenized_prompts, text_encoder, student, None,
                sup, que, idx, args.way, args.shot, args, use_text=True,
            )

            for name, key in metric_pairs:
                _, pred = out[key].max(-1)
                buckets[name].append(labels.eq(pred).sum().float().item() / labels.shape[0])

    for name, _ in metric_pairs:
        m, h = mean_confidence_interval(buckets[name])
        print(f'{name} Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')

    m, _ = mean_confidence_interval(buckets['acc_sum10'])
    args.logger.add_scalar('test/acc', m * 100, epoch)
    if args.test:
        for name, _ in metric_pairs:
            if name.startswith('acc_sum'):
                mm, _ = mean_confidence_interval(buckets[name])
                args.logger.add_scalar(f'test/{name}', mm * 100, epoch)
    return m


def main(args):
    args.tensorboard_dir = f'tensorboard/{args.dataset}/{args.model}/{args.exp}/'
    args.checkpoint_dir = f'checkpoint/{args.dataset}/{args.model}/{args.exp}/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

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

    teacher, _ = clip.load('ViT-B/32', device=f'cuda:{args.gpu}')
    teacher.float()
    teacher.requires_grad_(False)
    teacher.eval()

    train_names = [train_dataset.idx2text[i] for i in train_dataset.dataset.classes]
    test_names = [test_dataset.idx2text[i] for i in test_dataset.dataset.classes]
    all_classnames = train_names + test_names

    text_encoder = TextEncoder(teacher).cuda()
    prompt_learner = PromptLearner(all_classnames, teacher).cuda()
    tokenized_prompts = prompt_learner.tokenized_prompts
    prompts = prompt_learner()

    student = visformer.visformer_tiny(num_classes=num_classes, drop_rate=args.dropout)
    feature_dim = 192 if 2 <= args.stage < 3 else 384
    if args.adaptor == 'linear':
        student.adaptor = nn.Linear(512, feature_dim, bias=False)
    elif args.adaptor == 'mlp':
        student.adaptor = nn.Sequential(
            nn.Linear(512, feature_dim // 4), nn.LeakyReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
        )
    else:
        student.adaptor = adaptor(512, feature_dim)
    student = student.cuda(args.gpu)

    optim_ids = {id(p) for p in student.adaptor.parameters()}
    optim = torch.optim.AdamW([
        {'params': [p for p in student.parameters() if id(p) in optim_ids], 'lr': args.lr},
        {'params': [p for p in student.parameters() if id(p) not in optim_ids], 'lr': args.encoder_lr},
        {'params': prompt_learner.parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    if args.resume:
        args.init = args.resume
    if not args.init:
        raise ValueError('must provide pre-trained model')
    student.load_state_dict(torch.load(args.init, map_location=f'cuda:{args.gpu}')['state_dict'], strict=False)

    active = []
    if args.supcon_weight > 0:
        active.append(f'supcon={args.supcon_weight}')
    if args.text_dropout_p > 0:
        active.append(f'text_drop={args.text_dropout_p}')
    if args.clip_kd_weight > 0:
        active.append(f'clip_kd={args.clip_kd_weight}')
    print('Regularizers:', ', '.join(active) if active else 'none (baseline)')

    if args.test:
        print(
            f'[test mode] way={args.way}, shot={args.shot}, episodes={args.test_episodes}, '
            f'fusion sweep: acc_sum2/4/6/8/10 (sim_text + alpha * sim_im)'
        )
        test(prompt_learner, prompts, tokenized_prompts, text_encoder, student, test_loader, 0, args)
        return

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=50)
    best_acc, best_epoch = 0., 0

    for epoch in range(args.epochs):
        train(
            prompt_learner, prompts, tokenized_prompts, text_encoder, teacher, student,
            train_loader, optim, epoch, args,
        )
        acc = 0.
        if (epoch + 1) % args.test_freq == 0:
            acc = test(prompt_learner, prompts, tokenized_prompts, text_encoder, student, test_loader, epoch, args)
        if args.sheduler == 'True':
            scheduler.step(acc)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
            'prompt_learner': prompt_learner.state_dict(),
        }
        torch.save(checkpoint, args.checkpoint_dir + 'checkpoint_epoch_latest.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_{epoch + 1:03d}.pth')
        if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
            best_acc, best_epoch = acc, epoch
            torch.save(checkpoint, args.checkpoint_dir + 'checkpoint_epoch_best.pth')
        print(f'best_epoch: {best_epoch}, best_acc: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='reg')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR-FS',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true', default=True)
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--model', type=str, default='visformer-t')
    parser.add_argument('--stage', type=float, default=3.2)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--encoder_lr', type=float, default=1e-6)
    parser.add_argument('--init', type=str, default='checkpoint/cifar/checkpoint_epoch_800.pth')
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
    # --- regularizers (0 / 0.0 = disabled) ---
    parser.add_argument('--supcon_weight', type=float, default=0.0,
                        help='Episode SupCon on visual features; try 0.05–0.2')
    parser.add_argument('--supcon_t', type=float, default=0.07)
    parser.add_argument('--text_dropout_p', type=float, default=0.0,
                        help='Prob. of visual-only step (no text fusion); try 0.2–0.5')
    parser.add_argument('--clip_kd_weight', type=float, default=0.0,
                        help='Relational CLIP visual KD on support prototypes; try 0.3–1.0')

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
