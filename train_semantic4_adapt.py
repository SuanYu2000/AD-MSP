"""
train_semantic4_adapt: attribute-role prompts, episode routing, support-adaptive fusion.

Compared with train_semantic4_reg.py:
  - default templates are category roles (identity / parts / appearance / context),
    not imaging conditions; pass --template_set imaging to switch back
  - branch weights come from the current support prototypes (EpisodeRouter),
    with an entropy floor so a branch cannot be zeroed out
  - fusion alpha = sigmoid(w_agree * agreement + w_shot * log(shot) + b)
    and the training classification loss uses sim_text + alpha * sim_im
  - training episodes randomly use 1 or 5 support images so alpha sees both shots
  - SupCon / text-dropout / CLIP-KD are not part of this method

Train (mixed 1/5-shot):
  python train_semantic4_adapt.py --gpu 0 --dataset miniImageNet \\
    --init checkpoint/miniImageNet/visformer-t/pre-train-1shot/checkpoint_epoch_580.pth

Test 1-shot and 5-shot separately (adaptive alpha is the reported score;
fixed-alpha lines are diagnostics only):
  python train_semantic4_adapt.py --gpu 0 --dataset miniImageNet --test \\
    --resume checkpoint/.../checkpoint_epoch_best.pth --shot 1
  python train_semantic4_adapt.py --gpu 0 --dataset miniImageNet --test \\
    --resume checkpoint/.../checkpoint_epoch_best.pth --shot 5
"""

import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import clip

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from lib import visformer
from lib.multi_prompt_learner import (
    ATTRIBUTE4_ROLES,
    ATTRIBUTE4_TEMPLATES,
    SEMANTIC4_ROLES,
    SEMANTIC4_TEMPLATES,
    MultiPromptLearner,
)
from lib.episode_router import (
    EpisodeRouter,
    SupportFusion,
    aggregate_with_weights,
    routing_entropy,
    routing_entropy_penalty,
    routing_weights,
)
from lib.utils import mean_confidence_interval
from data.dataloader import EpisodeSampler
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC
from lib.train_FSL_mp2 import (
    JS_div,
    TextEncoder,
    build_adaptor,
    encode_sample_text,
    per_prompt_ce_loss,
    prototype_diversity_loss,
    text_diversity_loss,
)

NUM_PROMPTS = 4
QUERY_PER_CLASS = 15
MAX_SUPPORT = 5
TEMPLATE_SETS = {
    'attribute': (ATTRIBUTE4_TEMPLATES, ATTRIBUTE4_ROLES),
    'imaging': (SEMANTIC4_TEMPLATES, SEMANTIC4_ROLES),
}

_STUDENT_MEAN = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]], dtype=np.float32)
_STUDENT_STD = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]], dtype=np.float32)

_TRAIN_TEST_METRICS = [
    ('acc_im', 'sim_im'),
    ('acc_text', 'sim_text'),
    ('acc_adapt', 'sim_adapt'),
]
_DIAGNOSTIC_ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)


def slice_episode(episode, args, way, support_shot, images_per_class):
    """Take `support_shot` support images and the last 15 query images per class."""
    image = episode[0].cuda(args.gpu)
    glabels = episode[1].cuda(args.gpu)
    image = image.view(way, images_per_class, *image.shape[1:])
    support_budget = images_per_class - QUERY_PER_CLASS
    if support_shot > support_budget:
        raise ValueError(f'support_shot {support_shot} exceeds budget {support_budget}')
    sup = image[:, :support_shot].contiguous().view(-1, *image.shape[2:])
    que = image[:, support_budget:].contiguous().view(-1, *image.shape[2:])
    class_ids = glabels.view(way, images_per_class)[:, :support_shot].contiguous().view(-1)
    labels = torch.arange(way, device=image.device).unsqueeze(-1).repeat(1, QUERY_PER_CLASS).view(-1)
    return sup, que, class_ids, labels


def forward_episode(
    prompt_learner, text_encoder, student, router, fusion,
    sup, que, class_ids, way, shot, args, class_offset=0,
):
    class_ids = class_ids + class_offset
    _, sup_im = student(sup)
    _, que_im = student(que)
    que_im = F.normalize(que_im, dim=-1)

    im_proto = sup_im.view(way, shot, -1).mean(dim=1)
    im_proto = F.normalize(im_proto, dim=-1)
    sim_im = que_im @ im_proto.t()

    n_prompts = prompt_learner.n_prompts
    text_feat = encode_sample_text(prompt_learner, text_encoder, class_ids, args.eqnorm)
    text_adapt = student.adaptor(text_feat)
    sup_sem = sup_im.unsqueeze(1) + text_adapt
    proto_sem = sup_sem.view(way, shot, n_prompts, -1).mean(dim=1)
    proto_sem = F.normalize(proto_sem, dim=-1)
    sim_stack = torch.einsum('qd,wnd->qwn', que_im, proto_sem)

    weights = routing_weights(router(im_proto))
    sim_text = aggregate_with_weights(sim_stack, weights)
    alpha, agreement = fusion(im_proto, proto_sem, shot)
    sim_adapt = sim_text + alpha * sim_im

    out = {
        'sim_im': sim_im,
        'sim_text': sim_text,
        'sim_stack': sim_stack,
        'sim_adapt': sim_adapt,
        'text_features': text_feat,
        'proto_sem': proto_sem,
        'im_proto': im_proto,
        'route_weights': weights,
        'route_entropy': routing_entropy(weights),
        'alpha': alpha,
        'agreement': agreement,
    }
    for value in _DIAGNOSTIC_ALPHAS:
        out[f'sim_sum{int(value * 10)}'] = sim_text + value * sim_im
    return out


def compute_loss(out, labels, args):
    sim_im, sim_text, sim_adapt = out['sim_im'], out['sim_text'], out['sim_adapt']
    loss = (
        F.cross_entropy(sim_adapt / args.t, labels)
        + F.cross_entropy(sim_im / args.t, labels)
        + F.cross_entropy(sim_text / args.t, labels)
        + args.KD * JS_div(sim_im / args.t, sim_text / args.t)
    )
    aux = {
        'alpha': float(out['alpha'].detach()),
        'agreement': float(out['agreement'].detach()),
        'route_entropy': float(out['route_entropy'].detach()),
    }
    weights = out['route_weights'].detach()
    aux['route_weights'] = weights.float().cpu()

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
    if args.route_ent_weight > 0:
        penalty, _ = routing_entropy_penalty(
            out['route_weights'], args.num_prompts, floor_ratio=args.route_ent_floor,
        )
        loss = loss + args.route_ent_weight * penalty
        aux['route_pen'] = penalty.item()
    return loss, aux


def _print_config(args, templates, roles):
    print(f'[semantic4_adapt] template_set={args.template_set}, mixed_shot={args.mixed_shot}')
    for i, (text, role) in enumerate(zip(templates, roles)):
        print(f'  P{i}: "{text}"  ({role})')
    print(
        f'[semantic4_adapt] route_ent={args.route_ent_weight} (floor={args.route_ent_floor}), '
        f'div_text={args.div_text_weight}, div_proto={args.div_proto_weight}, pp_ce={args.pp_ce_weight}'
    )


def _sample_train_shot(args):
    if args.mixed_shot:
        return random.choice((1, MAX_SUPPORT))
    return args.shot


def train(prompt_learner, text_encoder, student, router, fusion, train_loader, optim, epoch, args):
    student.train()
    prompt_learner.train()
    router.train()
    fusion.train()
    meters = {k: 0.0 for k in ['loss', 'acc_adapt', 'alpha', 'route_entropy']}
    weight_sum = torch.zeros(NUM_PROMPTS)

    for idx, episode in enumerate(train_loader):
        shot = _sample_train_shot(args)
        sup, que, class_ids, labels = slice_episode(
            episode, args, args.train_way, shot, args.train_images_per_class,
        )
        out = forward_episode(
            prompt_learner, text_encoder, student, router, fusion,
            sup, que, class_ids, args.train_way, shot, args, class_offset=0,
        )
        loss, aux = compute_loss(out, labels, args)
        meters['loss'] += loss.item()
        meters['alpha'] += aux['alpha']
        meters['route_entropy'] += aux['route_entropy']
        weight_sum += aux['route_weights']
        _, pred = out['sim_adapt'].max(-1)
        meters['acc_adapt'] += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            n = idx + 1
            w = weight_sum / n
            print(
                f'Train epoch: {epoch}, step: {idx:3d}, loss: {meters["loss"] / n:.4f}, '
                f'acc_adapt: {meters["acc_adapt"] / n * 100:.2f}, '
                f'alpha: {meters["alpha"] / n:.3f}, entropy: {meters["route_entropy"] / n:.3f}, '
                f'weights: {" ".join(f"{v:.3f}" for v in w.tolist())}'
            )

    n = len(train_loader)
    args.logger.add_scalar('train/loss', meters['loss'] / n, epoch)
    args.logger.add_scalar('train/acc_adapt', meters['acc_adapt'] / n, epoch)
    args.logger.add_scalar('train/alpha', meters['alpha'] / n, epoch)
    args.logger.add_scalar('train/route_entropy', meters['route_entropy'] / n, epoch)


@torch.no_grad()
def test(prompt_learner, text_encoder, student, router, fusion, test_loader, epoch, args):
    student.eval()
    prompt_learner.eval()
    router.eval()
    fusion.eval()
    metric_pairs = list(_TRAIN_TEST_METRICS)
    if args.test:
        metric_pairs += [(f'acc_sum{int(a * 10)}', f'sim_sum{int(a * 10)}') for a in _DIAGNOSTIC_ALPHAS]
    buckets = {name: [] for name, _ in metric_pairs}
    weight_sum = torch.zeros(NUM_PROMPTS)
    entropies, alphas, agreements = [], [], []

    for episode in test_loader:
        sup, que, class_ids, labels = slice_episode(
            episode, args, args.way, args.shot, args.shot + QUERY_PER_CLASS,
        )
        out = forward_episode(
            prompt_learner, text_encoder, student, router, fusion,
            sup, que, class_ids, args.way, args.shot, args, class_offset=args.delta,
        )
        weight_sum += out['route_weights'].detach().float().cpu()
        entropies.append(float(out['route_entropy']))
        alphas.append(float(out['alpha']))
        agreements.append(float(out['agreement']))
        for name, key in metric_pairs:
            _, pred = out[key].max(-1)
            buckets[name].append(labels.eq(pred).sum().float().item() / labels.shape[0])

    n = len(test_loader)
    mean_w = weight_sum / max(n, 1)
    for name, _ in metric_pairs:
        m, h = mean_confidence_interval(buckets[name])
        print(f'{name} Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    print(
        f'route Test epoch: {epoch}, weights: {" ".join(f"{v:.3f}" for v in mean_w.tolist())}, '
        f'entropy: {np.mean(entropies):.3f}, min_weight: {mean_w.min().item():.3f}'
    )
    print(
        f'fusion Test epoch: {epoch}, alpha: {np.mean(alphas):.3f}, '
        f'agreement: {np.mean(agreements):.3f}, shot: {args.shot}'
    )

    m, _ = mean_confidence_interval(buckets['acc_adapt'])
    args.logger.add_scalar('test/acc_adapt', m * 100, epoch)
    args.logger.add_scalar('test/alpha', float(np.mean(alphas)), epoch)
    args.logger.add_scalar('test/route_entropy', float(np.mean(entropies)), epoch)
    if args.test:
        for name, _ in metric_pairs:
            if name.startswith('acc_sum'):
                mm, _ = mean_confidence_interval(buckets[name])
                args.logger.add_scalar(f'test/{name}', mm * 100, epoch)
    return m


def _build_loader(dataset, n_episodes, way, images_per_class, workers, fix_seed):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=EpisodeSampler(
            dataset.dataset.targets, n_episodes, way, images_per_class, fix_seed=fix_seed,
        ),
        num_workers=workers,
    )


def main(args):
    args.num_prompts = NUM_PROMPTS
    templates, roles = TEMPLATE_SETS[args.template_set]
    device = torch.device(f'cuda:{args.gpu}')
    args.tensorboard_dir = f'tensorboard/{args.dataset}/{args.model}/{args.exp}/'
    args.checkpoint_dir = f'checkpoint/{args.dataset}/{args.model}/{args.exp}/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)
    _print_config(args, templates, roles)

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
    args.train_images_per_class = (MAX_SUPPORT if args.mixed_shot else args.shot) + QUERY_PER_CLASS
    n_episodes = args.train_episodes
    if n_episodes == -1:
        n_episodes = int(len(train_dataset) / (args.train_way * args.train_images_per_class))

    train_loader = None
    if not args.test:
        train_loader = _build_loader(
            train_dataset, n_episodes, args.train_way, args.train_images_per_class,
            args.workers, fix_seed=False,
        )
    num_classes = len(train_dataset.dataset.classes)

    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split=args.split)
    test_loader = _build_loader(
        test_dataset, args.test_episodes, args.way, args.shot + QUERY_PER_CLASS,
        args.workers, fix_seed=True,
    )

    teacher, _ = clip.load('ViT-B/32', device=device)
    teacher.float()
    teacher.requires_grad_(False)
    teacher.eval()

    train_names = [train_dataset.idx2text[i] for i in train_dataset.dataset.classes]
    test_names = [test_dataset.idx2text[i] for i in test_dataset.dataset.classes]
    prompt_learner = MultiPromptLearner(
        train_names + test_names, teacher,
        num_prompts=NUM_PROMPTS,
        dataset_name=args.dataset,
        n_ctx=args.n_ctx,
        template_list=templates,
    ).to(device)
    text_encoder = TextEncoder(teacher).to(device)

    student = visformer.visformer_tiny(num_classes=num_classes, drop_rate=args.dropout)
    feature_dim = 192 if 2 <= args.stage < 3 else 384
    build_adaptor(student, 512, feature_dim, args)
    student = student.to(device)
    router = EpisodeRouter(feature_dim, NUM_PROMPTS).to(device)
    fusion = SupportFusion().to(device)

    optim = torch.optim.AdamW([
        {'params': list(student.adaptor.parameters()) + list(prompt_learner.parameters())
         + list(router.parameters()) + list(fusion.parameters()), 'lr': args.lr},
        {'params': [p for p in student.parameters() if id(p) not in {id(q) for q in student.adaptor.parameters()}],
         'lr': args.encoder_lr},
    ], weight_decay=args.weight_decay)

    if args.resume:
        args.init = args.resume
    if not args.init:
        raise ValueError('must provide pre-trained model')

    ckpt = torch.load(args.init, map_location=device)
    student.load_state_dict(ckpt['state_dict'], strict=False)
    print(f'Loaded init from {args.init}')

    start_epoch = 0
    loaded_router = False
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt['state_dict'], strict=False)
        if 'prompt_learner' in ckpt:
            prompt_learner.load_state_dict(ckpt['prompt_learner'])
        if 'prompt_learner_n_ctx_list' in ckpt:
            prompt_learner.n_ctx_list = list(ckpt['prompt_learner_n_ctx_list'])
            prompt_learner.n_ctx = max(prompt_learner.n_ctx_list)
        if 'router' in ckpt:
            router.load_state_dict(ckpt['router'])
            loaded_router = True
        if 'fusion' in ckpt:
            fusion.load_state_dict(ckpt['fusion'])
        if 'optimizer' in ckpt and not args.test:
            optim.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')
    if args.test and not loaded_router:
        print('warning: checkpoint has no router/fusion; they stay at initialization')

    if args.test:
        print(
            f'[test mode] way={args.way}, shot={args.shot}, episodes={args.test_episodes}, '
            f'reported score is acc_adapt; acc_sum* is a fixed-alpha diagnostic'
        )
        test(prompt_learner, text_encoder, student, router, fusion, test_loader, 0, args)
        return

    best_acc, best_epoch = 0.0, 0
    for epoch in range(start_epoch, args.epochs):
        train(prompt_learner, text_encoder, student, router, fusion, train_loader, optim, epoch, args)
        acc = 0.0
        if (epoch + 1) % args.test_freq == 0:
            acc = test(prompt_learner, text_encoder, student, router, fusion, test_loader, epoch, args)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
            'prompt_learner': prompt_learner.state_dict(),
            'prompt_learner_n_ctx_list': prompt_learner.n_ctx_list,
            'router': router.state_dict(),
            'fusion': fusion.state_dict(),
            'template_set': args.template_set,
        }
        torch.save(checkpoint, args.checkpoint_dir + 'checkpoint_epoch_latest.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_{epoch + 1:03d}.pth')
        if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(checkpoint, args.checkpoint_dir + 'checkpoint_epoch_best.pth')
        print(f'best_epoch: {best_epoch}, best_acc: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attribute-role prompts with episode routing and adaptive fusion')
    parser.add_argument('--exp', type=str, default='semantic4_adapt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
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
    parser.add_argument('--template_set', type=str, default='attribute', choices=['attribute', 'imaging'])
    parser.add_argument('--mixed_shot', action='store_true', default=True)
    parser.add_argument('--no_mixed_shot', action='store_false', dest='mixed_shot')
    parser.add_argument('--route_ent_weight', type=float, default=0.1)
    parser.add_argument('--route_ent_floor', type=float, default=0.5,
                        help='penalize routing entropy below log(N) * this ratio')
    parser.add_argument('--div_text_weight', type=float, default=0.05)
    parser.add_argument('--div_proto_weight', type=float, default=0.05)
    parser.add_argument('--pp_ce_weight', type=float, default=0.3)
    parser.add_argument('--div_margin', type=float, default=0.2)
    parser.add_argument('--workers', type=int, default=8)

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
