"""
Select visual-semantic views by whether they separate the episode's classes.

Candidates are the imaging and attribute templates together. Each view attends
over stage1+stage2+stage3, so the attended location is not assigned by hand.
Support-set class separation plus a learned bias picks top-k views. Only those
views are mixed into the class prototype used for training and testing.

The Visformer trunk still loads from a train_vit.py checkpoint.

Train:
  python train_semantic4_select.py --gpu 0 --dataset miniImageNet \\
    --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth

Test:
  python train_semantic4_select.py --gpu 0 --dataset miniImageNet --test \\
    --resume checkpoint/.../checkpoint_epoch_best.pth --shot 1
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
from lib.adaptive_view import (
    AdaptiveViewReadout,
    attention_floor_penalty,
    attention_overlap,
    class_separation,
    select_view_weights,
    stage_attention_mass,
)
from lib.multi_prompt_learner import MultiPromptLearner
from lib.utils import mean_confidence_interval
from data.dataloader import EpisodeSampler
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC
from lib.train_FSL_mp2 import (
    JS_div,
    TextEncoder,
    build_adaptor,
    encode_sample_text,
)
from train_semantic4_adapt import QUERY_PER_CLASS, _STUDENT_MEAN, _STUDENT_STD, slice_episode
from train_semantic4_view import encode_neutral_views

CANDIDATE_TEMPLATES = [
    'a photo of a',
    'a close-up photo of a',
    'a cropped photo of a',
    'a photo of a in natural light',
    'a photo showing the distinctive parts of a',
    'a photo showing the color and texture of a',
    'a photo of a in its typical surroundings',
]
CANDIDATE_NAMES = [
    'identity', 'close-up', 'crop', 'lighting', 'parts', 'appearance', 'context',
]
SCALE_NAME = ('stage1', 'stage2', 'stage3')


def forward_episode(
    prompt_learner, text_encoder, student, readout,
    sup, que, class_ids, way, shot, args, class_offset=0,
):
    class_ids = class_ids + class_offset
    sup_global, sup_maps = student.forward_multiscale(sup)
    que_global, que_maps = student.forward_multiscale(que)
    que_im = F.normalize(que_global, dim=-1)
    im_proto = F.normalize(sup_global.view(way, shot, -1).mean(dim=1), dim=-1)
    sim_im = que_im @ im_proto.t()

    ctx = prompt_learner.ctx
    f_sup, attn_sup, lengths = readout(sup_maps, ctx, prompt_learner.n_ctx_list)
    f_que, attn_que, _ = readout(que_maps, ctx, prompt_learner.n_ctx_list)

    text_feat = encode_sample_text(prompt_learner, text_encoder, class_ids, args.eqnorm)
    text_adapt = student.adaptor(text_feat)
    sup_sem = f_sup + text_adapt
    proto_sem = F.normalize(sup_sem.view(way, shot, prompt_learner.n_prompts, -1).mean(dim=1), dim=-1)
    f_que = F.normalize(f_que, dim=-1)
    sim_stack = torch.einsum('qpd,wpd->qwp', f_que, proto_sem)

    scores = class_separation(proto_sem) + readout.score_bias
    weights, index = select_view_weights(scores, args.top_k)
    sim_text = torch.einsum('qwp,p->qw', sim_stack, weights)

    out = {
        'sim_im': sim_im,
        'sim_text': sim_text,
        'sim_stack': sim_stack,
        'weights': weights,
        'selected': index,
        'scores': scores,
        'attn_sup': attn_sup,
        'attn_que': attn_que,
        'stage_len': lengths,
        'stage_mass': stage_attention_mass(attn_que, lengths),
    }
    return out


def compute_loss(out, labels, args):
    sim_im, sim_text = out['sim_im'], out['sim_text']
    loss = (
        F.cross_entropy(sim_text / args.t, labels)
        + F.cross_entropy(sim_im / args.t, labels)
        + args.KD * JS_div(sim_im / args.t, sim_text / args.t)
    )
    weights = out['weights']
    pp = sim_text.new_zeros(())
    for i in range(weights.shape[0]):
        pp = pp + weights[i] * F.cross_entropy(out['sim_stack'][:, :, i] / args.t, labels)
    loss = loss + args.pp_ce_weight * pp

    overlap_s = attention_overlap(out['attn_sup'])
    overlap_q = attention_overlap(out['attn_que'])
    floor_s, _ = attention_floor_penalty(out['attn_sup'], args.attn_floor)
    floor_q, _ = attention_floor_penalty(out['attn_que'], args.attn_floor)
    loss = loss + args.attn_div_weight * 0.5 * (overlap_s + overlap_q)
    loss = loss + args.attn_ent_weight * 0.5 * (floor_s + floor_q)
    return loss


def _format_selection(weights, names):
    order = torch.argsort(weights, descending=True)
    chosen = []
    for i in order.tolist():
        if float(weights[i]) <= 0:
            break
        chosen.append(f'{names[i]}:{float(weights[i]):.2f}')
    return ' '.join(chosen)


def _format_mass(mass, index):
    """mass: (P, 3). Average stage mass over the selected views."""
    selected = mass[index].mean(dim=0)
    return ' '.join(f'{SCALE_NAME[i]}:{float(selected[i]):.2f}' for i in range(selected.shape[0]))


def _build_loader(dataset, n_episodes, way, images_per_class, workers, fix_seed):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=EpisodeSampler(
            dataset.dataset.targets, n_episodes, way, images_per_class, fix_seed=fix_seed,
        ),
        num_workers=workers,
    )


def train(prompt_learner, text_encoder, student, readout, train_loader, optim, epoch, args):
    student.train()
    prompt_learner.train()
    readout.train()
    loss_sum = 0.0
    acc_sum = 0.0
    weight_sum = torch.zeros(len(args.templates))

    for idx, episode in enumerate(train_loader):
        sup, que, class_ids, labels = slice_episode(
            episode, args, args.train_way, args.shot, args.shot + QUERY_PER_CLASS,
        )
        out = forward_episode(
            prompt_learner, text_encoder, student, readout,
            sup, que, class_ids, args.train_way, args.shot, args,
        )
        loss = compute_loss(out, labels, args)
        loss_sum += loss.item()
        weight_sum += out['weights'].detach().float().cpu()
        _, pred = out['sim_text'].max(-1)
        acc_sum += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            n = idx + 1
            print(
                f'Train epoch: {epoch}, step: {idx:3d}, loss: {loss_sum / n:.4f}, '
                f'acc_text: {acc_sum / n * 100:.2f}, '
                f'selected: {_format_selection(out["weights"].detach(), args.names)}, '
                f'where: {_format_mass(out["stage_mass"].detach(), out["selected"].detach())}'
            )

    n = len(train_loader)
    args.logger.add_scalar('train/loss', loss_sum / n, epoch)
    args.logger.add_scalar('train/acc_text', acc_sum / n, epoch)
    for i, name in enumerate(args.names):
        args.logger.add_scalar(f'train/select/{name}', float(weight_sum[i]) / n, epoch)


@torch.no_grad()
def test(prompt_learner, text_encoder, student, readout, test_loader, epoch, args):
    student.eval()
    prompt_learner.eval()
    readout.eval()
    acc_im, acc_text = [], []
    weight_sum = torch.zeros(len(args.templates))
    mass_sum = torch.zeros(len(args.templates), 3)

    for episode in test_loader:
        sup, que, class_ids, labels = slice_episode(
            episode, args, args.way, args.shot, args.shot + QUERY_PER_CLASS,
        )
        out = forward_episode(
            prompt_learner, text_encoder, student, readout,
            sup, que, class_ids, args.way, args.shot, args, class_offset=args.delta,
        )
        weight_sum += out['weights'].detach().float().cpu()
        mass_sum += out['stage_mass'].detach().float().cpu()
        for bucket, logits in ((acc_im, out['sim_im']), (acc_text, out['sim_text'])):
            _, pred = logits.max(-1)
            bucket.append(labels.eq(pred).sum().float().item() / labels.shape[0])

    n = max(len(test_loader), 1)
    mean_w = weight_sum / n
    mean_mass = mass_sum / n
    for name, bucket in (('acc_im', acc_im), ('acc_text', acc_text)):
        m, h = mean_confidence_interval(bucket)
        print(f'{name} Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    print(f'selected Test epoch: {epoch}, {_format_selection(mean_w, args.names)}')
    for i, name in enumerate(args.names):
        mass = ' '.join(f'{SCALE_NAME[s]}:{float(mean_mass[i, s]):.2f}' for s in range(3))
        print(f'  {name}: weight {float(mean_w[i]):.3f}, attention {mass}')
    m, _ = mean_confidence_interval(acc_text)
    args.logger.add_scalar('test/acc_text', m * 100, epoch)
    return m


def main(args):
    args.templates = list(CANDIDATE_TEMPLATES)
    args.names = list(CANDIDATE_NAMES)
    if args.top_k > len(args.templates):
        args.top_k = len(args.templates)
    device = torch.device(f'cuda:{args.gpu}')
    args.tensorboard_dir = f'tensorboard/{args.dataset}/{args.model}/{args.exp}/'
    args.checkpoint_dir = f'checkpoint/{args.dataset}/{args.model}/{args.exp}/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)
    print(f'[semantic4_select] candidates={len(args.templates)}, top_k={args.top_k}')
    for name, text in zip(args.names, args.templates):
        print(f'  {name}: "{text}"')

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
        n_episodes = int(len(train_dataset) / (args.train_way * (args.shot + QUERY_PER_CLASS)))
    train_loader = None
    if not args.test:
        train_loader = _build_loader(
            train_dataset, n_episodes, args.train_way, args.shot + QUERY_PER_CLASS,
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
    text_encoder = TextEncoder(teacher).to(device)
    train_names = [train_dataset.idx2text[i] for i in train_dataset.dataset.classes]
    test_names = [test_dataset.idx2text[i] for i in test_dataset.dataset.classes]
    prompt_learner = MultiPromptLearner(
        train_names + test_names, teacher,
        num_prompts=len(args.templates),
        dataset_name=args.dataset,
        n_ctx=args.n_ctx,
        template_list=args.templates,
    ).to(device)

    student = visformer.visformer_tiny(num_classes=num_classes, drop_rate=args.dropout).to(device)
    with torch.no_grad():
        student.eval()
        probe, maps = student.forward_multiscale(
            torch.zeros(1, 3, args.image_size, args.image_size, device=device),
        )
    feature_dim = int(probe.shape[-1])
    build_adaptor(student, 512, feature_dim, args)
    student.adaptor.to(device)
    readout = AdaptiveViewReadout(
        [int(m.shape[1]) for m in maps], feature_dim, len(args.templates),
    ).to(device)
    view_text, _ = encode_neutral_views(teacher, text_encoder, args.templates)
    readout.set_view_text(view_text)

    head_params = (
        list(student.adaptor.parameters())
        + list(prompt_learner.parameters())
        + list(readout.parameters())
    )
    adaptor_ids = {id(p) for p in student.adaptor.parameters()}
    optim = torch.optim.AdamW([
        {'params': head_params, 'lr': args.lr},
        {'params': [p for p in student.parameters() if id(p) not in adaptor_ids], 'lr': args.encoder_lr},
    ], weight_decay=args.weight_decay)

    if args.resume:
        args.init = args.resume
    if not args.init:
        raise ValueError('must provide a train_vit.py checkpoint via --init')
    ckpt = torch.load(args.init, map_location=device)
    student.load_state_dict(ckpt['state_dict'], strict=False)
    print(f'Loaded Visformer trunk from {args.init}. Adaptor and view selector start untrained.')

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt['state_dict'], strict=False)
        if 'prompt_learner' in ckpt:
            prompt_learner.load_state_dict(ckpt['prompt_learner'])
        if 'prompt_learner_n_ctx_list' in ckpt:
            prompt_learner.n_ctx_list = list(ckpt['prompt_learner_n_ctx_list'])
            prompt_learner.n_ctx = max(prompt_learner.n_ctx_list)
        if 'readout' in ckpt:
            readout.load_state_dict(ckpt['readout'])
            readout.set_view_text(view_text)
        elif args.test:
            print('warning: checkpoint has no adaptive readout; it stays randomly initialized')
        if 'optimizer' in ckpt and not args.test:
            optim.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    if args.test:
        print(f'[test mode] way={args.way}, shot={args.shot}, episodes={args.test_episodes}, top_k={args.top_k}')
        test(prompt_learner, text_encoder, student, readout, test_loader, 0, args)
        return

    best_acc, best_epoch = 0.0, 0
    for epoch in range(start_epoch, args.epochs):
        train(prompt_learner, text_encoder, student, readout, train_loader, optim, epoch, args)
        acc = 0.0
        if (epoch + 1) % args.test_freq == 0:
            acc = test(prompt_learner, text_encoder, student, readout, test_loader, epoch, args)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
            'prompt_learner': prompt_learner.state_dict(),
            'prompt_learner_n_ctx_list': prompt_learner.n_ctx_list,
            'readout': readout.state_dict(),
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
    parser = argparse.ArgumentParser(description='select visual-semantic views for FSL prototypes')
    parser.add_argument('--exp', type=str, default='semantic4_select')
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
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--KD', type=float, default=1)
    parser.add_argument('--adaptor', type=str, default='mlp', choices=['linear', 'mlp', 'bottle'])
    parser.add_argument('--n_ctx', type=int, default=-1)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--pp_ce_weight', type=float, default=0.3)
    parser.add_argument('--attn_div_weight', type=float, default=0.05)
    parser.add_argument('--attn_ent_weight', type=float, default=0.05)
    parser.add_argument('--attn_floor', type=float, default=0.25)
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
