"""
View-conditioned visual readout on top of a train_vit.py Visformer checkpoint.

The trunk weights load from --init. The adaptor, prompt context, and the new
view readout are not in that checkpoint; this script trains them. Pretraining
does not need to be repeated.

Each view pools a different feature map with cross-attention:
  imaging P0 identity     stage3  high entropy
  imaging P1 close-up     stage2  low entropy
  imaging P2 composition  stage3  high entropy
  imaging P3 lighting     stage1  no spatial-entropy target

Train:
  python train_semantic4_view.py --gpu 0 --dataset miniImageNet \\
    --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth

Test (prints per-view accuracy and attention entropy, saves heatmaps):
  python train_semantic4_view.py --gpu 0 --dataset miniImageNet --test \\
    --resume checkpoint/.../checkpoint_epoch_best.pth --shot 1
"""

import os
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
from lib.view_readout import (
    ViewConditionedReadout,
    attention_cosine_penalty,
    attention_entropy_loss,
    neutral_sentence,
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
from train_semantic4_adapt import QUERY_PER_CLASS, _STUDENT_MEAN, _STUDENT_STD, slice_episode

NUM_PROMPTS = 4
# scale index into (stage1, stage2, stage3), entropy ratio or None to skip
VIEW_SPEC = {
    'imaging': (
        (2, 0.85),
        (1, 0.45),
        (2, 0.85),
        (0, None),
    ),
    'attribute': (
        (2, 0.85),
        (1, 0.45),
        (0, None),
        (2, 0.85),
    ),
}
SCALE_NAME = ('stage1', 'stage2', 'stage3')
TEMPLATE_SETS = {
    'attribute': (ATTRIBUTE4_TEMPLATES, ATTRIBUTE4_ROLES),
    'imaging': (SEMANTIC4_TEMPLATES, SEMANTIC4_ROLES),
}
_DIAGNOSTIC_ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0)


def encode_neutral_views(teacher, text_encoder, templates):
    sentences = [neutral_sentence(t) for t in templates]
    tokenized = clip.tokenize(sentences).to(next(teacher.parameters()).device)
    with torch.no_grad():
        emb = teacher.token_embedding(tokenized).type(teacher.dtype)
        text = text_encoder(emb, tokenized).float()
    return text, sentences


def _mean_entropies(attn_sup, attn_que):
    values = []
    for (alpha_s, _), (alpha_q, _) in zip(attn_sup, attn_que):
        ent_s = -(alpha_s * alpha_s.clamp(min=1e-8).log()).sum(-1).mean()
        ent_q = -(alpha_q * alpha_q.clamp(min=1e-8).log()).sum(-1).mean()
        values.append(0.5 * (ent_s + ent_q))
    return values


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
    f_sup, attn_sup = readout(sup_maps, ctx, prompt_learner.n_ctx_list)
    f_que, attn_que = readout(que_maps, ctx, prompt_learner.n_ctx_list)

    text_feat = encode_sample_text(prompt_learner, text_encoder, class_ids, args.eqnorm)
    text_adapt = student.adaptor(text_feat)
    sup_sem = f_sup + text_adapt
    proto_sem = sup_sem.view(way, shot, prompt_learner.n_prompts, -1).mean(dim=1)
    proto_sem = F.normalize(proto_sem, dim=-1)
    f_que = F.normalize(f_que, dim=-1)
    sim_stack = torch.einsum('qpd,wpd->qwp', f_que, proto_sem)
    sim_text = sim_stack.mean(dim=-1)

    out = {
        'sim_im': sim_im,
        'sim_text': sim_text,
        'sim_stack': sim_stack,
        'text_features': text_feat,
        'proto_sem': proto_sem,
        'attn_sup': attn_sup,
        'attn_que': attn_que,
        'view_entropy': _mean_entropies(attn_sup, attn_que),
    }
    for value in _DIAGNOSTIC_ALPHAS:
        out[f'sim_sum{int(value * 10)}'] = sim_text + value * sim_im
    return out


def compute_loss(out, labels, args):
    sim_im, sim_text = out['sim_im'], out['sim_text']
    loss = (
        F.cross_entropy(sim_text / args.t, labels)
        + F.cross_entropy(sim_im / args.t, labels)
        + args.KD * JS_div(sim_im / args.t, sim_text / args.t)
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

    ent_s, _ = attention_entropy_loss(out['attn_sup'], args.ent_ratios)
    ent_q, _ = attention_entropy_loss(out['attn_que'], args.ent_ratios)
    div_s = attention_cosine_penalty(out['attn_sup'])
    div_q = attention_cosine_penalty(out['attn_que'])
    if args.attn_ent_weight > 0:
        loss = loss + args.attn_ent_weight * 0.5 * (ent_s + ent_q)
    if args.attn_div_weight > 0:
        loss = loss + args.attn_div_weight * 0.5 * (div_s + div_q)
    aux['attn_ent'] = float(0.5 * (ent_s + ent_q).detach())
    aux['attn_div'] = float(0.5 * (div_s + div_q).detach())
    aux['view_entropy'] = [float(e.detach()) for e in out['view_entropy']]
    return loss, aux


def _save_heatmaps(image, attentions, directory, tag):
    os.makedirs(directory, exist_ok=True)
    mean = torch.tensor(_STUDENT_MEAN, device=image.device, dtype=image.dtype)[:, None, None]
    std = torch.tensor(_STUDENT_STD, device=image.device, dtype=image.dtype)[:, None, None]
    rgb = ((image * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    base = Image.fromarray(rgb)
    base.save(os.path.join(directory, f'{tag}_img.png'))
    height, width = rgb.shape[:2]
    for p, (alpha, (h, w)) in enumerate(attentions):
        heat = alpha.detach().float().view(1, 1, h, w)
        heat = F.interpolate(heat, size=(height, width), mode='bilinear', align_corners=False)[0, 0]
        heat = heat / heat.max().clamp(min=1e-6)
        heat_u8 = (heat.cpu().numpy() * 255).astype(np.uint8)
        red = np.stack([heat_u8, heat_u8 // 4, heat_u8 // 6], axis=-1)
        blend = Image.blend(base, Image.fromarray(red), 0.45)
        blend.save(os.path.join(directory, f'{tag}_p{p}.png'))


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
    meters = {k: 0.0 for k in ['loss', 'acc_text', 'attn_ent', 'attn_div']}
    ent_sum = np.zeros(NUM_PROMPTS, dtype=np.float64)

    for idx, episode in enumerate(train_loader):
        sup, que, class_ids, labels = slice_episode(
            episode, args, args.train_way, args.shot, args.shot + QUERY_PER_CLASS,
        )
        out = forward_episode(
            prompt_learner, text_encoder, student, readout,
            sup, que, class_ids, args.train_way, args.shot, args,
        )
        loss, aux = compute_loss(out, labels, args)
        meters['loss'] += loss.item()
        meters['attn_ent'] += aux['attn_ent']
        meters['attn_div'] += aux['attn_div']
        ent_sum += np.array(aux['view_entropy'])
        _, pred = out['sim_text'].max(-1)
        meters['acc_text'] += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            n = idx + 1
            ent = ent_sum / n
            print(
                f'Train epoch: {epoch}, step: {idx:3d}, loss: {meters["loss"] / n:.4f}, '
                f'acc_text: {meters["acc_text"] / n * 100:.2f}, '
                f'attn_div: {meters["attn_div"] / n:.3f}, attn_ent: {meters["attn_ent"] / n:.3f}, '
                f'entropy: {" ".join(f"{v:.2f}" for v in ent)}'
            )

    n = len(train_loader)
    args.logger.add_scalar('train/loss', meters['loss'] / n, epoch)
    args.logger.add_scalar('train/acc_text', meters['acc_text'] / n, epoch)


@torch.no_grad()
def test(prompt_learner, text_encoder, student, readout, test_loader, epoch, args):
    student.eval()
    prompt_learner.eval()
    readout.eval()
    names = ['acc_im', 'acc_text'] + [f'acc_p{p}' for p in range(NUM_PROMPTS)]
    if args.test:
        names += [f'acc_sum{int(a * 10)}' for a in _DIAGNOSTIC_ALPHAS]
    buckets = {name: [] for name in names}
    ent_sum = np.zeros(NUM_PROMPTS, dtype=np.float64)
    saved = 0

    for episode_idx, episode in enumerate(test_loader):
        sup, que, class_ids, labels = slice_episode(
            episode, args, args.way, args.shot, args.shot + QUERY_PER_CLASS,
        )
        out = forward_episode(
            prompt_learner, text_encoder, student, readout,
            sup, que, class_ids, args.way, args.shot, args, class_offset=args.delta,
        )
        ent_sum += np.array([float(e) for e in out['view_entropy']])
        pairs = [('acc_im', out['sim_im']), ('acc_text', out['sim_text'])]
        for p in range(NUM_PROMPTS):
            pairs.append((f'acc_p{p}', out['sim_stack'][:, :, p]))
        if args.test:
            for a in _DIAGNOSTIC_ALPHAS:
                key = f'sim_sum{int(a * 10)}'
                pairs.append((f'acc_sum{int(a * 10)}', out[key]))
        for name, logits in pairs:
            _, pred = logits.max(-1)
            buckets[name].append(labels.eq(pred).sum().float().item() / labels.shape[0])
        if args.test and saved < args.save_attn:
            one_image = [(alpha[0], hw) for alpha, hw in out['attn_que']]
            _save_heatmaps(
                que[0], one_image,
                os.path.join(args.checkpoint_dir, 'attn'),
                f'ep{episode_idx:02d}',
            )
            saved += 1

    for name in names:
        m, h = mean_confidence_interval(buckets[name])
        print(f'{name} Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    mean_ent = ent_sum / max(len(test_loader), 1)
    print(
        'attn Test epoch: '
        + ', '.join(
            f'P{p} {SCALE_NAME[args.view_scales[p]]} entropy {mean_ent[p]:.2f}'
            + ('' if args.ent_ratios[p] is None else f' target_ratio {args.ent_ratios[p]}')
            for p in range(NUM_PROMPTS)
        )
    )
    m, _ = mean_confidence_interval(buckets['acc_text'])
    args.logger.add_scalar('test/acc_text', m * 100, epoch)
    return m


def main(args):
    templates, roles = TEMPLATE_SETS[args.template_set]
    spec = VIEW_SPEC[args.template_set]
    args.view_scales = [scale for scale, _ in spec]
    args.ent_ratios = [ratio for _, ratio in spec]
    args.num_prompts = NUM_PROMPTS
    device = torch.device(f'cuda:{args.gpu}')

    args.tensorboard_dir = f'tensorboard/{args.dataset}/{args.model}/{args.exp}/'
    args.checkpoint_dir = f'checkpoint/{args.dataset}/{args.model}/{args.exp}/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

    print(f'[semantic4_view] template_set={args.template_set}')
    for i, (text, role) in enumerate(zip(templates, roles)):
        ratio = args.ent_ratios[i]
        ratio_text = 'none' if ratio is None else str(ratio)
        print(
            f'  P{i}: "{text}" ({role}) -> {SCALE_NAME[args.view_scales[i]]}, '
            f'neutral="{neutral_sentence(text)}", entropy_ratio={ratio_text}'
        )

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
        num_prompts=NUM_PROMPTS,
        dataset_name=args.dataset,
        n_ctx=args.n_ctx,
        template_list=templates,
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
    scale_dims = [int(m.shape[1]) for m in maps]
    print(
        '[semantic4_view] maps: '
        + ', '.join(f'{SCALE_NAME[i]} {tuple(m.shape[-2:])}x{m.shape[1]}' for i, m in enumerate(maps))
        + f', global_dim {feature_dim}'
    )
    readout = ViewConditionedReadout(scale_dims, feature_dim, NUM_PROMPTS, args.view_scales).to(device)
    view_text, _ = encode_neutral_views(teacher, text_encoder, templates)
    readout.set_view_text(view_text)

    head_params = (
        list(student.adaptor.parameters())
        + list(prompt_learner.parameters())
        + list(readout.parameters())
    )
    head_ids = {id(p) for p in student.adaptor.parameters()}
    optim = torch.optim.AdamW([
        {'params': head_params, 'lr': args.lr},
        {'params': [p for p in student.parameters() if id(p) not in head_ids], 'lr': args.encoder_lr},
    ], weight_decay=args.weight_decay)

    if args.resume:
        args.init = args.resume
    if not args.init:
        raise ValueError('must provide a train_vit.py checkpoint via --init')
    ckpt = torch.load(args.init, map_location=device)
    student.load_state_dict(ckpt['state_dict'], strict=False)
    print(
        f'Loaded Visformer trunk from {args.init}. '
        'train_vit weights are reused; adaptor and view readout are not in that file.'
    )

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
            print('warning: checkpoint has no view readout; it stays randomly initialized')
        if 'optimizer' in ckpt and not args.test:
            optim.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    if args.test:
        print(
            f'[test mode] way={args.way}, shot={args.shot}, episodes={args.test_episodes}. '
            'Reported score is acc_text. acc_sum* is a fixed-alpha diagnostic. '
            f'Attention maps go to {args.checkpoint_dir}attn/'
        )
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
    parser = argparse.ArgumentParser(description='view-conditioned visual readout for multi-prompt FSL')
    parser.add_argument('--exp', type=str, default='semantic4_view')
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
    parser.add_argument('--save_attn', type=int, default=4,
                        help='how many test episodes to dump attention overlays for, only with --test')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--KD', type=float, default=1)
    parser.add_argument('--adaptor', type=str, default='mlp', choices=['linear', 'mlp', 'bottle'])
    parser.add_argument('--n_ctx', type=int, default=-1)
    parser.add_argument('--template_set', type=str, default='imaging', choices=['imaging', 'attribute'])
    parser.add_argument('--attn_div_weight', type=float, default=0.05)
    parser.add_argument('--attn_ent_weight', type=float, default=0.05)
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
