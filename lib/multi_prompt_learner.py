"""Multi-prompt learner (plot-pp style) for FSL semantic routing."""

import torch
import torch.nn as nn
import clip

DEFAULT_TEMPLATES = [
    "a photo of a",
    "this is a photo of a",
    "this is a picture of a",
    "a close-up photo of a",
    "a cropped photo of a",
    "a texture of a",
    "a good photo of a",
    "a bad photo of a",
]

# Four orthogonal imaging/perception axes (cross-class, low paraphrase overlap).
# P0 baseline | P1 scale/detail | P2 composition/crop | P3 lighting
SEMANTIC4_TEMPLATES = [
    "a photo of a",
    "a close-up photo of a",
    "a cropped photo of a",
    "a photo of a in natural light",
]

# Category-relevant roles (cross-class). Used by train_semantic4_adapt.py.
# P0 identity | P1 parts | P2 appearance | P3 context
ATTRIBUTE4_TEMPLATES = [
    "a photo of a",
    "a photo showing the distinctive parts of a",
    "a photo showing the color and texture of a",
    "a photo of a in its typical surroundings",
]
ATTRIBUTE4_ROLES = ['identity', 'parts', 'appearance', 'context']
SEMANTIC4_ROLES = ['global baseline', 'scale / detail', 'composition / crop', 'lighting']

# Extra axes when num_prompts > 4 under semantic4 init (still cross-class).
SEMANTIC4_EXTRA_TEMPLATES = [
    "a texture of a",
    "a photo of a outdoors",
    "a photo of a on a plain background",
    "a good photo of a",
]


def get_prompt_templates(dataset_name, num_prompts, ctx_init='templates'):
    if ctx_init == 'single':
        return ["a photo of a"] * num_prompts
    if ctx_init == 'semantic4':
        templates = list(SEMANTIC4_TEMPLATES) + list(SEMANTIC4_EXTRA_TEMPLATES)
        if num_prompts <= len(templates):
            return templates[:num_prompts]
        extra = [templates[i % len(templates)] for i in range(len(templates), num_prompts)]
        return templates + extra
    if ctx_init == 'attribute4':
        templates = list(ATTRIBUTE4_TEMPLATES)
        if num_prompts <= len(templates):
            return templates[:num_prompts]
        extra = [templates[i % len(templates)] for i in range(len(templates), num_prompts)]
        return templates + extra
    templates = list(DEFAULT_TEMPLATES)
    if dataset_name in ('CIFAR-FS', 'FC100'):
        templates = [
            "a photo of a",
            "this is a photo of a",
            "a cropped photo of a",
            "a texture of a",
        ] + templates
    elif dataset_name in ('miniImageNet', 'tieredImageNet'):
        templates = [
            "a photo of a",
            "this is a photo of a",
            "this is a picture of a",
            "one picture of a",
        ] + templates
    if num_prompts <= len(templates):
        return templates[:num_prompts]
    extra = [templates[i % len(templates)] for i in range(len(templates), num_prompts)]
    return templates + extra


def _clip_device(clip_model):
    return next(clip_model.parameters()).device


class MultiPromptLearner(nn.Module):
    """N learnable text prompt templates shared across all classes."""

    def __init__(self, classnames, clip_model, num_prompts=4, ctx_init='templates',
                 dataset_name='CIFAR-FS', n_ctx=-1, template_list=None):
        super().__init__()
        self.n_prompts = num_prompts
        self.n_cls = len(classnames)
        self.fixed_n_ctx = n_ctx
        dtype = clip_model.dtype
        device = _clip_device(clip_model)
        if template_list is not None:
            explicit_templates = True
            template_list = list(template_list)
            if len(template_list) < num_prompts:
                raise ValueError(
                    f'template_list has {len(template_list)} entries, need {num_prompts}'
                )
            template_list = template_list[:num_prompts]
        else:
            explicit_templates = False
            template_list = get_prompt_templates(dataset_name, num_prompts, ctx_init)
        classnames = [name.replace("_", " ") for name in classnames]

        ctx_params = []
        prefix_buffers = []
        suffix_buffers = []
        tokenized_rows = []
        n_ctx_list = []

        for i in range(num_prompts):
            prefix_text = template_list[i].replace("_", " ")
            init_prompt = clip.tokenize(prefix_text).to(device)
            with torch.no_grad():
                init_emb = clip_model.token_embedding(init_prompt).type(dtype)

            eot_pos = init_prompt[0].argmax().item()
            natural_n_ctx = min(len(prefix_text.split()), max(1, eot_pos - 1))
            if n_ctx is not None and n_ctx > 0:
                cur_n_ctx = n_ctx
            else:
                cur_n_ctx = natural_n_ctx

            init_ctx = init_emb[0, 1: 1 + natural_n_ctx, :].clone()
            ctx_i = self._pad_ctx(init_ctx, cur_n_ctx)
            n_ctx_list.append(cur_n_ctx)

            prompts = [prefix_text + " " + name + "." for name in classnames]
            tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            with torch.no_grad():
                emb = clip_model.token_embedding(tokenized).type(dtype)

            prefix_buffers.append(emb[:, :1, :])
            suffix_buffers.append(emb[:, 1 + natural_n_ctx:, :])
            ctx_params.append(ctx_i)
            tokenized_rows.append(tokenized)

        self.n_ctx_list = n_ctx_list
        self.n_ctx = max(n_ctx_list)
        stacked = torch.stack([
            self._pad_ctx(c, self.n_ctx).detach().clone() for c in ctx_params
        ], dim=0)
        self.ctx = nn.Parameter(stacked)

        for i, prefix in enumerate(prefix_buffers):
            self.register_buffer(f'token_prefix_{i}', prefix)
        for i, suffix in enumerate(suffix_buffers):
            self.register_buffer(f'token_suffix_{i}', suffix)
        self.register_buffer('tokenized_prompts', torch.cat(tokenized_rows, dim=0))

        ctx_mode = f'fixed={n_ctx}' if (n_ctx is not None and n_ctx > 0) else 'auto'
        init_tag = 'template_list' if explicit_templates else ctx_init
        print(f'[MultiPromptLearner] N={num_prompts}, n_ctx_list={n_ctx_list}, n_ctx_mode={ctx_mode}, init={init_tag}')
        for i, t in enumerate(template_list[:num_prompts]):
            print(f'  prompt[{i}]: "{t}"')

    @staticmethod
    def _pad_ctx(ctx, target_len):
        if ctx.shape[0] == target_len:
            return ctx
        if ctx.shape[0] < target_len:
            pad = ctx[-1:].expand(target_len - ctx.shape[0], -1)
            return torch.cat([ctx, pad], dim=0)
        return ctx[:target_len]

    def _ctx_for_prompt(self, p):
        n_ctx = self.n_ctx_list[p]
        return self.ctx[p, :n_ctx]

    def forward(self):
        all_prompts = []
        for p in range(self.n_prompts):
            n_ctx = self.n_ctx_list[p]
            ctx = self._ctx_for_prompt(p).unsqueeze(0).expand(self.n_cls, -1, -1)
            prefix = getattr(self, f'token_prefix_{p}')
            suffix = getattr(self, f'token_suffix_{p}')
            all_prompts.append(torch.cat([prefix, ctx, suffix], dim=1))
        prompts = torch.cat(all_prompts, dim=0)
        return prompts, self.tokenized_prompts

    def class_prompt_indices(self, class_indices):
        B = class_indices.shape[0]
        device = class_indices.device
        p_idx = torch.arange(self.n_prompts, device=device).view(1, self.n_prompts).expand(B, self.n_prompts)
        flat_idx = p_idx * self.n_cls + class_indices.view(B, 1).expand(B, self.n_prompts)
        return flat_idx
