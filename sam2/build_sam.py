# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from importlib import import_module

import torch
import yaml
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
    # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
    # This typically happens because the user is running Python from the parent directory
    # that contains the sam2 repo they cloned.
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor_direct(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    """
    Hydra-free version:
      - Reads a YAML config file (path-like) into a Python dict.
      - Applies "override" strings of the form "++a.b.c=value" by updating the dict.
      - Instantiates SAM2VideoPredictor or SAM2VideoPredictorVOS directly.
      - Loads checkpoint, moves to device, sets eval/train mode.

    Notes:
      * This function assumes the YAML has a top-level key "model" with kwargs
        compatible with the predictor constructor.
      * The '++' prefix in overrides is ignored here; it's accepted for compatibility.
    """

    # Local helpers
    def _coerce_scalar(s: str):
        """Convert string to bool/int/float/None when possible; else return as-is."""
        sl = s.strip().lower()
        if sl in ("true", "false"):
            return sl == "true"
        if sl in ("none", "null"):
            return None
        try:
            if "." in sl or "e" in sl:
                return float(sl)
            return int(sl)
        except ValueError:
            return s  # keep original (may be non-scalar string)

    def _set_nested(d: dict, dotted_key: str, value):
        """Set a nested key 'a.b.c' in dict d to value, creating levels as needed."""
        cur = d
        keys = dotted_key.split(".")
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value

    def _apply_overrides(cfg: dict, overrides: list):
        """Apply a list of '++a.b.c=value' strings onto cfg (in-place)."""
        for ov in overrides or []:
            ov = ov.strip()
            if not ov:
                continue
            # Accept optional leading "++" or "+" used by Hydra
            while ov.startswith("+"):
                ov = ov[1:]
            if "=" not in ov:
                raise ValueError(f"Invalid override (missing '='): {ov}")
            lhs, rhs = ov.split("=", 1)
            _set_nested(cfg, lhs.strip(), _coerce_scalar(rhs))

    # Resolve config
    if isinstance(config_file, (str, os.PathLike)):
        cfg_path = os.fspath(config_file)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    elif isinstance(config_file, dict):
        # Allow passing a dict directly
        cfg = dict(config_file)
    else:
        raise TypeError("config_file must be a path-like or a dict")

    # Build the list of overrides just like the Hydra version
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",
        ]

    extra = list(hydra_overrides_extra or [])
    if apply_postprocessing:
        extra += [
            # dynamic multimask fallback
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # match interacted-frame masks in memory encoder
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # small hole filling before upsampling
            "++model.fill_hole_area=8",
        ]

    # Apply base + extra overrides on the plain dict
    _apply_overrides(cfg, hydra_overrides + extra)

    # Instantiate the predictor class directly
    # Determine target class; fallback to cfg["model"].get("_target_")
    target = None
    if "model" in cfg and isinstance(cfg["model"], dict):
        target = cfg["model"].get("_target_")

    if not target:
        # Default based on vos_optimized
        target = (
            "sam2.sam2_video_predictor.SAM2VideoPredictorVOS"
            if vos_optimized
            else "sam2.sam2_video_predictor.SAM2VideoPredictor"
        )

    if "." not in target:
        raise ValueError(f"Invalid _target_ path: {target}")
    mod_path, cls_name = target.rsplit(".", 1)
    PredictorCls = getattr(import_module(mod_path), cls_name)

    # Extract kwargs for constructor (everything under model except _target_)
    model_kwargs = {}
    if "model" in cfg and isinstance(cfg["model"], dict):
        model_kwargs = {k: v for k, v in cfg["model"].items() if k != "_target_"}

    # Allow kwargs from function call to override YAML (explicit > config)
    if kwargs:
        model_kwargs.update(kwargs)

    model = PredictorCls(**model_kwargs)

    # Load checkpoint if provided
    if ckpt_path:
        # _load_checkpoint is assumed to exist in the caller's module scope
        try:
            _load_checkpoint  # noqa: F821  # referenced before assignment if missing
        except NameError as e:
            raise NameError(
                "_load_checkpoint(model, ckpt_path) is not defined in scope."
            ) from e
        _load_checkpoint(model, ckpt_path)

    # Device & mode
    if hasattr(model, "to"):
        model = model.to(device)
    if mode == "eval" and hasattr(model, "eval"):
        model.eval()
    elif mode == "train" and hasattr(model, "train"):
        model.train()

    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

