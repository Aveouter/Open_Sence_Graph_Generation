"""The defaults and merge policy a training run is actually built from.

This module exists because those decisions were spread across the codebase
rather than owned anywhere:

* The metric list a run reports was derived inline inside ``train.py``'s
  ``if __name__ == "__main__"`` block, so nothing could import or test it --
  even though it decides which metric the checkpoint is selected on.
* The merge policy -- which CLI arguments survive a config-file load -- was
  written out seven times: twice in ``train.py``, once in the test that covers
  it, and four times across ``tools/analysis``, one of them inverted.

Stdlib-only by design.  The existing test for this area guards on torch and
therefore never runs in the dependency-free CI job; anything here has to be
readable there.
"""

from __future__ import annotations

#: The recall cut-offs every eval-mode metric list is built from.
METRIC_BUDGETS: tuple[int, ...] = (20, 50, 100)


def metrics_for_eval_mode(eval_mode: str) -> list[str]:
    """The metric names a run reports for ``eval_mode``.

    Moved out of ``train.py`` unchanged: recall first, then mean recall, at each
    budget in :data:`METRIC_BUDGETS`.
    """
    return [f"{eval_mode}_R@{k}" for k in METRIC_BUDGETS] + [
        f"{eval_mode}_mR@{k}" for k in METRIC_BUDGETS
    ]


#: Keys a config file may never overwrite.  ``method`` is protected because it is
#: how the config file was chosen: letting the file set it would let a mismatched
#: config silently redefine the run.
MERGE_ALWAYS_PROTECTED: tuple[str, ...] = ("method",)

#: Keys additionally protected unless ``--overwrite`` is set.  These are the ones
#: a sweep or a quick experiment is most likely to pass on the command line, so
#: silently losing them to a config file would be a quiet change of setting.
MERGE_PROTECTED_UNLESS_OVERWRITE: tuple[str, ...] = (
    "val_batch_size",
    "drop_path",
    "warmup_epoch",
)


def merge_exclude_keys(overwrite: bool) -> tuple[str, ...]:
    """The keys a config file may not overwrite, given the ``--overwrite`` flag.

    This is the one place the policy is spelled out.  It previously appeared in
    seven: twice in ``train.py``, once in the test covering it, and four times
    across ``tools/analysis`` -- one of those inverted, so it could drift from
    the rest without anyone noticing.
    """
    if overwrite:
        return MERGE_ALWAYS_PROTECTED
    return MERGE_ALWAYS_PROTECTED + MERGE_PROTECTED_UNLESS_OVERWRITE


def update_config(args, config, exclude_keys=None):
    """Update the args dict with a new config dict.

    CLI args (``args``) take priority: a config key is only applied when
    the corresponding CLI arg is missing (not in args) or explicitly None.
    Keys in ``exclude_keys`` are never overwritten by the config file.

    Moved here from ``utils.main_utils`` unchanged.  It is a pure dict merge and
    has nothing to do with cv2 or torch, but living in that module made it
    unimportable in the dependency-free CI job -- where the merge policy is
    exactly the kind of thing worth testing.
    """
    if exclude_keys is None:
        exclude_keys = []
    assert isinstance(args, dict) and isinstance(config, dict)
    for k in config.keys():
        if k in exclude_keys:
            continue  # keep command-line value, don't touch
        if k in args and args[k] is not None:
            if args[k] != config[k]:
                print(f'overwrite config key -- {k}: {config[k]} -> {args[k]}')
        else:
            args[k] = config[k]
    return args
