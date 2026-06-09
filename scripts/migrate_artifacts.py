#!/usr/bin/env python
"""One-time migration from the old scattered artifact layout to the new
unified ``outputs/`` structure.

Usage::

    python scripts/migrate_artifacts.py              # dry-run (default)
    python scripts/migrate_artifacts.py --copy        # copy to new structure
    python scripts/migrate_artifacts.py --move --confirm  # move + delete old

The script generates ``migration_report.md`` in the current directory.
"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Mapping logic
# ---------------------------------------------------------------------------

# Try to infer the method from model_param.json inside a results/ subdirectory.
def _infer_method_from_results_dir(src_dir: str) -> str:
    param_file = osp.join(src_dir, 'model_param.json')
    if osp.isfile(param_file):
        try:
            with open(param_file) as f:
                cfg = json.load(f)
            method = cfg.get('method', '').lower()
            if method:
                return method
        except Exception:
            pass
    return 'unknown'


def _build_migration_plan() -> List[Dict]:
    """Scan old directories and return a list of migration actions.

    Each action dict has keys: old_path, new_path, action (copy/move), reason.
    """
    plan: List[Dict] = []
    root = osp.dirname(osp.dirname(osp.abspath(__file__)))  # OpenSGG/

    # ---- checkpoints/ → outputs/pretrained/ ----
    old_ckpt = osp.join(root, 'checkpoints')
    if osp.isdir(old_ckpt):
        for sub in os.listdir(old_ckpt):
            sub_path = osp.join(old_ckpt, sub)
            if not osp.isdir(sub_path):
                # Top-level files stay
                plan.append({
                    'old_path': sub_path,
                    'new_path': osp.join(root, 'outputs', 'pretrained', sub),
                    'reason': 'pretrained weight',
                })
                continue
            for fname in os.listdir(sub_path):
                src = osp.join(sub_path, fname)
                if osp.isfile(src):
                    dst = osp.join(root, 'outputs', 'pretrained', sub, fname)
                    plan.append({
                        'old_path': src,
                        'new_path': dst,
                        'reason': f'pretrained weight ({sub})',
                    })

    # ---- results/<ex_name>/ → outputs/runs/<method>/<ex_name>/ ----
    old_results = osp.join(root, 'results')
    if osp.isdir(old_results):
        for ex_name in os.listdir(old_results):
            src_dir = osp.join(old_results, ex_name)
            if not osp.isdir(src_dir):
                # Flat files like flowsg_experiment_*.json
                plan.append({
                    'old_path': src_dir,
                    'new_path': osp.join(root, 'outputs', 'runs', 'misc', ex_name),
                    'reason': 'flat result file (method unknown)',
                })
                continue

            method = _infer_method_from_results_dir(src_dir)
            for dirpath, _, filenames in os.walk(src_dir):
                for fname in filenames:
                    src = osp.join(dirpath, fname)
                    rel = osp.relpath(src, src_dir)
                    dst = osp.join(root, 'outputs', 'runs', method, ex_name, rel)
                    plan.append({
                        'old_path': src,
                        'new_path': dst,
                        'reason': f'experiment result ({method})',
                    })

    # ---- lightning_logs/ → best-effort merge ----
    old_lightning = osp.join(root, 'lightning_logs')
    if osp.isdir(old_lightning):
        for ver in os.listdir(old_lightning):
            ver_dir = osp.join(old_lightning, ver)
            if not osp.isdir(ver_dir):
                continue
            for fname in os.listdir(ver_dir):
                src = osp.join(ver_dir, fname)
                if osp.isfile(src):
                    # Can't reliably map to a specific run; place in a _migrated_lightning dir
                    dst = osp.join(root, 'outputs', 'runs', '_migrated_lightning', ver, fname)
                    plan.append({
                        'old_path': src,
                        'new_path': dst,
                        'reason': 'lightning log (version not mapped to run)',
                    })

    # ---- logs/ ----
    old_logs = osp.join(root, 'logs')
    if osp.isdir(old_logs):
        for fname in os.listdir(old_logs):
            src = osp.join(old_logs, fname)
            if osp.isfile(src):
                dst = osp.join(root, 'outputs', 'runs', '_migrated_logs', fname)
                plan.append({
                    'old_path': src,
                    'new_path': dst,
                    'reason': 'legacy log file',
                })

    # ---- scripts/output/ ----
    scripts_out = osp.join(root, 'scripts', 'output')
    if osp.isdir(scripts_out):
        for fname in os.listdir(scripts_out):
            src = osp.join(scripts_out, fname)
            if osp.isfile(src):
                dst = osp.join(root, 'outputs', 'runs', '_migrated_scripts', fname)
                plan.append({
                    'old_path': src,
                    'new_path': dst,
                    'reason': 'legacy script output',
                })

    return plan


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(plan: List[Dict], mode: str, results: List[Dict]) -> str:
    """Generate ``migration_report.md``."""
    lines = [
        '# Migration Report',
        '',
        f'Generated: {datetime.now().isoformat()}',
        f'Mode: {mode}',
        f'Total items: {len(plan)}',
        '',
        '| # | Old Path | New Path | Status |',
        '|---|----------|----------|--------|',
    ]
    for i, item in enumerate(results, 1):
        status = item.get('status', 'pending')
        old = osp.relpath(item['old_path'])
        new = osp.relpath(item['new_path'])
        lines.append(f'| {i} | `{old}` | `{new}` | {status} |')

    ok = sum(1 for r in results if r.get('status') == 'ok')
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    errors = sum(1 for r in results if r.get('status') == 'error')

    lines.extend([
        '',
        '## Summary',
        '',
        f'- **OK**: {ok}',
        f'- **Skipped**: {skipped}',
        f'- **Errors**: {errors}',
    ])
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Migrate old artifacts to outputs/')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files to new locations (leave originals).')
    parser.add_argument('--move', action='store_true',
                        help='Move files to new locations.')
    parser.add_argument('--confirm', action='store_true',
                        help='Required with --move to confirm deletion of originals.')
    args = parser.parse_args()

    # Resolve mode
    if args.move and not args.confirm:
        print('ERROR: --move requires --confirm', file=sys.stderr)
        sys.exit(1)
    if args.copy and args.move:
        print('ERROR: --copy and --move are mutually exclusive', file=sys.stderr)
        sys.exit(1)

    if args.move:
        mode = 'move'
    elif args.copy:
        mode = 'copy'
    else:
        mode = 'dry-run'

    print(f'Migration mode: {mode}')
    print('Scanning old directories...')

    plan = _build_migration_plan()
    if not plan:
        print('Nothing to migrate.')
        return

    print(f'Found {len(plan)} items to migrate.\n')

    results = []
    for item in plan:
        old = item['old_path']
        new = item['new_path']
        entry = dict(item)

        if mode == 'dry-run':
            print(f'  [DRY-RUN] {osp.relpath(old)} → {osp.relpath(new)}')
            entry['status'] = 'dry-run'
        else:
            # Create destination directory
            os.makedirs(osp.dirname(new), exist_ok=True)

            try:
                if mode == 'copy':
                    if osp.isdir(old):
                        if osp.exists(new):
                            print(f'  [SKIP] {osp.relpath(old)} (dir already exists)')
                            entry['status'] = 'skipped'
                        else:
                            shutil.copytree(old, new)
                            print(f'  [COPY] {osp.relpath(old)} → {osp.relpath(new)}')
                            entry['status'] = 'ok'
                    else:
                        shutil.copy2(old, new)
                        print(f'  [COPY] {osp.relpath(old)} → {osp.relpath(new)}')
                        entry['status'] = 'ok'
                elif mode == 'move':
                    shutil.move(old, new)
                    print(f'  [MOVE] {osp.relpath(old)} → {osp.relpath(new)}')
                    entry['status'] = 'ok'
            except Exception as e:
                print(f'  [ERROR] {osp.relpath(old)}: {e}')
                entry['status'] = f'error: {e}'

        results.append(entry)

    # Remove empty old dirs after move
    if mode == 'move':
        print('\nCleaning up empty old directories...')
        old_roots = [
            osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), d)
            for d in ['results', 'checkpoints', 'lightning_logs', 'logs', 'scripts/output']
        ]
        for old_root in old_roots:
            if not osp.isdir(old_root):
                continue
            try:
                remaining = []
                for dirpath, dirnames, filenames in os.walk(old_root, topdown=False):
                    if dirpath != old_root and not os.listdir(dirpath):
                        os.rmdir(dirpath)
                        print(f'  [RMDIR] {osp.relpath(dirpath)}')
                    else:
                        remaining.append(dirpath)
                # If the root itself is empty, remove it
                if not os.listdir(old_root):
                    os.rmdir(old_root)
                    print(f'  [RMDIR] {osp.relpath(old_root)}')
            except Exception as e:
                print(f'  [WARN] Could not clean {old_root}: {e}')

    # Write report
    report = generate_report(plan, mode, results)
    with open('migration_report.md', 'w') as f:
        f.write(report)
    print(f'\nMigration report saved to migration_report.md')


if __name__ == '__main__':
    main()
