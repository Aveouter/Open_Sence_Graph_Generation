import json
import logging
import os
import os.path as osp
import shutil

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from .main_utils import check_dir, collect_env, output_namespace, print_log


class SetupCallback(Callback):
    """Creates run directories and initialises logging on ``on_fit_start``."""

    def __init__(self, prefix, setup_time, paths, args, method_info, argv_content=None):
        super().__init__()
        self.prefix = prefix
        self.setup_time = setup_time
        self.paths = paths            # dict from resolve_output_paths()
        self.args = args
        self.config = args.__dict__
        self.argv_content = argv_content
        self.method_info = method_info

    def on_fit_start(self, trainer, pl_module):
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'

        if trainer.global_rank == 0:
            run_dir = self.paths['run_dir']
            ckpt_dir = self.paths['ckpt_dir']
            config_dir = self.paths['config_dir']
            log_file = self.paths['log_file']
            config_file = self.paths['config_file']
            args_file = self.paths['args_file']

            # ensure directories
            check_dir(run_dir)
            check_dir(ckpt_dir)
            check_dir(config_dir)

            # setup logging
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(
                level=logging.INFO,
                filename=log_file,
                filemode='a',
                format='%(asctime)s - %(message)s',
            )

            # print env info
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

            # save config files
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            with open(args_file, 'w') as f:
                json.dump({'argv': self.argv_content} if self.argv_content else {}, f, indent=2)

            print_log(output_namespace(self.args))
            if self.method_info is not None:
                info, flops, fps, dash_line = self.method_info
                print_log('Model info:\n' + info + '\n' + flops + '\n' + fps + dash_line)


class EpochEndCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        self.avg_train_loss = trainer.callback_metrics.get('train_loss')

    def on_validation_epoch_end(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        avg_val_loss = trainer.callback_metrics.get('val_loss')

        if hasattr(self, 'avg_train_loss'):
            print_log(
                f"Epoch {trainer.current_epoch}: "
                f"Lr: {lr:.7f} | Train Loss: {self.avg_train_loss:.7f} | "
                f"Vali Loss: {avg_val_loss:.7f}"
            )


class BestCheckpointCallback(ModelCheckpoint):
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))
