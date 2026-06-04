# ssgoos/tasks/video_ssgoos.py
"""
Video-based Scene Graph Generation Task (future).

Handles temporal dimension in inputs: [B, T, C, H, W].
This is a skeleton implementation for future video SGG support.
"""

from ssgoos.registry import TASK_REGISTRY
from ssgoos.tasks.image_sgg import ImageSceneGraphTask


@TASK_REGISTRY.register('video_sgg')
class VideoSceneGraphTask(ImageSceneGraphTask):
    """
    Video SGG task: extends ImageSceneGraphTask with temporal handling.

    Input: [B, T, C, H, W] video clips
    Output: Temporal relation triplets with optional timestamp annotations.

    Future work:
      - Implement temporal-aware metrics (e.g., temporal consistency)
      - Support Action Genome, VidOR, and other video SGG datasets
      - Add temporal post-processing (tracklet smoothing, etc.)
    """

    def training_step(self, batch, batch_idx):
        """Video training step with temporal dimension."""
        images, targets = self._unpack_batch(batch)
        # images: [B, T, C, H, W] — temporal dimension preserved
        result = self.model_adapter(images, targets)
        loss = result['total_loss']

        loss_dict = result.get('loss_dict', {})
        for name, value in loss_dict.items():
            self.log(f'train_{name}', value, on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _unpack_batch(self, batch):
        """Handle video batch format: (frames, targets) or dict."""
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            return batch.get('frames', batch.get('images')), batch.get('targets')
        return batch, None
