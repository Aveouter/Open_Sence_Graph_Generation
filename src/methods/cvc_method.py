"""
CVC method wrapper for PyTorch Lightning.

CVC (Compositionally Verified Concept) replaces the final predicate classifier
of Motifs with a composition-aware relation head. It can be used on top of any
object encoder (Motifs, VCTree, or backbone).

The Motifs backbone provides object context features → PairFeatureGenerator
produces pair features → CVC head predicts predicates with debiasing.
"""
import torch
import torch.nn as nn
from .motifs_method import Motifs_Method
from src.models.motifs import build_motifs, generate_object_pairs, PairFeatureGenerator
from src.models.cvc import build_cvc, CVCLoss


class CVC_Method(Motifs_Method):
    """CVC Lightning method.

    Replaces Motifs' predicate classifier with CVC composition-aware head.

    Config keys:
        cvc_bias_lambda:  λ for inference bias subtraction (default 0.5).
        cvc_alpha:        α for KL alignment loss (default 0.1).
        cvc_beta:         β for prototype diversity loss (default 0.05).
        cvc_gamma:        γ for adversarial composition removal (default 0.1).
        cvc_temperature:  Temperature for softmax in KL (default 2.0).
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        """Build Motifs backbone + CVC relation head."""
        backbone = build_motifs(self.hparams)
        cvc_head = build_cvc(self.hparams)

        # Store the Motifs backbone for context extraction. The CVC head is
        # returned as ``self.model`` by Base_method.
        self._backbone = backbone

        # Build a pair feature generator compatible with the backbone's
        # hidden_dim.  The backbone's extract_object_context() returns
        # [N, hidden_dim] edge-context features that pair_gen consumes.
        hidden_dim = getattr(self.hparams, "hidden_dim", 512)
        cvc_head.pair_gen = PairFeatureGenerator(hidden_dim, hidden_dim)

        # Return CVC head as the primary model; backbone is accessed directly
        return cvc_head

    def _build_criterion(self, **args):
        return CVCLoss(
            alpha=args.get('cvc_alpha', 0.1),
            beta=args.get('cvc_beta', 0.05),
            gamma=args.get('cvc_gamma', 0.1),
            temperature=args.get('cvc_temperature', 2.0),
        )

    # ---------- forward ----------

    def _encode_objects(self, visual_feats, boxes, labels):
        """Run Motifs backbone to get context-encoded object features.

        Uses the new SGB/Kaihua backbone's public ``extract_object_context``
        entry-point instead of reaching into removed internal attributes.
        """
        return self._backbone.extract_object_context(visual_feats, boxes, labels)

    def forward(self, images, targets=None, **kwargs):
        is_training = targets is not None

        if targets is not None:
            all_outputs = []
            boxes_list = [t["boxes"] for t in targets]
            labels_list = [t["labels"] for t in targets]
            image_sizes = [t.get("orig_size", t.get("size")) for t in targets]

            # Extract visual features (backbone or embedding)
            visual_feats_list = self._extract_visual_features(
                images, boxes_list, labels_list, image_sizes)

            for i, (box, lab, vis) in enumerate(zip(boxes_list, labels_list, visual_feats_list)):
                device = box.device
                N = box.size(0)

                # Generate all directed pairs
                pairs = generate_object_pairs(N, device)
                if pairs.numel() == 0:
                    all_outputs.append({
                        "pred_logits": box.new_zeros(0, self.hparams.rel_nums),
                        "pair_indices": pairs,
                        "sub_boxes": box.new_zeros(0, 4),
                        "obj_boxes": box.new_zeros(0, 4),
                    })
                    continue

                # Backbone: object context encoding
                obj_feats = self._encode_objects(vis, box, lab)

                # Pair features from backbone context
                pair_feats = self.model.pair_gen(obj_feats, box, pairs)

                # CVC head: predicate prediction
                s_labels = lab[pairs[:, 0]]
                o_labels = lab[pairs[:, 1]]

                cvc_out = self.model(
                    pair_feats, s_labels, o_labels,
                    return_all=is_training,
                )

                cvc_out["pair_indices"] = pairs
                cvc_out["sub_boxes"] = box[pairs[:, 0]]
                cvc_out["obj_boxes"] = box[pairs[:, 1]]
                all_outputs.append(cvc_out)

            # Batch
            batched = {
                "pred_logits": [o["pred_logits"] for o in all_outputs],
                "pair_indices": [o["pair_indices"] for o in all_outputs],
                "sub_boxes": [o["sub_boxes"] for o in all_outputs],
                "obj_boxes": [o["obj_boxes"] for o in all_outputs],
            }
            if is_training:
                for k in ["bias_logits", "z_v", "z_b"]:
                    if k in all_outputs[0]:
                        batched[k] = [o[k] for o in all_outputs]

            out = {"outputs": batched}

            # CVC uses VG predicate ids 1..50 directly as CE targets,
            # meaning index 0 is background/unused. The evaluator default
            # assumes background is LAST, so we must override it here.
            out["outputs"]["predicate_bg_index"] = "first"
            out["outputs"]["relation_softmax_scope"] = "all"

            if is_training and self.criterion is not None:
                loss_dict = self.criterion(batched, targets)
                out["loss_dict"] = loss_dict
                out["loss"] = loss_dict["loss_total"]

            return out

        return {"outputs": {}}
