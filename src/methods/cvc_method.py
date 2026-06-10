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
from src.models.motifs import build_motifs, generate_object_pairs
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

        # Store for access in forward
        self._cvc_head = cvc_head
        self._backbone = backbone

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
        """Run Motifs backbone to get context-encoded object features."""
        sort_idx = self._backbone._get_motif_order(labels, boxes)
        unsort_idx = torch.argsort(sort_idx)

        obj_feats = self._backbone.obj_encoder(visual_feats[sort_idx], labels[sort_idx])
        obj_feats = self._backbone.obj_context(obj_feats.unsqueeze(0)).squeeze(0)
        obj_feats = self._backbone.obj_post(obj_feats)
        return obj_feats[unsort_idx]

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

                # Pair features from backbone
                pair_feats = self._backbone.pair_gen(obj_feats, box, pairs)

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
                        batched[k] = torch.cat([o[k] for o in all_outputs], dim=0)

            out = {"outputs": batched}

            if is_training and self.criterion is not None:
                loss_dict = self.criterion(batched, targets)
                out["loss_dict"] = loss_dict
                out["loss"] = loss_dict["loss_total"]

            return out

        return {"outputs": {}}
