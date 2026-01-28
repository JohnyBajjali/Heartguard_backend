# HeartGuardBE/ml/predictor.py
import json
from pathlib import Path
import numpy as np
import torch

from .protoecgnet_code.proto_models1D import ProtoECGNet1D
from .protoecgnet_code.proto_models2D import ProtoECGNet2D
from .protoecgnet_code.fusion import FusionProtoClassifier
from .protoecgnet_code.fusion import load_fusion_label_mappings


BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_labels():
    return load_fusion_label_mappings()

class FusionPredictor:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.label_map = load_labels()              
        self.fusion_labels = self.label_map["all"]

        self._build_and_load()

    def _torch_load_ckpt(self, path: Path):
        # compatible with older/newer torch
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    def _build_and_load(self):
        num_classes1 = len(self.label_map["1"])
        num_classes3 = len(self.label_map["3"])
        num_classes4 = len(self.label_map["4"])

        FUSION_WEIGHTS1 = CKPT_DIR / "1D_rhythm_train_fixed" / "last.ckpt"
        FUSION_WEIGHTS3 = CKPT_DIR / "2D_morphology_train1" / "last.ckpt"
        FUSION_WEIGHTS4 = CKPT_DIR / "2D_global_train" / "last.ckpt"
        FUSION_CKPT     = CKPT_DIR / "Fusion_Classifier" / "last.ckpt"

        # build submodels exactly like training 
        proto_act = "log"
        latent_space_type = "arc"
        add_on_layers_type = "linear"
        class_specific = True
        last_layer_connection_weight = 1.0
        m_margin = 0.05
        dropout = 0.35

        self.model1d = ProtoECGNet1D(
            num_classes=num_classes1,
            single_class_prototype_per_class=5,
            joint_prototypes_per_border=0,
            proto_dim=512,
            backbone="resnet1d18",
            prototype_activation_function=proto_act,
            latent_space_type=latent_space_type,
            add_on_layers_type=add_on_layers_type,
            class_specific=class_specific,
            last_layer_connection_weight=last_layer_connection_weight,
            m=m_margin,
            dropout=dropout,
            custom_groups=True,
            label_set="1",
            pretrained_weights=str(FUSION_WEIGHTS1),
        )

        self.model2d_partial = ProtoECGNet2D(
            num_classes=num_classes3,
            single_class_prototype_per_class=18,
            joint_prototypes_per_border=0,
            proto_dim=512,
            proto_time_len=3,
            backbone="resnet18",
            prototype_activation_function=proto_act,
            latent_space_type=latent_space_type,
            add_on_layers_type=add_on_layers_type,
            class_specific=class_specific,
            last_layer_connection_weight=last_layer_connection_weight,
            m=m_margin,
            dropout=dropout,
            custom_groups=True,
            label_set="3",
            pretrained_weights=str(FUSION_WEIGHTS3),
        )

        self.model2d_global = ProtoECGNet2D(
            num_classes=num_classes4,
            single_class_prototype_per_class=3,
            joint_prototypes_per_border=0,
            proto_dim=512,
            proto_time_len=32,
            backbone="resnet18",
            prototype_activation_function=proto_act,
            latent_space_type=latent_space_type,
            add_on_layers_type=add_on_layers_type,
            class_specific=class_specific,
            last_layer_connection_weight=last_layer_connection_weight,
            m=m_margin,
            dropout=dropout,
            custom_groups=True,
            label_set="4",
            pretrained_weights=str(FUSION_WEIGHTS4),
        )

        for m in [self.model1d, self.model2d_partial, self.model2d_global]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        self.fusion = FusionProtoClassifier(self.model1d, self.model2d_partial, self.model2d_global, num_classes=71).to(DEVICE)
        self.fusion.eval()

        ckpt = self._torch_load_ckpt(FUSION_CKPT)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        # strip "model." prefix
        sd2 = {}
        for k, v in sd.items():
            sd2[k[len("model."):] if k.startswith("model.") else k] = v

        self.fusion.load_state_dict(sd2, strict=False)

    @torch.no_grad()
    def predict_from_X(self, X_1x1000x12: np.ndarray):
        # X: (1,1000,12)
        x1d = torch.from_numpy(X_1x1000x12).float().to(DEVICE).permute(0, 2, 1)  # (1,12,1000)
        x2d = x1d.unsqueeze(1)                                                   # (1,1,12,1000)

        logits = self.fusion(x2d)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()          # (71,)

        return probs
