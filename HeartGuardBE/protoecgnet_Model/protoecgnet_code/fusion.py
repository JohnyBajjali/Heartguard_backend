import os
import torch
import pandas as pd
import torch.nn as nn
from pathlib import Path
from .proto_models1D import ProtoECGNet1D
from .proto_models2D import ProtoECGNet2D

class FusionProtoClassifier(nn.Module):
    def __init__(self, model_1d, model_2d_partial, model_2d_global, num_classes=71):
        super().__init__()
        self.model1d = model_1d.eval()
        self.model2d_partial = model_2d_partial.eval()
        self.model2d_global = model_2d_global.eval()

        for submodel in [self.model1d, self.model2d_partial, self.model2d_global]:
            for param in submodel.parameters():
                param.requires_grad = False

        self.num_classes = num_classes

        self.num_prototypes_1d = self.model1d.num_prototypes
        self.num_prototypes_2d_partial = self.model2d_partial.num_prototypes
        self.num_prototypes_2d_global = self.model2d_global.num_prototypes
        self.total_prototypes = self.num_prototypes_1d + self.num_prototypes_2d_partial + self.num_prototypes_2d_global

        self.classifier = nn.Linear(self.total_prototypes, num_classes, bias=True)

    def forward(self, x):
        x1d = x.squeeze(1)  # Convert [N, 1, 12, 1000] → [N, 12, 1000] for 1D branch only
        _, _, sim1d = self.model1d(x1d)
        _, _, sim2d_partial = self.model2d_partial(x)
        _, _, sim2d_global = self.model2d_global(x)

        sims = torch.cat([sim1d, sim2d_partial, sim2d_global], dim=1)
        logits = self.classifier(sims)
        return logits

SCP_GROUP_PATH = Path(__file__).resolve().parent.parent / "assets" / "scp_statementsRegrouped2.csv"

def load_fusion_label_mappings():
    label_df = pd.read_csv(SCP_GROUP_PATH, index_col=0)
    assert "prototype_category" in label_df.columns

    labels_1 = label_df[label_df["prototype_category"] == 1].index.tolist()
    labels_3 = label_df[label_df["prototype_category"] == 3].index.tolist()
    labels_4 = label_df[label_df["prototype_category"] == 4].index.tolist()

    all_labels = label_df.index.tolist()

    return {
        "1": labels_1,
        "3": labels_3,
        "4": labels_4,
        "all": all_labels
    }

