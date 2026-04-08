"""
model.py  –  SplitFed Neural Network Models
============================================
Implements the Learning Layer from Section II-B of the paper.

Architecture overview
---------------------
  SplitFedModel  (Teacher / main model)
    • Backbone  : torchvision.models.mobilenet_v3_small
    • Split     : dynamically chosen L_cut splits the backbone into
                  [client_layers | server_layers]
    • Forward   : two-phase
        Phase 1 – client_forward(x, l_cut) → smashed_data   (runs on drone)
        Phase 2 – server_forward(smashed_data, l_cut) → logits  (runs on server)
    • In this single-process simulation both phases run sequentially in the
      same Python call-stack; PyTorch autograd naturally propagates gradients
      through both halves.

  StudentModel  (Lightweight KD fallback)
    • A tiny 2-conv + 2-fc network that runs entirely on the drone.
    • Activated when CLF detects a stability violation (jamming fallback,
      Section II-B point 3).
    • Trained via Online Knowledge Distillation using the Teacher's soft labels.

Dataset assumption
------------------
  EuroSAT  –  64×64 RGB remote-sensing images, 10 classes.
  If EuroSAT is unavailable, mock random tensors are used (see main.py).
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES  = 10    # EuroSAT has 10 land-use classes
INPUT_SHAPE  = (3, 64, 64)   # C × H × W


# ---------------------------------------------------------------------------
# Helper: extract the ordered list of feature layers from MobileNetV3-small
# ---------------------------------------------------------------------------
def _get_mobilenet_layers(pretrained: bool = False) -> nn.ModuleList:
    """
    Return the `features` block of MobileNetV3-small as a ModuleList so we
    can index individual layers for dynamic splitting.

    MobileNetV3-small.features has 13 sub-modules (indices 0–12).
    We expose these directly so that an L_cut value selects exactly which
    layers execute on the drone (client side).
    """
    backbone = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    )
    # features is an nn.Sequential; wrap in ModuleList for index access
    return nn.ModuleList(list(backbone.features.children()))


# ---------------------------------------------------------------------------
# SplitFed Main Model
# ---------------------------------------------------------------------------
class SplitFedModel(nn.Module):
    """
    Dynamic split model built on MobileNetV3-small.

    The backbone's `features` block contains L_MAX=13 layers (0..12).
    L_cut ∈ {1, …, L_MAX-1} partitions them:
        Client layers  : features[0 .. L_cut-1]    (runs on drone)
        Server layers  : features[L_cut .. L_MAX-1] (runs on server / neighbour)

    After `features`, a small classifier head:
        AdaptiveAvgPool2d → Flatten → Linear(hidden, NUM_CLASSES)

    Usage in SplitFed mode (two-phase call, Eq. 2 / Section II-B)
    --------------------------------------------------------------
        smashed = model.client_forward(x, l_cut)   # Phase 1
        logits  = model.server_forward(smashed, l_cut)  # Phase 2

    Usage in full (fallback) mode
    ------------------------------
        logits  = model(x, l_cut=13)   # passes all layers; identical to normal forward
    """

    L_MAX = 13   # number of feature sub-modules in MobileNetV3-small

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes

        # ---- Backbone feature layers ----------------------------------------
        self.feature_layers = _get_mobilenet_layers(pretrained)

        # ---- Classifier head -------------------------------------------------
        # MobileNetV3-small features output 576 channels at 2×2 for 64×64 input
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.flatten    = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    # ------------------------------------------------------------------
    # Phase 1 – CLIENT side (runs on drone)
    # ------------------------------------------------------------------
    def client_forward(self, x: torch.Tensor, l_cut: int) -> torch.Tensor:
        """
        Execute feature layers [0 .. l_cut-1] on the CLIENT (drone).

        Parameters
        ----------
        x     : input image tensor  (B, C, H, W)
        l_cut : int ∈ [1, L_MAX]

        Returns
        -------
        smashed_data : intermediate activation tensor sent to server
        """
        l_cut = int(max(1, min(l_cut, self.L_MAX)))
        out   = x
        for i in range(l_cut):
            out = self.feature_layers[i](out)
        # `out` is the "smashed data" transmitted to the server / neighbour
        return out

    # ------------------------------------------------------------------
    # Phase 2 – SERVER side (runs on server or capable neighbour)
    # ------------------------------------------------------------------
    def server_forward(self, smashed: torch.Tensor, l_cut: int) -> torch.Tensor:
        """
        Execute feature layers [l_cut .. L_MAX-1] + classifier on the SERVER.

        Parameters
        ----------
        smashed : intermediate activation from client_forward()
        l_cut   : must match the l_cut used in client_forward()

        Returns
        -------
        logits  : (B, num_classes) raw class scores
        """
        l_cut = int(max(1, min(l_cut, self.L_MAX)))
        out   = smashed
        for i in range(l_cut, self.L_MAX):
            out = self.feature_layers[i](out)

        out    = self.pool(out)
        out    = self.flatten(out)
        logits = self.classifier(out)
        return logits

    # ------------------------------------------------------------------
    # Convenience: full forward (l_cut = L_MAX means "no split")
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, l_cut: Optional[int] = None) -> torch.Tensor:
        """
        Full forward pass.  If l_cut is None or L_MAX, no split is applied;
        the entire backbone runs in one shot (useful for local fallback or
        loss computation in main.py).
        """
        if l_cut is None:
            l_cut = self.L_MAX
        smashed = self.client_forward(x, l_cut)
        logits  = self.server_forward(smashed, l_cut)
        return logits

    # ------------------------------------------------------------------
    # SplitFed cost function helper  –  Eq. (13)
    # ------------------------------------------------------------------
    def splitfed_cost(self, l_cut: int, sinr: float,
                      w_c: float = 0.5, w_t: float = 0.5) -> float:
        """
        C_split(L, t) = w_c * E_comp(L) + w_t * DataSize(L) / (B * log2(1+SINR))

        E_comp(L) is approximated proportionally to the number of client-side
        parameters (more layers = more FLOPs on the drone).

        DataSize(L) is the number of elements in the smashed tensor produced
        at layer L, converted to a byte count (float32 = 4 bytes).

        Returns
        -------
        cost : float (lower is better)
        """
        # Approximate compute cost (count client-side parameters)
        client_params = sum(
            p.numel() for i in range(l_cut)
            for p in self.feature_layers[i].parameters()
        )
        e_comp = client_params / 1e6   # normalise to millions of parameters

        # Approximate data size: run a dummy forward to measure activation shape
        dummy_input = torch.zeros(1, *INPUT_SHAPE)
        with torch.no_grad():
            smashed = self.client_forward(dummy_input, l_cut)
        data_size = smashed.numel() * 4   # bytes (float32)

        sinr  = max(sinr, 1e-6)
        rate  = math.log2(1.0 + sinr)    # Shannon capacity (normalised B=1)
        t_tx  = data_size / rate

        cost  = w_c * e_comp + w_t * t_tx
        return float(cost)

    # ------------------------------------------------------------------
    # Produce soft labels for Knowledge Distillation  (Section II-B pt 3)
    # ------------------------------------------------------------------
    def soft_labels(self, x: torch.Tensor,
                    l_cut: int,
                    temperature: float = 4.0) -> torch.Tensor:
        """
        Generate soft probability targets used to train the Student model.

            p_soft = softmax(logits / T)

        A high temperature T flattens the distribution, making it more
        informative for the Student (Hinton et al., KD original paper).
        """
        with torch.no_grad():
            logits = self.forward(x, l_cut)
        return F.softmax(logits / temperature, dim=-1)


# ---------------------------------------------------------------------------
# Student Model  –  Lightweight KD fallback (Section II-B pt 2 & 3)
# ---------------------------------------------------------------------------
class StudentModel(nn.Module):
    """
    Tiny CNN that lives entirely on the drone.

    Architecture
    ------------
        Conv2d(3, 16, 3) → ReLU → MaxPool(2)
        Conv2d(16, 32, 3) → ReLU → MaxPool(2)
        AdaptiveAvgPool2d(4) → Flatten
        Linear(32*4*4, 128) → ReLU → Linear(128, num_classes)

    Trained via two loss terms:
        L_total = (1-α)*L_hard + α*L_soft_KD

    where L_hard = CrossEntropy(logits, true_labels)
      and L_soft_KD = KLDiv(log_softmax(student/T), soft_teacher_labels)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # → (16, 32, 32)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # → (32, 16, 16)
        )
        self.pool       = nn.AdaptiveAvgPool2d(4)   # → (32, 4, 4)
        self.flatten    = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.pool(out)
        out = self.flatten(out)
        return self.classifier(out)

    # ------------------------------------------------------------------
    # Knowledge-Distillation combined loss
    # ------------------------------------------------------------------
    @staticmethod
    def kd_loss(student_logits: torch.Tensor,
                teacher_soft:   torch.Tensor,
                hard_labels:    torch.Tensor,
                temperature:    float = 4.0,
                alpha:          float = 0.7) -> torch.Tensor:
        """
        Combined KD loss for online distillation.

        Parameters
        ----------
        student_logits  : raw logits from StudentModel  (B, C)
        teacher_soft    : soft probabilities from Teacher at temperature T  (B, C)
        hard_labels     : integer ground-truth class indices  (B,)
        temperature     : T used in Teacher's soft_labels()
        alpha           : weight of soft KD loss (1-α for hard CE)

        Returns
        -------
        total_loss : scalar tensor
        """
        # Hard cross-entropy loss
        ce_loss = F.cross_entropy(student_logits, hard_labels)

        # Soft KD loss (scaled by T² as per Hinton et al.)
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        kd_loss_val  = F.kl_div(student_soft, teacher_soft,
                                 reduction="batchmean") * (temperature ** 2)

        return (1.0 - alpha) * ce_loss + alpha * kd_loss_val


# ---------------------------------------------------------------------------
# Quick sanity-check shapes (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model   = SplitFedModel(num_classes=NUM_CLASSES, pretrained=False)
    student = StudentModel(num_classes=NUM_CLASSES)

    x = torch.randn(2, *INPUT_SHAPE)   # batch of 2 images

    for l_cut in [1, 4, 7, 10, 13]:
        smashed = model.client_forward(x, l_cut)
        logits  = model.server_forward(smashed, l_cut)
        cost    = model.splitfed_cost(l_cut, sinr=10.0)
        print(f"  l_cut={l_cut:2d} | smashed: {tuple(smashed.shape)} "
              f"| logits: {tuple(logits.shape)} | cost: {cost:.4f}")

    s_logits = student(x)
    soft     = model.soft_labels(x, l_cut=7)
    kd_l     = StudentModel.kd_loss(s_logits, soft,
                                     torch.randint(0, NUM_CLASSES, (2,)))
    print(f"\nStudent logits: {tuple(s_logits.shape)}  KD loss: {kd_l.item():.4f}")
