"""
model.py — DeepfakeDetector: ResNet-50 backbone with hierarchical multi-level
feature extraction and a regularized MLP classification head.

Architecture:
  • Backbone  : ResNet-50 pretrained on ImageNet (frozen early layers, fine-tuned later)
  • Feature taps:
      - Stage 1 (layer1) → 256-ch, 56×56  → low-level edges / blending artifacts
      - Stage 2 (layer2) → 512-ch, 28×28  → facial geometry / landmark patterns
      - Stage 4 (layer4) → 2048-ch, 7×7   → high-level semantics / illumination
  • Fusion    : GAP → project [256, 256, 512] → concatenate → 1024-d forensic vector
  • Head      : BN → Linear(512) → ReLU → Dropout(0.4) → Linear(2) → Softmax
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MultiLevelFeatureExtractor(nn.Module):
    """
    Wraps a ResNet-50 backbone and registers forward hooks to tap
    feature maps at three hierarchical stages.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── Load backbone ──────────────────────────────────────────────────
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # ── Expose individual stages as sub-modules ────────────────────────
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # 256-ch, 56x56
        self.layer2 = backbone.layer2   # 512-ch, 28x28
        self.layer3 = backbone.layer3   # 1024-ch, 14x14  (not tapped)
        self.layer4 = backbone.layer4   # 2048-ch, 7x7

    def forward(self, x: torch.Tensor):
        """
        Returns:
            feat_low  : (B, 256, 56, 56)
            feat_mid  : (B, 512, 28, 28)
            feat_high : (B, 2048, 7, 7)
        """
        x = self.stem(x)
        feat_low  = self.layer1(x)          # Stage 1 tap
        feat_mid  = self.layer2(feat_low)   # Stage 2 tap
        x         = self.layer3(feat_mid)   # Stage 3 (pass-through)
        feat_high = self.layer4(x)          # Stage 4 tap
        return feat_low, feat_mid, feat_high


class DeepfakeDetector(nn.Module):
    """
    Full deepfake detection model:
      1. Multi-level feature extraction (3 ResNet-50 taps)
      2. Global Average Pooling on each tap
      3. Linear projection to standardised dims [256, 256, 512]
      4. Concatenation -> 1024-d fused forensic vector
      5. Regularised MLP head -> binary classification (REAL / FAKE)
    """

    def __init__(self, pretrained: bool = True, freeze_stem: bool = True):
        super().__init__()

        # ── Backbone with hierarchical taps ───────────────────────────────
        self.backbone = MultiLevelFeatureExtractor(pretrained=pretrained)

        # Optionally freeze stem + layer1 so fine-tuning focuses on
        # the deeper, more semantic layers
        if freeze_stem:
            frozen = list(self.backbone.stem.parameters()) + \
                     list(self.backbone.layer1.parameters())
            for p in frozen:
                p.requires_grad = False

        # ── Global Average Pooling ─────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B, C, 1, 1)

        # ── Projection layers (GAP output -> standardised dim) ─────────────
        #   low  : 256  -> 256
        #   mid  : 512  -> 256
        #   high : 2048 -> 512
        self.proj_low  = nn.Sequential(
            nn.Linear(256,  256),
            nn.ReLU(inplace=True),
        )
        self.proj_mid  = nn.Sequential(
            nn.Linear(512,  256),
            nn.ReLU(inplace=True),
        )
        self.proj_high = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
        )

        # ── Regularised MLP Classifier Head ───────────────────────────────
        # Fused dim: 256 + 256 + 512 = 1024
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 2),       # Binary: 0 = REAL, 1 = FAKE
        )

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 3, 224, 224) — normalised facial crop tensor
        Returns:
            logits  : (B, 2)    — raw class scores
            probs   : (B, 2)    — softmax probabilities [p_real, p_fake]
            forensic: (B, 1024) — fused feature vector (for diagnostics)
        """
        # 1. Hierarchical feature extraction
        feat_low, feat_mid, feat_high = self.backbone(x)

        # 2. Global Average Pooling -> flatten
        v_low  = self.gap(feat_low ).squeeze(-1).squeeze(-1)   # (B, 256)
        v_mid  = self.gap(feat_mid ).squeeze(-1).squeeze(-1)   # (B, 512)
        v_high = self.gap(feat_high).squeeze(-1).squeeze(-1)   # (B, 2048)

        # 3. Projections
        p_low  = self.proj_low (v_low )   # (B, 256)
        p_mid  = self.proj_mid (v_mid )   # (B, 256)
        p_high = self.proj_high(v_high)   # (B, 512)

        # 4. Concatenate -> 1024-d forensic vector
        forensic = torch.cat([p_low, p_mid, p_high], dim=1)   # (B, 1024)

        # 5. Classifier
        logits = self.classifier(forensic)                      # (B, 2)
        probs  = torch.softmax(logits, dim=1)                   # (B, 2)

        return logits, probs, forensic

    # ── Convenience helpers ────────────────────────────────────────────────
    def predict(self, x: torch.Tensor):
        """Returns (class_idx, confidence) for a single-image batch."""
        self.eval()
        with torch.no_grad():
            _, probs, _ = self.forward(x)
        class_idx  = probs.argmax(dim=1)          # 0 = REAL, 1 = FAKE
        confidence = probs.max(dim=1).values
        return class_idx, confidence

    @staticmethod
    def class_label(idx: int) -> str:
        return ["REAL", "FAKE"][int(idx)]
