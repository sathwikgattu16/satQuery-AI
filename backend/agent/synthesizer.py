"""
backend/agent/synthesizer.py
Structured Geospatial Synthesis Engine for SatQuery AI.

A deterministic, offline natural-language synthesis layer grounded strictly in
real Prithvi backbone, Change, and Fusion computed outputs.
Does not claim trained generative heads or fabricate unsupported semantic entities.

Owner: Member 1 & Member 4
"""

from typing import Dict, Any, Optional

class StructuredGeospatialSynthesizer:
    """
    Translates structured Prithvi representations, change metrics, and
    multimodal fusion tensors into transparent, authoritative Earth Observation reports.
    """

    @staticmethod
    def synthesize_vqa(
        query: str,
        feature_metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synthesizes a truthful VQA answer grounded in extracted Prithvi optical features.
        """
        q_str = query.strip() if query else "General visual inquiry"

        if feature_metrics and "cls_dim" in feature_metrics:
            cls_dim = feature_metrics.get("cls_dim", 768)
            num_patches = feature_metrics.get("num_patches", 196)
            cls_norm = feature_metrics.get("cls_norm", "N/A")
            return (
                f"The adapted Prithvi-EO-2.0 encoder extracted a {cls_dim}-dimensional global CLS representation "
                f"(L2 norm: {cls_norm}) and {num_patches} spatial patch representations from the supplied six-band optical scene. "
                f"Regarding query '{q_str}': this multi-spectral representation is indexed for downstream analysis; "
                f"the downstream language reasoning head is currently a transparent prototype placeholder (no calibrated object/class prediction was produced)."
            )

        return (
            f"Visual Question Answering feature evaluation for query: '{q_str}'. "
            f"Input optical scene processed through single-image pipeline. "
            f"Downstream natural language reasoning head is currently operating as a prototype placeholder."
        )

    @staticmethod
    def synthesize_caption(
        feature_metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Synthesizes a truthful Captioning description grounded in extracted Prithvi optical features.
        """
        if feature_metrics and "cls_dim" in feature_metrics:
            cls_dim = feature_metrics.get("cls_dim", 768)
            num_patches = feature_metrics.get("num_patches", 196)
            cls_norm = feature_metrics.get("cls_norm", "N/A")
            return (
                f"The supplied Sentinel-2 multispectral scene was encoded by the adapted Prithvi-EO-2.0 backbone "
                f"into a {cls_dim}-dimensional global CLS feature vector (L2 norm: {cls_norm}) and {num_patches} spatial patch features. "
                f"This prototype provides a feature-grounded scene representation summary; downstream autoregressive natural-language caption generation is currently a prototype placeholder."
            )

        return (
            "Remote sensing scene analysis: Multispectral optical tile ingested and processed under single-image captioning pipeline. "
            "Downstream descriptive caption generation is operating as a prototype placeholder."
        )

    @staticmethod
    def synthesize_change(
        change_metrics: Dict[str, Any],
        query: Optional[str] = None
    ) -> str:
        """
        Synthesizes a truthful bi-temporal Change answer grounded in Prithvi cosine distance metrics.
        """
        cls_dist = float(change_metrics.get("cls_distance", 0.0))
        mean_patch = float(change_metrics.get("mean_patch_change", 0.0))
        max_patch = float(change_metrics.get("max_patch_change", 0.0))
        h = change_metrics.get("heatmap_height", 14)
        w = change_metrics.get("heatmap_width", 14)

        # Explicit threshold for identical or numerically indistinguishable observations
        if cls_dist <= 1e-4 and mean_patch <= 1e-3:
            level = "No significant representation-level change"
        elif cls_dist < 0.03 and mean_patch < 0.20:
            level = "Low representation-level change"
        elif cls_dist < 0.10 or mean_patch < 0.40:
            level = "Moderate representation-level change"
        else:
            level = "Pronounced representation-level change"

        q_context = f" Query: '{query.strip()}'." if query and query.strip() else ""

        return (
            f"{level} was detected between T1 baseline and T2 observation. "
            f"Prithvi global CLS cosine distance = {cls_dist:.4f}, mean spatial patch divergence = {mean_patch:.4f}, "
            f"and maximum local patch divergence = {max_patch:.4f} across the {h}x{w} spatial grid.{q_context}"
        )

    @staticmethod
    def synthesize_fusion(
        fusion_metrics: Dict[str, Any],
        query: Optional[str] = None
    ) -> str:
        """
        Synthesizes a truthful Multimodal Fusion answer grounded in joint optical+SAR representations.
        """
        opt_dim = fusion_metrics.get("optical_dim", 768)
        sar_dim = fusion_metrics.get("sar_dim", 256)
        fused_dim = fusion_metrics.get("fused_dim", 1024)
        opt_norm = float(fusion_metrics.get("optical_norm", 0.0))
        sar_norm = float(fusion_metrics.get("sar_norm", 0.0))
        fused_norm = float(fusion_metrics.get("fused_norm", 0.0))

        q_context = f" for query: '{query.strip()}'" if query and query.strip() else ""

        return (
            f"Optical-SAR multimodal feature fusion completed{q_context}. "
            f"Combined the {opt_dim}-dimensional Sentinel-2 optical representation (L2 norm: {opt_norm:.2f}) "
            f"with the {sar_dim}-dimensional Sentinel-1 SAR representation (L2 norm: {sar_norm:.2f}) "
            f"into a {fused_dim}-dimensional joint feature representation (L2 norm: {fused_norm:.2f}). "
            f"Joint feature space integrates multispectral reflectance with polarimetric backscatter characteristics without assuming a trained downstream classifier head."
        )
