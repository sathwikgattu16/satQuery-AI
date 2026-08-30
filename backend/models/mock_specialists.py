"""
backend/models/mock_specialists.py
Intelligent, Content-Aware Mock/Demo Specialists for SatQuery AI.
Dynamically inspects uploaded raster pixels/bands and queries to generate unique,
domain-accurate remote sensing intelligence for different images.

Owners: Member 1, Member 4, Member 5, Member 6
"""

import hashlib
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from backend.models.base import BaseSpecialist

def analyze_image_properties(img_data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float, str, str]:
    """
    Computes visual/spectral properties from image tensors, PIL images, or raster bytes.
    Returns:
        (green_ratio, blue_ratio, brightness, dominant_class, signature_id)
    """
    filename = (context or {}).get("primary", {}).get("filename", "") if context else ""
    if not filename and context:
        filename = context.get("filename", "")
    f_lower = filename.lower()

    # Filename keyword priority for test predictability & semantic accuracy
    if any(k in f_lower for k in ["harbor", "coast", "marine", "port"]):
        return 0.1, 0.8, 0.5, "Coastal Harbor & Marine Logistics", "SIG-COASTAL-HARBOR"
    if any(k in f_lower for k in ["forest", "agri", "crop", "vegetation"]):
        return 0.7, 0.2, 0.4, "Dense Agricultural & Forest Canopy", "SIG-AGRI-CANOPY"
    if any(k in f_lower for k in ["urban", "city", "building"]):
        return 0.2, 0.2, 0.6, "Urban Metropolis & Built-up Infrastructure", "SIG-URBAN-INFRA"
    if any(k in f_lower for k in ["water", "river", "lake", "ocean"]):
        return 0.1, 0.9, 0.4, "Hydrological Water Bodies & River Basin", "SIG-HYDROLOGY"

    # 1. Multi-band Tensor Analysis [B, 6, 1, 224, 224] or [C, H, W]
    if isinstance(img_data, torch.Tensor):
        arr = img_data.detach().cpu().numpy()
        # Flatten to channel means
        if arr.ndim == 5:  # [1, C, 1, H, W]
            ch_means = arr[0, :, 0, :, :].mean(axis=(1, 2))
        elif arr.ndim == 3:  # [C, H, W]
            ch_means = arr.mean(axis=(1, 2))
        else:
            ch_means = np.array([0.3, 0.4, 0.3])
            
        if len(ch_means) >= 6:
            # Bands: B02(Blue), B03(Green), B04(Red), B05, B06, B07(NIR)
            blue = float(ch_means[0])
            green = float(ch_means[1])
            red = float(ch_means[2])
            nir = float(ch_means[5])
            brightness = (red + green + blue) / 3.0
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)
            
            if ndvi > 0.35:
                return 0.7, 0.2, brightness, "Dense Forest & Agriculture", "SPECTRAL-NDVI-HIGH"
            elif ndwi > 0.2:
                return 0.1, 0.8, brightness, "Hydrological Surface Water", "SPECTRAL-NDWI-WATER"
            elif brightness > 0.5:
                return 0.2, 0.2, brightness, "Urban Impervious Settlements", "SPECTRAL-URBAN-HIGH"
            else:
                return 0.3, 0.3, brightness, "Barren & Fallow Terrain", "SPECTRAL-ARID-LAND"

    # 2. PIL Image Analysis
    if isinstance(img_data, Image.Image):
        small = img_data.resize((64, 64)).convert("RGB")
        np_img = np.array(small, dtype=float) / 255.0
        r_mean = np_img[:, :, 0].mean()
        g_mean = np_img[:, :, 1].mean()
        b_mean = np_img[:, :, 2].mean()
        total = r_mean + g_mean + b_mean + 1e-6
        
        green_ratio = g_mean / total
        blue_ratio = b_mean / total
        brightness = total / 3.0
        
        if green_ratio > 0.40:
            return green_ratio, blue_ratio, brightness, "Dense Agricultural & Forest Canopy", "RGB-DOMINANT-GREEN"
        elif blue_ratio > 0.40:
            return green_ratio, blue_ratio, brightness, "Coastal Water Body & River Channels", "RGB-DOMINANT-BLUE"
        elif r_mean > 0.40 and g_mean > 0.40 and b_mean > 0.40:
            return green_ratio, blue_ratio, brightness, "Urban Settlements & Transport Logistics", "RGB-DOMINANT-URBAN"
        else:
            return green_ratio, blue_ratio, brightness, "Mixed Terrain & Shrublands", "RGB-MIXED-TERRAIN"

    # 3. Fallback Deterministic Variation based on Filename / Hash
    hash_val = int(hashlib.md5(str(filename or "scene").encode()).hexdigest()[:6], 16)
    classes = [
        "Dense Agricultural Canopy",
        "Coastal Harbor & Marine Basin",
        "Urban Metropolis & Highway Infrastructure",
        "Mountainous Forest & River Valley",
        "Barren Arid Desert Terrain"
    ]
    chosen_class = classes[hash_val % len(classes)]
    return 0.4, 0.3, 0.5, chosen_class, f"SIG-{hash_val % 1000:03d}"

class MockVQASpecialist(BaseSpecialist):
    """Content-aware VQA Specialist providing customized answers for each photo."""

    def __init__(self):
        super().__init__(name="Mock-VQASpecialist-v1")
        self.is_mock = True

    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        q_lower = (query or "").lower()
        _, _, brightness, dom_class, sig_id = analyze_image_properties(image_primary, context)
        filename = (context or {}).get("primary", {}).get("filename", "satellite_scene.tif")

        if any(w in q_lower for w in ["count", "how many", "number of", "enumerate"]):
            count_num = (int(hashlib.md5(str(filename + query).encode()).hexdigest()[:4], 16) % 18) + 4
            answer = (
                f"[MOCK DEMO] Target spatial enumeration on scene '{filename}': "
                f"Detected {count_num} distinct spatial features matching '{query or 'target structures'}' "
                f"with high positional confidence (bounding box IoU > 0.88, resolution 10m GSD)."
            )
        elif any(w in q_lower for w in ["water", "river", "lake", "ocean", "sea", "drainage"]):
            if "Water" in dom_class or "Harbor" in dom_class:
                answer = (
                    f"[MOCK DEMO] Hydrological feature analysis for '{filename}': "
                    "Detected major open surface water bodies (NDWI: +0.62). Identified active drainage channels "
                    "with high depth reflectance and adjacent riparian wetland corridors."
                )
            else:
                answer = (
                    f"[MOCK DEMO] Hydrological analysis for '{filename}': "
                    "No major ocean or lake detected in primary sector. Isolated secondary drainage reservoir "
                    "identified covering ~3.8% of scene area."
                )
        elif any(w in q_lower for w in ["runway", "airport", "aircraft", "apron", "plane"]):
            answer = (
                f"[MOCK DEMO] Transportation infrastructure assessment for '{filename}': "
                "Identified airfield corridor with active runway alignment, taxiway junctions, and "
                "surrounding clear-zone security buffers."
            )
        elif any(w in q_lower for w in ["crop", "agriculture", "vegetation", "forest", "tree"]):
            answer = (
                f"[MOCK DEMO] Vegetation analysis on scene '{filename}': "
                f"Spectral reflectance signature ({sig_id}) confirms {dom_class}. "
                "Estimated Normalized Difference Vegetation Index (NDVI) is +0.68, indicating healthy photosynthetic canopy."
            )
        else:
            # Custom answer reflecting specific image class
            answer = (
                f"[MOCK DEMO] Multi-spectral analysis of '{filename}' for query '{query or 'General Scene Analysis'}': "
                f"Identified dominant land-cover as {dom_class}. Radiometric brightness score: {brightness:.2f}, "
                f"Spectral Feature Tag: {sig_id}."
            )

        return {
            "answer": answer,
            "confidence": round(0.91 + (brightness % 0.08), 2),
            "evidence": {
                "type": "overlay",
                "title": f"Spatial Grounding Overlay ({dom_class})",
                "description": f"Spatial attention anchors and bounding box predictions for '{filename}'.",
                "metrics": {
                    "Identified Class": dom_class,
                    "Ground Sample Distance": "10m GSD",
                    "Signature Tag": sig_id,
                    "Execution Mode": "MOCK / DEMO SPECIALIST"
                }
            },
            "execution_detail": f"Executed Mock-VQASpecialist-v1 with content-aware analysis on {filename}."
        }

class MockCaptionSpecialist(BaseSpecialist):
    """Content-aware Captioning Specialist generating custom descriptions per photo."""

    def __init__(self):
        super().__init__(name="Mock-CaptionSpecialist-v1")
        self.is_mock = True

    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        _, _, brightness, dom_class, sig_id = analyze_image_properties(image_primary, context)
        filename = (context or {}).get("primary", {}).get("filename", "optical_scene.tif")

        answer = (
            f"[MOCK DEMO] High-resolution remote sensing observation of '{filename}': "
            f"The scene exhibits {dom_class} under clear atmospheric conditions. "
            f"Spatial structure indicates high feature contrast with {dom_class.lower()} characteristics, "
            f"continuous road/boundary connectivity, and low cloud cover (< 2%)."
        )

        return {
            "answer": answer,
            "confidence": 0.92,
            "evidence": {
                "type": "overlay",
                "title": f"Land-Cover Segmentation Overlay ({dom_class})",
                "description": f"Multi-class semantic segmentation mask for {filename}.",
                "metrics": {
                    "Dominant Class": dom_class,
                    "Spectral Signature": sig_id,
                    "Cloud Cover Fraction": "< 1.8%",
                    "Execution Mode": "MOCK / DEMO SPECIALIST"
                }
            },
            "execution_detail": f"Executed Mock-CaptionSpecialist-v1 for {filename}."
        }

class MockChangeSpecialist(BaseSpecialist):
    """Bi-temporal Change Specialist reflecting differences between T1 and T2."""

    def __init__(self):
        super().__init__(name="Mock-ChangeSpecialist-v1")
        self.is_mock = True

    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        q_lower = (query or "").lower()
        t1_name = (context or {}).get("primary", {}).get("filename", "T1_baseline.tif")
        t2_name = (context or {}).get("secondary", {}).get("filename", "T2_target.tif")

        hash_seed = int(hashlib.md5(f"{t1_name}_{t2_name}_{query}".encode()).hexdigest()[:4], 16)
        area_ha = round(10.0 + (hash_seed % 40) * 0.7, 1)

        if any(w in q_lower for w in ["flood", "water", "inundation"]):
            answer = (
                f"[MOCK DEMO] Bi-temporal spatial difference analysis between '{t1_name}' and '{t2_name}': "
                f"Detected {area_ha} hectares of surface water inundation along drainage basins. "
                "Agricultural sectors show significant saturation backscatter increase between T1 and T2."
            )
        elif any(w in q_lower for w in ["urban", "construction", "expand", "building"]):
            answer = (
                f"[MOCK DEMO] Urban infrastructure expansion analysis between '{t1_name}' and '{t2_name}': "
                f"Detected {area_ha} hectares of new impervious built-up expansion replacing former green canopy."
            )
        else:
            answer = (
                f"[MOCK DEMO] Bi-temporal change detection ({t1_name} vs {t2_name}): "
                f"Isolated {area_ha} hectares of morphological modifications across the observation window: "
                "including vegetative canopy shifts, new road corridors, and seasonal water line adjustments."
            )

        return {
            "answer": answer,
            "confidence": 0.93,
            "evidence": {
                "type": "change_mask",
                "title": f"Bi-Temporal Change Map ({t1_name} vs {t2_name})",
                "description": "Red indicates built-up expansion; Blue indicates hydrological expansion.",
                "metrics": {
                    "Changed Area": f"{area_ha} hectares",
                    "Co-registration Error": "< 0.3 pixels",
                    "Execution Mode": "MOCK / DEMO SPECIALIST"
                }
            },
            "execution_detail": f"Executed Mock-ChangeSpecialist-v1 comparing {t1_name} and {t2_name}."
        }

class MockOpticalSARSpecialist(BaseSpecialist):
    """Cross-sensor Optical+SAR Specialist providing sensor-aware analysis."""

    def __init__(self):
        super().__init__(name="Mock-OpticalSARSpecialist-v1")
        self.is_mock = True

    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        opt_name = (context or {}).get("primary", {}).get("filename", "optical.tif")
        sar_name = (context or {}).get("secondary", {}).get("filename", "sar.tif")
        q_lower = (query or "").lower()

        if any(w in q_lower for w in ["ship", "vessel", "maritime", "boat"]):
            answer = (
                f"[MOCK DEMO] Optical-SAR cross-sensor analysis ({opt_name} + {sar_name}): "
                "Detected maritime vessels in maritime corridor. SAR polarimetric double-bounce scattering "
                "confirmed metallic vessels obscured beneath optical cloud layers."
            )
        elif any(w in q_lower for w in ["crop", "agriculture", "soil", "moisture"]):
            answer = (
                f"[MOCK DEMO] Synergistic Optical-SAR crop analysis ({opt_name} + {sar_name}): "
                "Fusing optical spectral bands with SAR VV/VH backscatter indicates active crop growth with "
                "37.5% volumetric soil moisture content."
            )
        else:
            answer = (
                f"[MOCK DEMO] Multi-sensor fusion ({opt_name} + {sar_name}): "
                "Successfully penetrated cloud cover using SAR C-band microwave backscatter, resolving "
                "ground surface topology and road network continuity with high radiometric confidence."
            )

        return {
            "answer": answer,
            "confidence": 0.95,
            "evidence": {
                "type": "heatmap",
                "title": f"Optical + SAR Fusion Matrix ({opt_name} + {sar_name})",
                "description": "SAR VH radar backscatter overlaid on Optical L2A reflectance.",
                "metrics": {
                    "Cloud Penetration": "98.4%",
                    "Polarization Mode": "Dual Pol (VV + VH)",
                    "Optical Source": opt_name,
                    "SAR Source": sar_name,
                    "Execution Mode": "MOCK / DEMO SPECIALIST"
                }
            },
            "execution_detail": f"Executed Mock-OpticalSARSpecialist-v1 fusing {opt_name} and {sar_name}."
        }
