import { QueryRequest, QueryResponse, VisualEvidence } from '../types';

/**
 * SATQUERY AI - API Service
 * 
 * LOCKED BACKEND CONTRACT:
 * Endpoint: POST http://localhost:8000/api/analyze
 * Transport: multipart/form-data
 * Fields: task_hint, question, image, image_t2, sar
 */

const API_BASE_URL = 'http://127.0.0.1:8000';
const ANALYZE_ENDPOINT = `${API_BASE_URL}/api/analyze`;

// Default to false to enable live backend mode by default.
// Users can still toggle to Mock Mode via the UI DataSource switch.
export const USE_MOCK_DEFAULT = false;

/**
 * Generates an isolated mock response simulating the ISRO agent decision pipeline.
 */
export function generateMockResponse(request: QueryRequest): QueryResponse {
  const hasImage1 = !!request.image;
  const hasImage2 = !!request.image_t2;
  const hasSar = !!request.sar;

  const numImages = [hasImage1, hasImage2, hasSar].filter(Boolean).length;
  const questionLower = (request.question || '').toLowerCase();
  const hint = request.task_hint || 'vqa';

  // Agent task selection logic simulation (matching ISRO Remote Sensing requirements)
  let selectedTask = hint;
  let answer = '';
  let confidence = 0.94;
  let modelsUsed: string[] = [];
  let compatibilityNotes = 'All input sensor channels verified and calibrated.';
  let visualization: VisualEvidence | null = null;

  if (hasImage1 && hasImage2) {
    // Bi-temporal scenario
    selectedTask = 'change';
    modelsUsed = ['ISRO-Bitemporal-ChangeFormer-v2', 'Sentinel-2-DiffNet', 'SpatialTemporal-Agent-Core'];
    confidence = 0.92;
    compatibilityNotes = 'T1 and T2 coordinate grids co-registered. Cloud cover difference < 4.2%. Multi-date spectral alignment verified.';

    if (questionLower.includes('flood') || questionLower.includes('water')) {
      answer = 'Bi-temporal spatial difference analysis indicates significant water body inundation (+34.8% surface area) along the river delta between T1 and T2. Agricultural zones in Sectors 4 and 7 exhibit severe saturation backscatter signatures.';
    } else if (questionLower.includes('urban') || questionLower.includes('construction') || questionLower.includes('expand')) {
      answer = 'Detected 14.2 hectares of new built-up infrastructure expansion between T1 baseline and T2 observation. Vegetation loss of ~8.7 hectares observed in the eastern perimeter.';
    } else {
      answer = 'Bi-temporal change detection successfully isolated 3 distinct morphological alteration zones between T1 and T2: (1) 18.5% decrease in dense canopy, (2) 12.1% expansion in built-up impervious surfaces, and (3) seasonal water line retreat along the western reservoir.';
    }

    visualization = {
      type: 'change_mask',
      title: 'Bi-Temporal Change Intensity Map',
      description: 'Red highlights indicate built-up expansion; Blue highlights indicate hydrological surface inundation between T1 and T2.',
      metrics: {
        'Changed Area': '23.4 sq km',
        'NDVI Difference': '-0.28 avg',
        'Co-registration Error': '< 0.3 pixels',
      },
    };
  } else if (hasImage1 && hasSar) {
    // Optical + SAR Fusion scenario
    selectedTask = 'multimodal_fusion';
    modelsUsed = ['ISRO-OptSAR-FusionNet-v3', 'RISAT-1A-Polarimetric-Analyzer', 'Prithvi-EO-Multimodal'];
    confidence = 0.95;
    compatibilityNotes = 'Optical RGB/NIR bands fused with SAR VV/VH dual-polarization backscatter. Penetrated high-density cloud cover in Sector B.';

    if (questionLower.includes('ship') || questionLower.includes('vessel') || questionLower.includes('ocean')) {
      answer = 'Optical-SAR cross-sensor analysis detected 6 maritime vessels in the designated EEZ corridor. SAR polarimetric double-bounce scattering confirmed 2 metal-hulled container ships obscured beneath cirrus cloud layers in the optical spectrum.';
    } else if (questionLower.includes('crop') || questionLower.includes('agriculture')) {
      answer = 'Synergistic Optical-SAR classification identified Paddy/Rice cultivation in saturated vegetative stage (High VH/VV ratio) and Wheat in vegetative growth. Soil moisture estimated at 38.2% volumetric water content.';
    } else {
      answer = 'Multi-sensor fusion successfully resolved optical cloud occlusion using SAR C-band penetration. Identified clear ground topography, road network continuity, and hidden hydrological pooling with high radiometric confidence.';
    }

    visualization = {
      type: 'heatmap',
      title: 'Optical + SAR Polarimetric Fusion Matrix',
      description: 'Synthetic Aperture Radar VH backscatter overlaid on Optical L2A reflectance.',
      metrics: {
        'Cloud Penetration': '98.4%',
        'Polarization Mode': 'Dual Pol (VV + VH)',
        'Spatial Resolution': '10m / GSD',
      },
    };
  } else {
    // Single image scenario
    if (hint === 'caption') {
      selectedTask = 'caption';
      modelsUsed = ['ISRO-RemoteSense-Captioner-L', 'GeoChat-Vision-Encoder'];
      confidence = 0.91;
      compatibilityNotes = 'Single optical satellite tile processed. Standard radiometric normalization applied.';
      answer = 'High-resolution multispectral optical scene displaying an active coastal harbor with shipping docks, breakwater structures, adjacent intertidal mudflats, and mixed commercial transport logistics infrastructure under clear atmospheric conditions.';
    } else {
      selectedTask = 'vqa';
      modelsUsed = ['Prithvi-EO-2.0-VQA', 'ISRO-EarthObservation-LLM', 'SpatialAnchor-Detector'];
      confidence = 0.93;
      compatibilityNotes = 'Single optical image accepted. Visual question parsed against 10m Ground Sample Distance.';

      if (questionLower.includes('how many') || questionLower.includes('count')) {
        answer = 'Target spatial enumeration detected 14 distinct features matching your query criteria across the scene coordinates with high positional confidence (bounding box IoU > 0.88).';
      } else if (questionLower.includes('water') || questionLower.includes('river') || questionLower.includes('lake')) {
        answer = 'Hydrological feature identification confirms a perennial river channel with an average width of 145 meters traversing northwest to southeast, with secondary tributary drainage patterns in the northern sector.';
      } else {
        answer = `Analysis of the remote sensing scene for query "${request.question || 'Feature analysis'}": High-confidence land cover segmentation shows 45% agricultural canopy, 30% barren/fallow terrain, 15% settlements/roadways, and 10% surface water reservoirs.`;
      }
    }
  }

  // Handle agent override demonstration (e.g. user provided hint 'vqa' but bi-temporal images provided)
  return {
    success: true,
    task: selectedTask,
    answer,
    confidence,
    processing_time: +(Math.random() * 1.5 + 1.8).toFixed(2), // 1.8s - 3.3s
    execution_summary: {
      selected_task: selectedTask,
      task_hint_provided: request.task_hint || 'none',
      models_used: modelsUsed,
      num_images_provided: numImages,
      compatibility_notes: compatibilityNotes,
      trace_steps: [
        `Received ${numImages} input file(s) with user query: "${request.question || 'N/A'}"`,
        `Task hint evaluated: "${request.task_hint || 'none'}" -> Agent authoritative task chosen: "${selectedTask}"`,
        `Sensor alignment & compatibility check: ${compatibilityNotes}`,
        `Dispatched specialized model ensemble: [${modelsUsed.join(', ')}]`,
        `Synthesized geospatial response with confidence score: ${(confidence * 100).toFixed(1)}%`,
      ],
    },
    visualization,
  };
}

/**
 * Submits the multi-modal analysis request to the backend or mock simulator.
 * 
 * Transport: multipart/form-data
 * Fields:
 * - task_hint (optional)
 * - question (optional)
 * - image (file, primary image)
 * - image_t2 (file, used for bi-temporal change)
 * - sar (file, used for optical + SAR)
 */
export async function submitAnalysis(
  request: QueryRequest,
  options: { useMock?: boolean } = {}
): Promise<QueryResponse> {
  const useMock = options.useMock !== undefined ? options.useMock : USE_MOCK_DEFAULT;

  if (useMock) {
    // Simulate real network delay for genuine UX testing
    await new Promise((resolve) => setTimeout(resolve, 1400));
    return generateMockResponse(request);
  }

  // LOCKED CONTRACT: Construct multipart/form-data
  const formData = new FormData();

  if (request.task_hint) {
    formData.append('task_hint', request.task_hint);
  }
  if (request.question) {
    formData.append('question', request.question);
  }
  if (request.image) {
    formData.append('image', request.image);
  }
  if (request.image_t2) {
    formData.append('image_t2', request.image_t2);
  }
  if (request.sar) {
    formData.append('sar', request.sar);
  }

  try {
    const response = await fetch(ANALYZE_ENDPOINT, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server returned ${response.status}: ${errorText || response.statusText}`);
    }

    const data: QueryResponse = await response.json();
    return data;
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown connection error';
    throw new Error(`Failed to communicate with SATQUERY backend at ${ANALYZE_ENDPOINT}: ${message}`);
  }
}
