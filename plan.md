IIIF Polygon Masking ML Pipeline - Detailed Implementation Plan
Executive Summary
This plan outlines the development of a two-stage ML pipeline for document analysis:
1. Stage 1: Region detection model that identifies text regions and outputs polygon boundaries
2. Stage 2: OCR/text recognition model that benefits from accurately masked regions
Key Innovation: Using Stage 1's polygon predictions to create clean, masked training data for Stage 2, eliminating background noise and improving model accuracy.

Architecture Overview
┌─────────────────┐
│  IIIF Server    │
│  (Source Docs)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              IIIF Polygon Masking Endpoint              │
│  • Fetches regions from IIIF server                     │
│  • Applies polygon masks                                │
│  • Returns clean, bounded images                        │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
    ┌────────┴─────────┐
    │                  │
    ▼                  ▼
┌─────────┐      ┌──────────┐
│ Stage 1 │      │ Stage 2  │
│ Region  │─────▶│   OCR    │
│ Model   │      │  Model   │
└─────────┘      └──────────┘

Phase 1: Core Endpoint Development
1.1 Enhance Base Endpoint (Week 1-2)
Current Status: Basic polygon masking implemented
Enhancements Needed:
A. Batch Processing
@app.route('/mask_annotations_batch', methods=['POST'])
def mask_annotations_batch():
    """
    Process multiple annotations in a single request.
    
    Payload:
    {
        "iiif_base_url": "...",
        "annotations": [
            {"id": "region_1", "polygon": [[x,y], ...], "scale": 1.0},
            {"id": "region_2", "polygon": [[x,y], ...], "scale": 0.5},
            ...
        ],
        "output_format": "zip"  // or "json" for base64
    }
    
    Returns: ZIP file with all masked regions or JSON with base64 images
    """
Benefits:
* Process entire document pages in one request
* Reduced HTTP overhead
* Easier integration with training pipelines
B. Caching Layer
* Cache IIIF requests to reduce server load
* Use Redis or in-memory cache
* Cache both full regions and info.json responses
* TTL: 1 hour for active training, 24 hours for static datasets
C. Error Handling & Retry Logic
* Retry failed IIIF requests (3 attempts with exponential backoff)
* Handle partial batch failures gracefully
* Return detailed error information per annotation
* Circuit breaker for consistently failing IIIF servers
D. Performance Optimization
* Async IIIF fetching for batch operations
* Image processing thread pool
* Streaming responses for large batches
* Memory-efficient processing (don't load all images at once)

1.2 Add Annotation Format Support (Week 2)
Support Multiple Standards:
Web Annotation Model (W3C)
{
  "@context": "http://www.w3.org/ns/anno.jsonld",
  "type": "Annotation",
  "target": {
    "source": "https://iiif.example.org/image1",
    "selector": {
      "type": "SvgSelector",
      "value": "<svg><polygon points='x1,y1 x2,y2 ...'/></svg>"
    }
  }
}
IIIF Annotations
{
  "@context": "http://iiif.io/api/presentation/3/context.json",
  "type": "Annotation",
  "motivation": "commenting",
  "body": { "type": "TextualBody", "value": "text content" },
  "target": {
    "type": "SpecificResource",
    "source": "https://iiif.example.org/canvas",
    "selector": {
      "type": "FragmentSelector",
      "value": "xywh=pixel:x,y,w,h"
    }
  }
}
Custom JSON Format
{
  "image_id": "doc_001_page_03",
  "regions": [
    {
      "region_id": "heading_1",
      "type": "title",
      "coordinates": [[x1,y1], [x2,y2], ...],
      "confidence": 0.95
    }
  ]
}
Implementation:
* Create adapter pattern for different formats
* Auto-detect annotation format
* Normalize to internal representation

1.3 Add Metadata & Tracking (Week 2)
Annotation Metadata:
{
    "annotation_id": "unique_id",
    "source_image": "iiif_url",
    "bounding_box": [x, y, width, height],
    "polygon_area": 12345,  // pixels
    "extraction_timestamp": "2025-10-16T10:30:00Z",
    "scale_factor": 1.0,
    "region_type": "text_block",  // optional classification
    "confidence": 0.95  // from Stage 1 model
}
Benefits:
* Track which annotations were processed
* Filter low-confidence regions
* Audit training data provenance
* Reproducibility

Phase 2: Stage 1 - Region Detection Model
2.1 Data Preparation (Week 3-4)
Dataset Requirements:
* Minimum 1,000 annotated document images
* Diverse document types (manuscripts, newspapers, forms, books)
* Ground truth polygon annotations
* Multiple annotators for quality control
Recommended Datasets:
* DocBank - 500K document pages with layout annotations
* PubLayNet - 360K document images with layout segmentation
* FUNSD - Forms understanding dataset
* READ-BAD - Historical document baselines
* Custom Annotations - Your specific document corpus
Data Pipeline:
Raw Documents → Annotation Tool → Quality Control → Format Conversion → Training Set
                                                   ↓
                                        Ground Truth Polygons
Annotation Tools:
* VGG Image Annotator (VIA)
* LabelMe
* CVAT
* Prodigy (for active learning)

2.2 Model Architecture (Week 4-5)
Recommended Approach: Instance Segmentation
Model Options:
Option A: Mask R-CNN (Recommended for Start)
* Pros: Proven architecture, good documentation, polygon output
* Cons: Slower inference than other options
* Framework: Detectron2 (PyTorch)
* Training Time: 2-3 days on 4x V100 GPUs
Option B: YOLACT/YOLACT++
* Pros: Real-time performance, simpler architecture
* Cons: Slightly lower accuracy than Mask R-CNN
* Framework: PyTorch
* Training Time: 1-2 days on 4x V100 GPUs
Option C: LayoutLMv3 + Segmentation Head
* Pros: Multimodal (text + layout), state-of-the-art for documents
* Cons: Requires text data, more complex
* Framework: Hugging Face Transformers
* Training Time: 3-4 days on 8x V100 GPUs
Option D: Custom U-Net + Polygon Fitting
* Pros: Full control, lightweight
* Cons: More development work
* Framework: PyTorch/TensorFlow
* Training Time: 1-2 days on 2x V100 GPUs
Recommended Starting Point: Mask R-CNN with ResNet-50 backbone

2.3 Training Configuration (Week 5-6)
Hyperparameters:
model:
  backbone: resnet50-fpn
  anchor_sizes: [32, 64, 128, 256, 512]
  roi_heads:
    num_classes: 5  # title, paragraph, caption, table, figure
    
training:
  batch_size: 4  # per GPU
  learning_rate: 0.0025
  lr_scheduler: warmup_multistep
  max_iterations: 40000
  checkpoint_period: 5000
  
data_augmentation:
  - random_flip: horizontal
  - random_rotation: [-5, 5]  # degrees
  - color_jitter: [0.4, 0.4, 0.4, 0.1]
  - random_crop: [0.8, 1.0]
  
validation:
  eval_period: 2000
  metric: AP@0.5:0.95  # COCO-style mAP
Training Process:
1. Initialize with COCO pre-trained weights
2. Train backbone layers first (freeze other layers)
3. Unfreeze all layers and train end-to-end
4. Fine-tune on specific document types
Expected Performance:
* mAP@0.5: 0.75-0.85 (good)
* mAP@0.75: 0.60-0.70 (precise boundaries)
* Inference: 100-300ms per image on GPU

2.4 Polygon Post-Processing (Week 6)
Raw Model Output → Clean Polygons:
1. Mask to Polygon Conversion:
    * Use contour detection (OpenCV)
    * Apply Douglas-Peucker algorithm for simplification
    * Remove small artifacts (< 100 pixels)
2. Polygon Refinement:
    * Smooth jagged edges
    * Snap to text baselines (for text regions)
    * Merge overlapping regions
    * Split complex regions
3. Quality Filtering: def filter_predictions(predictions, min_confidence=0.7, min_area=500):
4.     filtered = []
5.     for pred in predictions:
6.         if pred['score'] >= min_confidence and pred['area'] >= min_area:
7.             filtered.append(pred)
8.     return filtered
9. 

Phase 3: Integration Layer (Week 7-8)
3.1 Pipeline Orchestration
Workflow:
Document → Stage 1 Model → Predictions → Masking Endpoint → Masked Regions → Stage 2 Model
Implementation:
class MLPipeline:
    def __init__(self, stage1_model, masking_endpoint_url, stage2_model):
        self.stage1 = stage1_model
        self.masking_url = masking_endpoint_url
        self.stage2 = stage2_model
    
    def process_document(self, iiif_url):
        # Stage 1: Detect regions
        predictions = self.stage1.predict(iiif_url)
        
        # Filter low-confidence predictions
        filtered = self.filter_predictions(predictions)
        
        # Prepare batch request for masking endpoint
        batch_request = {
            "iiif_base_url": iiif_url,
            "annotations": [
                {"id": p['id'], "polygon": p['polygon'], "scale": 1.0}
                for p in filtered
            ]
        }
        
        # Get masked regions
        masked_regions = requests.post(self.masking_url, json=batch_request)
        
        # Stage 2: Process each masked region
        results = []
        for region_id, image in masked_regions.items():
            text = self.stage2.recognize_text(image)
            results.append({
                "region_id": region_id,
                "text": text,
                "confidence": predictions[region_id]['score']
            })
        
        return results

3.2 Training Data Generation for Stage 2
Key Innovation: Use Stage 1 predictions to create training data for Stage 2
Process:
Ground Truth Annotations → Masking Endpoint → Masked Training Images
                                           ↓
                                  Ground Truth Text Labels
                                           ↓
                                Stage 2 Training Dataset
Benefits:
* Stage 2 never sees background noise during training
* More robust to varied layouts
* Better generalization to new document types
* Learns from precisely bounded regions
Data Generation Script:
def generate_stage2_training_data(annotations_file, iiif_urls, output_dir):
    """
    Generate masked training images for Stage 2 OCR model.
    """
    annotations = load_annotations(annotations_file)
    
    for doc_id, doc_annotations in annotations.items():
        iiif_url = iiif_urls[doc_id]
        
        # Prepare batch request
        batch_request = {
            "iiif_base_url": iiif_url,
            "annotations": doc_annotations,
            "output_format": "png"
        }
        
        # Get masked regions
        response = requests.post(MASKING_ENDPOINT, json=batch_request)
        
        # Save with metadata
        for region_id, image_data in response.items():
            image_path = f"{output_dir}/{doc_id}_{region_id}.png"
            save_image(image_data, image_path)
            
            # Save ground truth text
            text_path = f"{output_dir}/{doc_id}_{region_id}.txt"
            with open(text_path, 'w') as f:
                f.write(doc_annotations[region_id]['text'])

Phase 4: Stage 2 - OCR/Text Recognition Model
4.1 Model Selection (Week 9)
Options:
Option A: TrOCR (Recommended)
* Type: Transformer-based OCR
* Pros: State-of-the-art accuracy, handles various fonts
* Cons: Slower inference
* Best For: High-quality transcription, mixed fonts
Option B: CRNN + CTC
* Type: CNN + RNN with CTC loss
* Pros: Fast, well-established
* Cons: Limited to single-line text
* Best For: Simple text lines, forms
Option C: Tesseract 5.0
* Type: Traditional OCR with LSTM
* Pros: Easy to deploy, good baseline
* Cons: Less accurate on degraded documents
* Best For: Quick prototyping, modern printed text
Option D: PaddleOCR
* Type: Multi-stage (detection + recognition)
* Pros: Very fast, good multilingual support
* Cons: Chinese-focused documentation
* Best For: Production deployment, speed priority
Recommendation: Start with TrOCR, fine-tune on your masked regions

4.2 Training Stage 2 (Week 9-10)
Dataset: Masked regions from your masking endpoint + ground truth text
Training Configuration:
model:
  base_model: microsoft/trocr-large-handwritten
  max_length: 256  # max characters per region
  
training:
  batch_size: 16
  learning_rate: 5e-5
  epochs: 20
  early_stopping_patience: 3
  
data:
  input: masked_png_images
  labels: ground_truth_text
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  
augmentation:
  # Minimal - regions already cleaned by masking
  - random_brightness: [0.9, 1.1]
  - random_contrast: [0.9, 1.1]
Key Advantage: Because inputs are masked, the model learns to focus on text within boundaries, not background elements

4.3 Evaluation Metrics (Week 10)
Character Error Rate (CER):
cer = levenshtein_distance(prediction, ground_truth) / len(ground_truth)
Word Error Rate (WER):
wer = levenshtein_distance(pred_words, gt_words) / len(gt_words)
Per-Region Accuracy:
accuracy = (predictions == ground_truth).mean()
Expected Performance:
* CER: < 5% (good), < 2% (excellent)
* WER: < 10% (good), < 5% (excellent)
* Per-Region Accuracy: > 85% (good), > 95% (excellent)

Phase 5: Deployment & Production (Week 11-12)
5.1 API Service Architecture
┌──────────────┐
│ Load Balancer│
└──────┬───────┘
       │
   ┌───┴────┬─────────┬──────────┐
   │        │         │          │
   ▼        ▼         ▼          ▼
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│API 1│  │API 2│  │API 3│  │API 4│
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │         │          │
   └────────┴─────────┴──────────┘
            │
            ▼
    ┌───────────────┐
    │ Message Queue │  (RabbitMQ/Redis)
    └───────┬───────┘
            │
    ┌───────┴────────┬──────────┐
    │                │          │
    ▼                ▼          ▼
┌─────────┐    ┌─────────┐  ┌─────────┐
│Worker 1 │    │Worker 2 │  │Worker 3 │
│Stage 1  │    │Masking  │  │Stage 2  │
└─────────┘    └─────────┘  └─────────┘
Components:
1. API Gateway: Rate limiting, authentication, request routing
2. Worker Pool: Async processing of ML tasks
3. Message Queue: Job distribution and retry logic
4. Cache Layer: Redis for IIIF responses and model predictions
5. Storage: S3/MinIO for masked images and results

5.2 Containerization
Docker Compose Setup:
version: '3.8'

services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on:
      - redis
      - rabbitmq
  
  stage1_worker:
    build: ./workers/stage1
    deploy:
      replicas: 2
    environment:
      - GPU_DEVICE=0,1
      - MODEL_PATH=/models/stage1
    volumes:
      - ./models:/models
    runtime: nvidia
  
  masking_worker:
    build: ./workers/masking
    deploy:
      replicas: 4
    environment:
      - IIIF_CACHE_TTL=3600
  
  stage2_worker:
    build: ./workers/stage2
    deploy:
      replicas: 2
    environment:
      - GPU_DEVICE=0,1
      - MODEL_PATH=/models/stage2
    runtime: nvidia
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
  
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "15672:15672"

volumes:
  redis_data:

5.3 Monitoring & Logging
Metrics to Track:
1. Performance:
    * Request latency (p50, p95, p99)
    * Throughput (requests/second)
    * GPU utilization
    * Cache hit rate
2. Model Quality:
    * Stage 1: Average confidence score, mAP
    * Stage 2: Average CER, WER
    * End-to-end accuracy
    * Error rate by document type
3. System Health:
    * Queue depth
    * Worker availability
    * Failed jobs
    * Memory usage
Tools:
* Prometheus: Metrics collection
* Grafana: Dashboards
* ELK Stack: Log aggregation
* Sentry: Error tracking

5.4 Cost Optimization
Strategies:
1. IIIF Caching:
    * Cache frequently accessed regions
    * Estimated savings: 60-80% on IIIF requests
2. Batch Processing:
    * Process multiple regions per GPU batch
    * 3-5x throughput improvement
3. Model Optimization:
    * TensorRT/ONNX for faster inference
    * Quantization (FP16/INT8)
    * 2-3x speed improvement
4. Auto-scaling:
    * Scale workers based on queue depth
    * Use spot instances for batch jobs
    * 40-60% cost reduction
Estimated Costs (AWS, monthly):
* 4x g4dn.xlarge (inference): $1,200
* Redis/RabbitMQ (t3.medium): $100
* S3 storage (1TB): $25
* Data transfer: $200
* Total: ~$1,500/month for 1M documents/month

Phase 6: Testing & Validation (Week 13-14)
6.1 Unit Tests
# Test masking endpoint
def test_polygon_masking():
    request = {
        "iiif_base_url": TEST_IIIF_URL,
        "polygon": [[0,0], [100,0], [100,100], [0,100]],
        "scale_factor": 1.0
    }
    response = client.post('/mask_annotation', json=request)
    assert response.status_code == 200
    assert response.content_type == 'image/png'

# Test batch processing
def test_batch_masking():
    request = {
        "iiif_base_url": TEST_IIIF_URL,
        "annotations": [
            {"id": "1", "polygon": [...]},
            {"id": "2", "polygon": [...]}
        ]
    }
    response = client.post('/mask_annotations_batch', json=request)
    assert len(response.json()['results']) == 2

6.2 Integration Tests
def test_end_to_end_pipeline():
    # Submit document for processing
    job_id = submit_document(iiif_url)
    
    # Wait for completion
    result = wait_for_job(job_id, timeout=300)
    
    # Verify results
    assert result['status'] == 'completed'
    assert len(result['regions']) > 0
    assert all(r['text'] for r in result['regions'])
    
    # Check quality
    avg_confidence = np.mean([r['confidence'] for r in result['regions']])
    assert avg_confidence > 0.7

6.3 Load Testing
Scenarios:
1. Single Document: 1 user, 1 document
    * Expected: < 5 seconds end-to-end
2. Moderate Load: 10 concurrent users, 100 documents
    * Expected: < 10 seconds per document
3. High Load: 100 concurrent users, 1000 documents
    * Expected: Queue processing, < 60 seconds per document
4. Stress Test: 1000 concurrent users
    * Expected: Graceful degradation, no crashes
Tools: Locust, JMeter, or Artillery

Phase 7: Documentation & Handoff (Week 15)
7.1 Technical Documentation
Deliverables:
1. API Documentation (OpenAPI/Swagger)
2. Model Cards (Stage 1 & Stage 2)
3. Deployment Guide (Docker, Kubernetes)
4. Training Guide (How to retrain models)
5. Troubleshooting Guide
6. Performance Tuning Guide

7.2 User Documentation
Deliverables:
1. Quick Start Guide
2. API Usage Examples
3. Best Practices
4. FAQ
5. Video Tutorials

Risk Assessment & Mitigation
Technical Risks
Risk	Impact	Probability	Mitigation
IIIF server downtime	High	Medium	Caching, retry logic, fallback servers
Model accuracy below target	High	Low	Larger training set, ensemble methods
Slow inference speed	Medium	Medium	Model optimization, GPU scaling
Memory issues with large images	Medium	Low	Streaming, tiling, compression
Data privacy concerns	High	Low	On-premise deployment option
Business Risks
Risk	Impact	Probability	Mitigation
Training data availability	High	Medium	Partner with libraries/archives
Annotation quality issues	Medium	Medium	Multi-annotator validation
Cost overruns	Medium	Low	Cost monitoring, optimization
Adoption challenges	Medium	Medium	Good documentation, support
Success Metrics
Technical KPIs
* Stage 1 mAP: > 0.75
* Stage 2 CER: < 5%
* End-to-end latency: < 10 seconds per document
* System uptime: > 99.5%
* Cache hit rate: > 70%
Business KPIs
* Documents processed: > 100K/month
* API adoption: > 50 active users
* Cost per document: < $0.02
* User satisfaction: > 4.0/5.0

Timeline Summary
Phase	Duration	Deliverables
1. Core Endpoint	2 weeks	Enhanced masking API
2. Stage 1 Model	4 weeks	Region detection model
3. Integration	2 weeks	Pipeline orchestration
4. Stage 2 Model	2 weeks	OCR model
5. Deployment	2 weeks	Production system
6. Testing	2 weeks	Validated pipeline
7. Documentation	1 week	Complete docs
Total	15 weeks	Production ML pipeline
Next Steps
Immediate (Week 1)
1. ✅ Set up development environment
2. ✅ Deploy basic masking endpoint
3. ✅ Test with Library of Congress images
4. 🔲 Gather training data for Stage 1
Short-term (Weeks 2-4)
1. 🔲 Implement batch processing
2. 🔲 Add annotation format support
3. 🔲 Begin Stage 1 model training
4. 🔲 Set up annotation workflow
Medium-term (Weeks 5-10)
1. 🔲 Complete Stage 1 training
2. 🔲 Generate Stage 2 training data
3. 🔲 Train Stage 2 model
4. 🔲 Build integration layer
Long-term (Weeks 11-15)
1. 🔲 Production deployment
2. 🔲 Performance optimization
3. 🔲 Documentation
4. 🔲 User onboarding

Resources Required
Team
* 1x ML Engineer (full-time)
* 1x Backend Engineer (full-time)
* 1x DevOps Engineer (part-time)
* 1x Data Annotator (part-time)
Infrastructure
* 4x GPU servers (V100 or A100)
* Redis/RabbitMQ servers
* S3/object storage
* CI/CD pipeline
Budget
* Infrastructure: $2,000/month
* Training data annotation: $5,000 one-time
* External services (IIIF, monitoring): $500/month
* Total: ~$7,500 first month, ~$2,500/month ongoing

Conclusion
This pipeline represents a significant advancement in document analysis ML workflows. By using Stage 1's accurate polygon predictions to create clean, masked training data for Stage 2, we eliminate background noise and improve model accuracy. The IIIF polygon masking endpoint is the key innovation that enables this two-stage approach.
Expected Outcomes:
* 15-25% improvement in Stage 2 OCR accuracy vs. traditional approaches
* More robust models that generalize better to varied layouts
* Cleaner training data with reduced annotation overhead
* Scalable architecture ready for production workloads
