# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a QGIS plugin called "YOLO Interactive Object Detector" that allows users to create custom YOLO object detection models through interactive annotation directly on QGIS canvas. The plugin uses deep learning (YOLO) for object detection combined with interactive annotation tools to train specialized models for geospatial data.

**Current Version**: 1.6.6 (from metadata.txt) - Note: Directory shows v1.5.7 but actual version is 1.6.6
**Architecture**: Production-ready with Smart AI Assistant integration (YOLO→SAM pipeline)

## Development Commands

### Plugin Installation & Testing
- **Install plugin**: Copy the `qgis_yolo_detector` folder to QGIS plugins directory (`~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`)
- **Auto-install**: Run `install_plugin.bat` (Windows) for automatic installation from project root
- **Reload plugin**: Use `Plugin Reloader` plugin in QGIS or `python reload_plugin.py`
- **Package plugin**: `python package_plugin.py [version]` (creates timestamped ZIP with auto-versioning from project root)
- **Create icon**: `python create_icon.py` (generates plugin icon)

### Version & Build Commands
- **Check current version**: Read from `metadata.txt` line 23 (`version=1.6.6`)
- **Build package**: From project root: `python package_plugin.py` (auto-increments and timestamps)
- **Manual versioning**: `python package_plugin.py 1.7.0` (specify version)

### Dependency Management
- **Check dependencies**: Run automatic dependency checker via plugin interface
- **Install dependencies**: `pip install ultralytics torch opencv-python pillow`
- **Test dependencies**: `python test_dependencies.py`
- **Diagnose storage**: `python diagnose_storage.py` (checks database and file structure)

### Testing Commands
- **Test complete workflow**: `python test_complete_workflow.py`
- **Test detection layer**: `python test_detection_layer.py`
- **Test patch extraction**: `python test_patch_extraction.py`
- **Test class synchronization**: `python test_class_sync.py`
- **Reload plugin code**: `python reload_plugin.py`

## Architecture Overview

### Core Components

**Plugin Entry Point**
- `plugin_main.py`: Main plugin class `YOLODetectorPlugin` that initializes UI, handles QGIS integration
- `__init__.py`: Plugin factory function for QGIS plugin system

**Core Engine Modules**
- `core/yolo_engine.py`: YOLO model management with LRU cache, training, and inference with GPU/CPU optimization
- `core/annotation_manager.py`: SQLite-based persistent storage for annotations with geospatial metadata and AnnotationExample dataclass
- `core/smart_annotation_engine.py`: AI-assisted annotation using YOLO→SAM pipeline with advanced candidate selection
- `core/raster_extractor.py`: Extracts image patches from QGIS raster layers with coordinate mapping (RasterPatchExtractor class)
- `core/annotation_quality_manager.py`: **v1.6.0+** - Real-time quality tracking and optimization recommendations
- `core/geospatial_training_optimizer.py`: **v1.6.0+** - Specialized training pipeline with geospatial optimizations

**Data Management**
- `core/yolo_dataset_generator.py`: Converts annotations to YOLO training format (images + YOLO labels)
- `core/simple_dataset_generator.py`: Simplified dataset generation for rapid prototyping

**User Interface**
- `ui/main_dialog.py`: Primary QDockWidget with tabbed interface (Classes, Annotation, Training, Detection) - integrated into QGIS UI
- `ui/annotation_tool.py`: Interactive canvas tool for object annotation with Smart Mode support and QGIS canvas integration
- `ui/smart_validation_dialog.py`: Mini-dialog for AI-assisted annotation validation with keyboard shortcuts
- `ui/smart_auto_detection_dialog.py`: Dialog for reviewing automatic detections with visual previews
- `ui/rapid_annotation_interface.py`: **v1.6.0+** - High-productivity annotation interface with batch processing

**Utilities**
- `utils/dependency_installer.py`: Automatic installation of Python dependencies (PyTorch, Ultralytics, OpenCV)

### Data Flow Architecture

1. **Annotation Workflow**:
   - User creates object classes via main dialog
   - Interactive annotation tool captures click events on QGIS canvas
   - Raster extractor extracts image patches with geospatial coordinates
   - Annotation manager stores examples in SQLite with full metadata
   - Smart annotation engine optionally enhances annotations using AI

2. **Training Pipeline**:
   - Dataset generator converts stored annotations to YOLO format
   - YOLO engine handles transfer learning from pre-trained models
   - Trained models are stored with metadata linking to datasets
   - Training progress tracked via Qt signals and callbacks

3. **Detection Workflow**:
   - YOLO engine loads trained models for inference
   - Batch processing extracts patches from large raster datasets
   - Results converted to QGIS vector layers with geospatial accuracy
   - Multiple detection modes: manual, smart-assisted, and automatic

### Key Design Patterns

**Plugin Architecture**
- QDockWidget-based main interface integrated into QGIS UI (`Qt.RightDockWidgetArea`)
- Signal/slot pattern for component communication between UI and core modules
- Lazy loading of heavy dependencies (PyTorch, YOLO models) with graceful fallbacks
- Plugin lifecycle management with proper cleanup on unload

**Data Management**
- SQLite database for persistent annotation storage with structured schema
- AnnotationExample dataclass for type-safe annotation handling
- LRU cache for YOLO model management (configurable maxsize=3)
- Separation of image data (files) and metadata (database)
- Pathlib.Path usage for cross-platform file handling

**Performance Optimization**
- CPU profiling with automatic hardware-based optimization (psutil integration)
- Batch processing with configurable batch sizes based on available RAM
- Memory management with explicit cleanup for GPU resources (torch.cuda.empty_cache)
- Offline-first architecture to avoid network dependencies (environment variables set globally)

**AI Integration - ENHANCED v1.6.0**
- Multi-model pipeline: YOLO for object detection + SAM for refinement
- **NEW**: Advanced candidate selection with geospatial metrics (centrality, aspect ratio)
- **NEW**: Adaptive SAM refinement decisions based on object type and context
- **NEW**: Intelligent model selection for geospatial domains
- Confidence-based decision making for automatic vs manual annotation
- Performance profiles adapting to hardware capabilities (CPU cores, RAM)

**Quality Management - NEW v1.6.0**
- Real-time annotation session tracking
- Productivity metrics and optimization recommendations  
- Continuous improvement feedback loops
- Batch validation for high-density areas

## Development Guidelines

### Code Organization
- Core ML/AI logic in `core/` modules (engine, annotation, training)
- User interface components in `ui/` modules (dock widgets, dialogs, tools)
- Utility functions in `utils/` modules (dependencies, file handling)
- Model storage in `models/` with pretrained and custom subdirectories
- All file paths use `pathlib.Path` for cross-platform compatibility

### Error Handling
- Graceful fallbacks when dependencies unavailable (try/except with AVAILABLE flags)
- Comprehensive logging with emojis for visual debugging (`print` statements with status icons)
- User-friendly error messages via QGIS QMessageBox system
- Import error handling with conditional feature availability

### Security & Stability
- **STRICT OFFLINE MODE**: No automatic model downloads - all models must be pre-downloaded and included
- Models stored in `models/pretrained/` (YOLO) and `models/sam/` (SAM) directories
- Secure path handling to avoid directory traversal (use project directory or secure plugin data directory)
- NumPy 2.x compatibility with warning suppression
- Robust exception handling in all critical paths with proper cleanup
- **Model Security**: Extension validates model file existence before loading, never downloads from internet
- **Complete Model Inventory**: 5 pre-included models (3 YOLO + 2 SAM) totaling ~128MB for full offline functionality
- **Model Paths**: `models/pretrained/` (YOLO: yolo11n.pt, yolo11s.pt, yolo11m.pt) and `models/sam/` (FastSAM-s.pt, mobile_sam.pt)

### Database Schema
- `annotations`: Core annotation storage with geospatial metadata (bbox_map, bbox_normalized, CRS, pixel_size)
- `classes`: Object class definitions with visual styling and statistics
- `datasets`: Generated YOLO datasets with configuration and training parameters
- `trained_models`: Trained model registry with performance metrics and file paths

### Plugin Integration
- Uses QGIS Processing Framework provider pattern
- Integrates with QGIS coordinate reference systems
- Respects QGIS project structure for data organization
- Canvas tools follow QGIS interaction patterns

### Performance Considerations
- Models cached using LRU strategy to balance memory and performance (OrderedDict-based LRUCache class)
- Automatic batch size calculation based on available hardware (psutil memory detection)
- Progressive enhancement: basic functionality works on any hardware, advanced features require GPU
- Smart annotation engine with adaptive thresholds based on CPU profile
- Memory cleanup with garbage collection and GPU cache clearing

### Packaging & Deployment
- **Automated packaging**: `package_plugin.py` creates timestamped ZIP files with auto-versioning
- **Version management**: Reads from `metadata.txt`, auto-increments build numbers  
- **Exclusion patterns**: Automatically excludes development files (`__pycache__`, tests, logs)
- **Installation script**: `install_plugin.bat` for automatic Windows deployment
- **Plugin structure**: Standard QGIS plugin structure with metadata.txt configuration

### Plugin Lifecycle
- **Initialization**: `YOLODetectorPlugin.__init__()` sets up offline mode and plugin paths
- **GUI Setup**: `initGui()` creates dock widget and integrates with QGIS interface
- **Processing Integration**: Optional Processing Framework provider for algorithm integration
- **Cleanup**: `unload()` properly removes UI elements and cleans up resources
- **State Management**: Handles first-start welcome messages and plugin reactivation

This plugin represents a sophisticated integration of modern deep learning techniques with traditional GIS workflows, making AI-powered object detection accessible to QGIS users without requiring machine learning expertise.