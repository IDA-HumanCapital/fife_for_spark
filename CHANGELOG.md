## 0.0.1 v2 - 2021-09-01

### Added

- 'CACHE' configuration parameter, stored in ArgumentParser

### Fixed

- Improved the efficiency of computing evaluation metrics by performing the following changes:
    - Added cache toggles to the compute_binary_evaluation_metrics() function
    - Switched the function that computes AUROC to one that takes PySpark DataFrame elements as inputs
    - Brought 'total' variable calculation outside of compute_binary_evaluation_metrics() function

### Changed
- Updated REIGN example notebook with most recent functions
    - Limited the outputs of the display() function
- Fixed spelling errors in documentation. Edited the below documents:
    - user_guide
    - spark_help

## 0.0.1 - 2021-08-30

### Added

- File Architecture
- PanelDataProcessor
- LGBSurvivalModeler
- Data simulation through create_example_data() function
- ReadTheDocs website for documentation
- REIGN and simulated data example notebooks

### Fixed



### Changed

