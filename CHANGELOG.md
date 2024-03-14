## 0.0.3 - 2024-03-14

### Fixed
- Corrected the subsetting algorithms within BaseModelers
- Adjusted the computation and typing of select variables created from build_reserve_cols()


## 0.0.2 - 2022-04-01

### Added
- Requirements.txt files (both full and top-level requirements)

### Fixed
- Updated (the now deprecated) name for MMLSpark package (mmlspark -> synapse.ml)

### Changed
- Replaced code that relies on Koalas with pyspark.pandas API
- Slight formatting changes to build_package.bat file

## 0.0.1 - 2021-08-30

### Added

- File Architecture
- PanelDataProcessor
- LGBSurvivalModeler
- GBTSurvivalModeler
- RFModeler (Random Forest)
- Data simulation through create_example_data() function
- ReadTheDocs website for documentation
- REIGN and simulated data example notebooks