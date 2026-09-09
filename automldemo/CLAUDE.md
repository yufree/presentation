# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **democode**, a repository containing various data analysis tools for environmental science and metabolomics research. The repository consists of multiple independent projects and scripts, primarily using **R** and **Python**.

### Key Subprojects

#### 1. **spectral_lib_matcher/** - Mass Spectrometry Spectral Matching
- Python project for matching mass spectrometry spectra
- Uses the `matchms` library for similarity calculations
- Supports MGF file processing and binary library formats
- Has its own Docker/conda environment configuration
- **Command**: `python src/processor.py [options] query.mgf database.mgf`

#### 2. **thermoflask/** - Thermo RAW File Processing Web App
- Flask-based web application
- Processes Thermo `.raw` files using ThermoRawFileParser
- **Build & Run**:
  ```bash
  docker build -t thermoflask .
  docker run -p 5000:5000 thermoflask
  ```

#### 3. **aisummary/** - AI/ML Paper Screening
- Python scripts for automated paper screening and analysis
- Contains presentation generation tools
- **Run**: Use Python directly (requires virtual environment in `venv/`)

#### 4. **llm/** - LLM Integration Scripts
- OpenAI-based tools for research automation
- `update.py`: PubMed RSS feed analysis with scoring
- `title.py`, `score.py`, `rsssummary.py`: Various LLM utilities
- **Note**: Contains hardcoded OpenAI API keys

### Common Directories

- **plot/**: R scripts for data visualization (heatmaps, PCA, mass spectra, etc.)
- **meta/**: R scripts for metabolomics analysis (XCMS, mass defect, peak alignment)
- **data/**: Various datasets (CSV, MGF, mzML files)
- **rreport/**: R-based reporting templates and tools
- **serum/**, **dbs/**: Processed metabolomics datasets

### Root-Level Scripts

- `pmc.py`, `pmc2.py`: EuropePMC API queries for publication analysis
- `video_to_english_srt.py`: Video subtitle translation
- `mailbox*.py`: Email processing scripts
- Various `.R` files for specific analyses

## Development Commands

### R Scripts
```bash
# Run R scripts
Rscript script.R

# RStudio recommended for interactive development
# Install packages as needed (xcms, xsetplus, xMSanalyzer, etc.)
```

### Python Scripts
```bash
# Standard Python execution
python script.py

# For subprojects with venv:
source venv/bin/activate  # or venv\Scripts\activate on Windows
python script.py

# For spectral_lib_matcher:
conda env create -f environment.yml
conda activate spectral_lib_matcher
python src/processor.py ...
```

### Docker Projects
```bash
# thermoflask
docker build -t thermoflask thermoflask/
docker run -p 5000:5000 thermoflask

# spectral_lib_matcher
docker build -t spectrallibmatcher spectral_lib_matcher/
docker run -it --rm -v $PWD:/app spectrallibmatcher bash
```

## Architecture Notes

- **No unified build system**: Each subproject is independent
- **Mixed languages**: R for statistical analysis, Python for ML/web tools
- **Data-heavy**: Large CSV files, mass spectrometry formats (MGF, mzML, RAW)
- **Multiple Docker images**: Each subproject may have its own Dockerfile
- **No centralized testing**: Individual projects handle their own tests

## Important Data Files

- **automldemo/MTBLS28negmzrt.csv** & **MTBLS28posmzrt.csv**: Metabolomics datasets
- **data/**: Various environmental and metabolomics datasets
- **test.mgf**, **dda.mgf**, **dbs.mgf**: Mass spectrometry spectral files

## Common Workflows

1. **Mass Spectrometry Analysis**:
   - Use `meta/xcms*.R` scripts for LC-MS data processing
   - `spectral_lib_matcher/` for spectral matching
   - `thermoflask/` for RAW file conversion

2. **Environmental Data Analysis**:
   - Scripts in `plot/` for visualization
   - Data files in `data/` directory
   - R-based statistical analysis

3. **Literature Analysis**:
   - `pmc.py` for publication counts
   - `llm/` scripts for automated analysis

## Key Dependencies

### R Packages
- xcms, xsetplus, xMSanalyzer (metabolomics)
- ggplot2, plotly (visualization)
- sva (batch effect correction)

### Python Packages
- matchms (spectral matching)
- flask (web framework)
- openai (LLM integration)
- feedparser (RSS parsing)

## Notes

- Large data files (hundreds of MB) are present
- Some scripts contain hardcoded credentials (e.g., OpenAI API keys in `llm/update.py`)
- Each subproject has its own documentation (check subproject README files)
- The repository is a collection of tools rather than a single application
