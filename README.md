# A Statistical Shape Modeling Pipeline for Assessing Arrhythmic Risk in Mitral Valve Disease Patients

This repository contains code and data processing workflows for the project "A Statistical Shape Modeling Pipeline for Assessing
Arrhythmic Risk in Mitral Valve Disease Patients", analyzing left ventricular (LV) geometry and systolic motion in patients with mitral valve prolapse (MVP) and mitral annular disjunction (MAD). The project was developed during the **Simula Summer School in Computational Physiology 2025**.  

---

## Project Overview

This pipeline demonstrates that principal component analysis (PCA)-derived shape and motion modes can capture clinically relevant remodeling in MVP/MAD, offering a complementary approach for arrhythmic risk assessment.

---

## Contributors

- **Ingvild Askim Adde** – Kristiania University of Applied Sciences, Norway  
- **Eva Schuijt** – Department of Physiology, Maastricht, the Netherlands  
- **Diana Vucevic** – Department of Bioengineering, UC San Diego, USA  
- **Nina Ziegenbein** – Steno Diabetes Center Aarhus & Department of Public Health, Aarhus University, Denmark  

*All authors contributed equally.*  

**Supervised by:** Guilia Monopoli, Nickolas Forsch, Molly Maleckar (Simula Research Laboratory, Oslo, Norway)

---

## Dataset

- **Source:** Oslo University Hospital, in collaboration with the ProCardio Centre for Research-Based Innovation.  
- **Cohort:** 91 patients with clinically confirmed MAD, 75% with possible concomitant MVP.  
- **Demographics:** 59 males, 32 females; ages 18–78.  
- **Imaging:** Cine CMR in SAX view (~30 frames per cardiac cycle).  
- **Clinical data:** Standard measurements and disease status for each patient.  

Additional details: [Aabel et al., 2021][1].

[1]: https://doi.org/10.1007/s10554-021-02288-4

---

## Repository Structure & Workflow

### 1. Segmentation Pipeline (`segmentation_pipeline.ipynb`)

This module handles preprocessing, cleaning, and segmentation of the imaging data:

- Combines multiple data sources with differing folder structures.  
- Uses CSV files (`ED_Prev_Segmentation.csv`, `ED_slices_and_timepoints.csv`, `newDataSegmentation.csv`) to determine ED timepoints and slices (base to apex).  
- Applies LV segmentation using the **MONAI** model ([credit](https://huggingface.co/MONAI/ventricular_short_axis_3label)):  
  - Resizes and converts images to NumPy arrays.  
  - Saves the ED and included slices as **NIfTI stacks** and **HDF5 files**.  
- Direct segmentation on NIfTI files (bottom of the script) was tested but not used due to lower performance.

---

### 2. Point Cloud Generation, Alignment, PCA Analysis

We use **saxomode** ([documentation](https://computationalphysiology.github.io/MAD-SSA/README.html)) and `myo_cavity.py` (Giulia Monopoli) to convert segmentation masks into 3D point clouds:

- Automatic identification of LV endocardial and epicardial boundaries.  
- Conversion to real-world coordinates using voxel dimensions.  
- Third-order B-spline fitting to smooth contours and correct for SAX breath-hold misalignment.  
- Sampling: 20 SAX planes × 40 points + 1 apical point → 1,602 points per patient.  
- Contours are visually inspected; smoothing factor adjusted if needed.  
- Bash scripts provided to run commands for point cloud generation.
- Afterwards alignment of point clouds, analysis using PCA for both ED an motion (ES-ED) scenarios  

---

### 3. PCA Analysis & Clinical Correlation

Performed in `ED_analysis.ipynb` and `motion_analysis.ipynb`, with utility functions in `analyze_modes.py`:

- LV point clouds are aligned to a **common anatomical coordinate system**:  
  - Origin at epicardial center-of-mass.  
  - Z-axis along the LV base–apex axis.  
  - Y-axis toward RV; X-axis defined orthogonally.  
- Height normalization applied to reduce shape differences due to patient height.  
- Ensures one-to-one anatomical correspondence across subjects.  
- Derived PCA modes are correlated with clinical variables for both ED shape and systolic motion.

---

### Virtual environment setup

You can create a virtual environment and install all required packages using `pipenv`. If you don't already have `pipenv` installed you can install it by running the following command:

```
pip install --user pipenv
```

Make sure you have both `python` and `pip` installed before running the command. Full documentation of `pipenv` installation can be found here: [https://pipenv.pypa.io/en/latest/installation.html](URL).


To install and activate the virtual environment run the following command in your terminal where the `Pipfile` and `Pipfile.lock` are located:

```
pipenv install
```

If you just want to activate the environment without re-running installation you can run

```
pipenv shell
```

You can add new packages to the virtual environment in the following way:

```
pipenv install <package-name>
```