<a name="readme-top"></a>

<h1 align="center">Predicting Urban River Water Quality from Sentinel-2 Catchment Characteristics</h1>
<h3 align="center">A Machine Learning Approach Applied to the River Roding, East London</h3>

<p align="center">
  <img src="figure_for_presentation/roding-ilford-scaled.jpeg">
</p>

<p align="center">
  <em>Figure 1. The River Roding at Ilford, East London — representative of the urban river corridor investigated in this project.</em>
</p>

**Course:** GEOL0069 – Artificial Intelligence for Earth Observation  
**Author:** James Ge  
**Institution:** University College London


---

<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li><a href="#project-overview">Project Overview</a></li>
    <li><a href="#background-and-motivation">Background and Motivation</a></li>
    <li><a href="#research-questions">Research Questions</a></li>
    <li><a href="#methodology">Methodology</a>
      <ul>
        <li><a href="#study-area">Study Area</a></li>
        <li><a href="#field-data">Field Data</a></li>
        <li><a href="#sentinel-2-data">Sentinel-2 Data</a></li>
        <li><a href="#spectral-indices">Spectral Indices</a></li>
        <li><a href="#machine-learning-approaches">Machine Learning Approaches</a></li>
        <li><a href="#validation-strategy">Validation Strategy</a></li>
      </ul>
    </li>
    <li><a href="#notebook-structure">Notebook Structure</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#environmental-impact">Environmental Impact</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#limitations">Limitations</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

---

## 🏠Project Overview

This project investigates whether Sentinel-2 Earth Observation data and machine learning can explain spatial variability in water quality along the River Roding, East London. It builds directly on previous field-based hydrochemical mapping carried out as part of the GEOL0024 Environmental Mapping module, during which in-situ measurements of pH, electrical conductivity (EC), and nitrate concentration, alongside ICP-OES elemental analysis of major ions and trace elements at 15 sites. These were collected across two seasons.

The River Roding is too narrow (~10–30m) to be directly imaged at Sentinel-2's 10m resolution. Rather than attempting to image the channel itself, this project extracts spectral and land-cover characteristics from the **surrounding catchment environment** of each sampling location and uses these within a machine learning regression framework to predict water quality parameters — principally EC and sodium concentration (Na).

A key methodological decision separates sites strongly influenced by Thames estuarine backwash (EC > 1800 µS/cm) from the freshwater training set, using them instead as an **out-of-domain evaluation dataset**. This mirrors the cross-ecosystem generalisation approach used in remote sensing ML literature, and is physically motivated by the ICP-OES evidence for saline intrusion at downstream sites.

---

## 💡Background and Motivation

This project is inspired by my water quality mapping study of the River Roding conducted as part of the Environmental Mapping module under Dr. Alex Lipp, during which I mapped the London reach of the river through in-situ measurements of pH, electrical conductivity, and nitrate, alongside laboratory ICP-OES analysis of major ion and trace element concentrations at 15 sites. That fieldwork deepened my understanding of the controls on river chemistry, with anthropogenic land use emerging as one of the most significant drivers.

For this AI4EO project, I explored how machine learning applied to Sentinel-2 satellite imagery could extend those findings beyond what fieldwork alone can reveal. Earth observation is particularly powerful for characterising land use and land cover change. Since the transition from suburban to urban and industrial landscapes strongly influences ionic load and nutrient concentrations in rivers, satellite catchment data is scientifically valuable. By combining my in-situ measurements with machine learning on Sentinel-2 spectral features, this project aims to produce a comprehensive evaluation of how catchment land use drives river chemistry, and whether satellite-based methods can usefully predict water quality parameters in narrow urban rivers such as the Roding.

---

## 🧐Research Questions

1. Can Sentinel-2-derived catchment characteristics predict electrical conductivity and sodium concentration along the River Roding?
2. Which spectral and land-cover features contribute most strongly to prediction performance, as revealed by SHAP analysis?
3. Does model performance differ between summer low-flow and winter high-flow conditions?
4. Can a freshwater-trained machine learning model generalise to estuarine sites influenced by Thames tidal backwash?

---

## 🖍Methodology

### Study Area

The study focuses on the London reach of the River Roding — approximately 30 km from Loughton (Essex) to Barking Creek (East London) where it joins the Thames. This corridor passes through a strong urban gradient: semi-natural woodland at Loughton, suburban residential areas in Redbridge, and heavily urbanised and industrial landscapes near Barking. This gradient is the core scientific signal this project tries to capture from space.

38 sampling locations across two seasons (summer 2025 low-flow; winter 2025/26 high-flow) are used. 15 of these sites also have ICP-OES laboratory analysis providing Na, Ca, K, Mg, S, and Sr concentrations.

### Field Data

| Parameter | Method | n sites | Seasons |
|-----------|--------|---------|---------|
| EC (µS/cm) | Handheld probe | 38 | Summer + Winter |
| pH | Handheld probe | 38 | Summer + Winter |
| Temperature (°C) | Handheld probe | 38 | Summer + Winter |
| NO₃⁻ (ppm) | Colorimetric strip | 37 | Summer + Winter |
| Na, Ca, K, Mg, S, Sr (ppm) | ICP-OES laboratory | 15 | Summer + Winter |

**Estuarine split:** Sites with EC > 1800 µS/cm are flagged as Thames backwash-influenced and excluded from model training. ICP-OES data confirms saline intrusion at these sites through elevated Na, Mg, S, and Sr concentrations.

### Sentinel-2 Data

Sentinel-2 Level-2A imagery (atmospherically corrected surface reflectance) is downloaded from the Copernicus Data Space using the authentication and download functions introduced in GEOL0069 Week 3. Two scenes are acquired:

| Scene | Date window | Purpose |
|-------|-------------|---------|
| Summer | August–October 2025 | Matches low-flow sampling period |
| Winter | December 2025–January 2026 | Matches high-flow sampling period |

The full tile (100×100 km, T30UXC) is cropped to the Roding catchment bounding box before processing, reducing memory requirements substantially.

**Bands used:**

| Band | Wavelength | Resolution | Use |
|------|-----------|------------|-----|
| B02 | Blue 490 nm | 10 m | RGB, NDWI |
| B03 | Green 560 nm | 10 m | RGB, NDWI |
| B04 | Red 665 nm | 10 m | RGB, NDVI |
| B08 | NIR 842 nm | 10 m | NDVI, NDWI |
| B11 | SWIR1 1610 nm | 20 m | NDBI |

### Spectral Indices

Three spectral indices are computed from the bands above and used as the primary features in regression:

**NDVI (Normalised Difference Vegetation Index)**
$$\text{NDVI} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}}$$
Higher values indicate denser vegetation. Inverse proxy for impervious surface cover.

**NDWI (Normalised Difference Water Index)**
$$\text{NDWI} = \frac{\text{B03} - \text{B08}}{\text{B03} + \text{B08}}$$
Detects open water and moisture. Higher values near water bodies.

**NDBI (Normalised Difference Built-up Index)**
$$\text{NDBI} = \frac{\text{B11} - \text{B08}}{\text{B11} + \text{B08}}$$
Highlights impervious and built-up surfaces. Expected to be the strongest EC predictor because impervious surfaces increase ionic runoff into rivers.

The final feature set used in regression (`FEATURE_COLS`) comprises 7 physically interpretable variables: summer NDVI, NDWI, NDBI; winter NDVI, NDWI, NDBI; and a distance rank proxy (upstream position along the river).

### Machine Learning Approaches

#### Supervised Regression — Random Forest (main model)
Random Forest Regressor with 200 estimators. Chosen for its ability to handle non-linear relationships between spectral features and water chemistry, and for providing feature importance scores used in SHAP analysis.

Applied to three targets:
- **EC** (n=38 freshwater sites) — expected to be predictable from impervious surface features
- **Na** (n=15 ICP-OES sites) — expected similar pattern but weaker signal
- **pH** (n=38 freshwater sites) — expected non-result, confirming carbonate buffering as investigated during my water quality mapping project

#### Supervised Regression — Ridge Regression (baseline)
Linear baseline is used to assess whether the Random Forest's non-linear capability provides meaningful improvement.

#### Unsupervised — K-Means Clustering
K-means applied to the summer Sentinel-2 bands (B04, B03, B02, B08) over the full Roding catchment, producing 4 land-cover clusters. Cluster labels are then sampled at each field site to test whether sites in more urbanised clusters show higher EC — an independent, label-free validation of the regression model logic.

#### Explainability — SHAP
SHapley Additive exPlanations applied to the fitted Random Forest models to identify which spectral features drive EC, Na, and pH predictions. Expected to show NDBI dominates EC and Na (impervious surfaces → ionic load), while no feature dominates pH (geology controls buffering capacity regardless of land use).

### Validation Strategy

**Leave-One-Out Cross-Validation (LOO-CV)** is used instead of a fixed train/test split. With only 38 sites, a held-out test set would be unrepresentatively small. LOO-CV is the standard approach for small geochemical datasets: each site takes a turn as the single test point, with all remaining sites used for training. This produces n unbiased predictions which are then compared to observed values.

**Out-of-domain evaluation:** The freshwater-trained EC model is additionally applied to the excluded estuarine sites. The model is expected to underpredict dramatically, as tidal salinity intrusion from the Thames is a process invisible to Sentinel-2 catchment features. This failure is scientifically meaningful rather than problematic, as it demonstrates that the freshwater-trained model captures land-use-driven hydrochemical variability but cannot resolve estuarine tidal mixing processes invisible to Sentinel-2 surface reflectance data.

**Feature ablation:** An additional comparison runs three separate EC models — using spatial position only, EO features only, and both combined — to isolate how much the satellite data contributes beyond a simple upstream-downstream proxy.

**Seasonal comparison:** EC models are fitted separately on summer-only, winter-only, and combined feature sets. If summer features outperform winter, this confirms the EC dilution signal in water quality field investigation (lower flow → more concentrated ionic load → stronger land-cover signal).

---

## 📝Notebook Structure

The project is contained in a single Google Colab notebook with 12 cells:

| Cell | Description | Adapted from |
|------|-------------|------------------|
| 1 | Setup, Drive mount, EnvironmentalCostTracker class | — |
| 2 | Load field data CSV, estuarine split, study area map | — |
| 3 | Copernicus authentication + Sentinel-2 query + download | **Week 3** |
| 4 | Band loading, RGB preview, catchment crop | **Week 3** |
| 5 | Spectral index calculation (NDVI, NDWI, NDBI) | **Week 4/5** |
| 6 | Feature extraction at 38 GPS sampling points | **Week 5** |
| 7 | Regression: RF + Ridge (EC, Na, pH) + feature ablation | **Week 5** |
| 8 | SHAP explainability (bar + beeswarm plots) | **Week 9** |
| 9 | Estuarine out-of-domain evaluation | **Week 5** |
| 10 | K-Means unsupervised land cover classification | **Week 4** |
| 11 | Seasonal comparison (summer vs winter features) | **Week 5** |
| 12 | Environmental cost assessment | — |

---

## 📊Results

### Model Performance Summary (LOO-CV)

| Target | Model | n | R² | RMSE | MAE |
|--------|-------|---|----|------|------|
| EC (µS/cm) | Ridge Regression | 31 | 0.255 | 128.5 | 107.7 |
| EC (µS/cm) | Random Forest | 31 | 0.170 | 135.6 | 111.9 |
| Na (ppm) | Ridge Regression | 11 | 0.009 | 14.2 | 11.5 |
| Na (ppm) | Random Forest | 11 | -0.048 | 14.6 | 12.0 |
| pH | Random Forest | 31 | -0.116 | 0.25 | 0.19 |

The strongest predictive performance was obtained for EC, where both Ridge Regression and Random Forest captured part of the downstream hydrochemical variability associated with urbanisation and Thames influence. Ridge Regression slightly outperformed Random Forest, suggesting the relationship between the selected Sentinel-2 features and EC is relatively low-dimensional and approximately linear. Na concentrations showed weak predictive performance, likely reflecting the small ICP-OES sample size (n=11) and the stronger influence of hydrological mixing processes. pH was effectively unpredictable from EO features, supporting the interpretation that carbonate buffering and geological controls dominate over land-cover effects.

### Feature Ablation (EC)

| Feature set | R² | RMSE | MAE |
|-------------|-----|------|------|
| Spatial only (distance rank) | 0.191 | 133.9 | 112.1 |
| EO only (NDVI, NDWI, NDBI) | 0.015 | 147.7 | 122.5 |
| Combined | 0.170 | 135.6 | 111.9 |

Feature ablation demonstrates that spatial position along the river explains substantially more EC variability than Sentinel-2 EO features alone. The EO-only model produced very weak predictive skill (R² = 0.015), indicating that catchment spectral characteristics contribute only limited additional explanatory power beyond the longitudinal hydrochemical gradient. This result is scientifically important because it highlights both the capability and limitations of Sentinel-2 for narrow urban river systems.

### Seasonal Comparison (EC)

| Feature set | R² | RMSE |
|-------------|-----|------|
| Summer only | 0.057 | 144.5 µS/cm |
| Winter only | -0.113 | 157.1 µS/cm |
| Combined | 0.170 | 135.6 µS/cm |

Summer Sentinel-2 features produced slightly stronger predictive performance than winter imagery, consistent with reduced dilution during low-flow conditions. Winter performance weakened substantially, suggesting that high-flow hydrological conditions reduce the relationship between surface land cover and observed ionic concentration.

### Out-of-Domain Evaluation (Estuarine Sites)

| Domain | R² | RMSE |
|--------|-----|------|
| Freshwater (in-domain) | 0.170 | 136 µS/cm |
| Estuarine (out-of-domain) | -9.210 | 2641 µS/cm |

The freshwater-trained Random Forest model failed completely when applied to estuarine sites influenced by Thames tidal backwash. This failure is scientifically meaningful rather than problematic, as it demonstrates that the model captures land-use-driven freshwater hydrochemistry but cannot resolve tidal salinity intrusion processes that are not directly observable through Sentinel-2 surface reflectance data.

Overall, the project demonstrates that Sentinel-2-derived environmental features can partially explain urban freshwater hydrochemistry, but also highlights important physical limitations associated with narrow river geometry, hydrological mixing, and estuarine tidal influence.

## Key Figures

### Study Area — Sampling Sites
<p align="center">
<img src="figures/sampling_sites_map.png" width="600"/>
</p>
<p align="center"><em>Figure 2. River Roding sampling sites across the London reach. Blue circles indicate freshwater sites used for model training (n=31); red triangles indicate estuarine sites excluded from training and reserved as an out-of-domain evaluation set (n=7).</em></p>

### Sentinel-2 Spectral Indices (Summer & Winter)
<p align="center">
<img src="figures/spectral_indices.png" width="800"/>
</p>
<p align="center"><em>Figure 3. Spectral indices computed from Sentinel-2 Level-2A imagery for summer (October 2025, top row) and winter (January 2026, bottom row) over the Roding catchment. NDVI (left) shows the north-to-south transition from vegetated suburban areas to dense urban surfaces. NDWI (centre) highlights the Thames and its tributaries. NDBI (right) reveals the concentration of impervious built-up surfaces in inner East London.</em></p>

### Model Results — Predicted vs Observed (LOO-CV)
<p align="center">
<img src="figures/predicted_vs_observed.png" width="800"/>
</p>
<p align="center"><em>Figure 4. LOO-CV results for Random Forest regression across three water quality targets. EC (left) shows modest predictive skill driven by the urban-to-rural gradient. Na (centre) performs poorly, reflecting the small ICP-OES sample size (n=11) and stronger hydrological mixing controls. pH (right) is effectively unpredictable from EO features, confirming that carbonate buffering and geological controls dominate over land-cover effects, consistent with the fieldwork findings.</em></p>

### SHAP Feature Importance — Electrical Conductivity
<p align="center">
<img src="figures/shap_beeswarm_ec.png" width="700"/>
</p>
<p align="center"><em>Figure 5. SHAP beeswarm plot for EC prediction. Each dot represents one sampling site. Red dots indicate high feature values, blue dots indicate low feature values. Distance rank dominates. Sites further downstream (high rank, red) receive large positive SHAP values, confirming the urban gradient as the primary EC driver. Winter NDBI shows a secondary positive effect, consistent with impervious surface runoff contributing to ionic load. The spectral indices contribute modestly relative to position, explaining the weak EO-only model performance.</em></p>

### Estuarine Out-of-Domain Evaluation
<p align="center">
<img src="figures/estuarine_ood_evaluation.png" width="800"/>
</p>
<p align="center"><em>Figure 6. Domain generalisation test: the freshwater-trained Random Forest EC model applied to estuarine sites near Barking Creek. Left: in-domain freshwater performance showing modest but positive predictive skill. Right: out-of-domain estuarine performance showing complete model failure. This failure is scientifically expected: Thames tidal backwash drives saline intrusion that is entirely invisible to Sentinel-2 catchment land-cover features, confirming tidal mixing as a distinct hydrological control.</em></p>

### Unsupervised Land Cover Classification (K-Means)
<p align="center">
<img src="figures/kmeans_landcover.png" width="800"/>
</p>
<p align="center"><em>Figure 7. K-Means unsupervised clustering (4 clusters) applied to Sentinel-2 summer bands (B04, B03, B02, B08) over the Roding catchment, shown alongside the RGB reference image. The algorithm separates the scene into distinct land cover types without any labelled training data: red/brown tones in the north correspond to suburban and vegetated area, while green dominates the densely urbanised and impervious surfaces of inner East London in the south. The Thames is visible as a dark green winding feature through the lower portion of the scene. The spatial pattern aligns visually with the RGB composite, confirming the clusters capture meaningful land cover variation rather than noise.</em></p>

---

## 🌐Environmental Impact

Computational carbon was tracked throughout the notebook using a custom `EnvironmentalCostTracker` class that logs energy use and CO₂ per phase. The carbon factor used is 0.5 kg CO₂/kWh (UK grid average).

| Phase | Duration | Energy (Wh) | Carbon (g CO₂) |
|-------|:--------:|:-----------:|:--------------:|
| Data Loading & Study Area | 0.0 min | 0.0 | 0.0 |
| Sentinel-2 Query & Download (Summer) | 1.1 min | 1.4 | 0.7 |
| Sentinel-2 Query & Download (Winter) | 0.6 min | 0.8 | 0.4 |
| Band Loading & RGB Preview | 1.2 min | 1.5 | 0.8 |
| Spectral Index Calculation | 0.3 min | 0.4 | 0.2 |
| Feature Extraction | 0.0 min | 0.0 | 0.0 |
| Regression Models (EC, Na, pH) | 0.3 min | 0.3 | 0.2 |
| SHAP Explainability | 0.1 min | 0.1 | 0.1 |
| Estuarine Out-of-Domain Evaluation | 0.0 min | 0.0 | 0.0 |
| K-Means Unsupervised Clustering | 0.5 min | 0.7 | 0.3 |
| Seasonal Comparison | 0.3 min | 0.5 | 0.2 |
| **Total ML pipeline** | **~0.1 hours** | **6** | **3** |
| **Traditional fieldwork baseline** | — | — | **~25,000** |

**Fieldwork baseline:** Two sampling campaigns along the Roding corridor (Loughton to Barking, ~60 km round trip × 2 seasons × 0.21 kg CO₂/km by car) = approximately 25 kg CO₂. The satellite ML pipeline produced only 3.0 g CO₂ — a 99.9% reduction compared to the estimated 25 kg CO₂ fieldwork baseline.

![Environmental Cost](figures/environmental_cost.png)

---

## 👩‍💻Getting Started

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JAE-G23/Sentinel2-Roding-Water-Quality-ML/blob/main/River_Roding_AI4EO.ipynb)

All code runs in **Google Colab**. Click the badge to open the notebook directly.

### Prerequisites

1. A free [Copernicus Data Space](https://dataspace.copernicus.eu) account is required for satellite data download. Register before running Cell 3.
2. Field data CSV (`field_measurements.csv`) must be uploaded to Google Drive at the path specified in Cell 2.

### Installation

```bash
git clone https://github.com/JAE-G23/Sentinel2-Roding-Water-Quality-ML.git
```

The notebook installs all required packages automatically at Cell 1:

```python
!pip install pysheds contextily codecarbon rasterio shap
```

Standard packages (NumPy, pandas, scikit-learn, matplotlib, seaborn) are pre-installed in Colab.

### Running the Notebook

Run cells sequentially from Cell 1. Cell 3 will prompt for Copernicus credentials interactively using `getpass` — credentials are never stored in the notebook.

---

## 🔒Limitations

- **Small sample size:** n=38 field sites and n=15 ICP-OES samples constrains model complexity and statistical power. LOO-CV mitigates this but does not eliminate it.
- **Point-based spectral sampling:** Sentinel-2 features are extracted at the GPS coordinate of each sampling location rather than from hydrologically delineated upstream sub-catchments or spatial buffers. This captures the immediate bankside spectral environment but does not fully resolve upstream transport pathways.
- **Narrow river channel:** The Roding is narrower than a single Sentinel-2 pixel at most sampling locations. Spectral values at the sampling points reflect bankside land cover rather than the water surface itself — this is a deliberate methodological choice, not a limitation.
- **Two scenes only:** Imagery for one summer and one winter scene is used. A multi-year or multi-date composite would produce more robust spectral features.
- **Estuarine sites:** The Thames tidal backwash signal cannot be captured by any land-cover satellite feature. This is clearly demonstrated in Cell 9 and is a finding rather than a flaw.

---

## 📚References

Environment Agency. Lower Roding (Loughton to Thames) Water Body – Catchment Data Explorer. https://environment.data.gov.uk/catchmentplanning/WaterBody/GB106037028181

Kaushal, S.S., et al. (2018). Freshwater salinization syndrome on a continental scale. *PNAS*, 115(4), E574–E583. https://doi.org/10.1073/pnas.1711234115

Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Neal, C. (2001). Alkalinity measurements within natural waters: towards a standardised approach. *Science of the Total Environment*, 265(1–3), 99–113.

Pennino, M.J., et al. (2016). Stream restoration and sewers impact sources and fluxes of water, carbon, and nutrients in urban watersheds. *Hydrology and Earth System Sciences*, 20, 3419–3439.

Thames21 (2024). Citizen Science Water Quality Monitoring Programme: Technical Report 2024 – Reclaim Our Roding. https://www.thames21.org.uk/

Walsh, C.J., et al. (2005). The urban stream syndrome: current knowledge and the search for a cure. *Journal of the North American Benthological Society*, 24(3), 706–723.

*GEOL0069 AI for Earth Observation Course Materials* — Dr Michel Tsamados and Weibin Chen, UCL Earth Sciences.

---

## 📬Contact

**James Ge**  
Email: james.ge.23@ucl.ac.uk  
Institution: University College London  
Course: GEOL0069 – AI for Earth Observation

Project Link: https://github.com/JAE-G23/Sentinel2-Roding-Water-Quality-ML

---

## 🎓Acknowledgements

This project was completed as part of GEOL0069 at University College London, taught by Dr Michel Tsamados and Weibin Chen. The field data underpinning this project was collected during GEOL0024 Environmental Mapping under Dr. Alex Lipp. The notebook adapts code and concepts from the GEOL0069 weekly materials, applied to a new research context.

---

## 📜Disclaimer

This repository contains coursework developed as part of the module GEOL0069(AI4EO) in the UCL Earth Sciences Department.

The content represents my personal academic work and understanding. It is not an official course resource. Although care has been taken to ensure correctness, the material may not reflect future updates to course content.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
