<a id="top"></a>

# **Which Method of Machine Learning is the Best?**
## **Evaluating the potential of multiple machine learning approaches for low-cost remote-sensing-only detection of Chinese urban villages**

## **Contents**

- [Project Overview](#project-overview)
1. [Background Information](#1-background-information)
2. [Existing Literature Review](#2-existing-literature-review)
3. [Current Research Gap](#3-current-research-gap)
4. [Research Question](#4-research-question)
5. [Data, Justification of the AOI Selection and Platform Used](#5-data-justification-of-the-aoi-selection-and-platform-used)
6. [Methods](#6-methods)
7. [Results](#7-results)
8. [How to Use this and Repository Structure](#8-how-to-use-this-and-repository-structure)
9. [Environmental Cost Assessment](#9-environmental-cost-assessment)
10. [Discussion and Limitations](#10-discussion-and-limitations)
11. [Future Work](#11-future-work)
12. [Conclusion](#12-conclusion)
- [References](#references)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [Generative AI Statement](#generative-ai-statement)

## **Project Overview**

This project tests whether Chinese urban villages can be detected from free satellite data alone, without any manual labelling. Using only Sentinel-1 SAR, Sentinel-2 optical and texture, and ESA WorldCover context, it builds a 250m grid across ten cities and derives weak training labels automatically from OpenStreetMap. Four machine-learning families (Random Forest, XGBoost, GraphSAGE and a Gaussian Process) are then compared under a strict six-city-train / four-city-test split, so accuracy is always measured on entirely unseen cities. The central aim is to establish whether such a low-cost, annotation-free pipeline can transfer across cities, and to point out which machine method performs best and worth further analysing.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **1. Background Information**

Accompanied by the rapid urbanization process across the world, the inequality of development within modern cities is outstanding. The classical urban model, in which a central business district (CBD) is surrounded by a deteriorating inner 'zone of transition' or slum, was first set out by Burgess back in 1925. However, after the start of the 21st century, the skyrocketing growth rate of countless Global South countries has added new layers to that model. This is especially true in China, one of the fastest growing countries, which has lifted close to 800 million people out of extreme poverty over the past four decades (World Bank, 2022) and where 66.16% of the total population was living in cities by the end of 2023 (National Bureau of Statistics of China, 2024). Throughout this transformation, the inequality within cities has been comparatively neglected and not focused on.

In 2015, the United Nations adopted the 2030 Agenda for Sustainable Development, whose Goal 11 calls on countries to 'make cities and human settlements inclusive, safe, resilient and sustainable' (United Nations, 2015). After mass urbanisation, the inequality within fast developing cities cannot be ignored anymore. In China, the State Council issued its first dedicated guideline on urban-village redevelopment in 2023, initially targeting 35 mega and super-large cities. Since late 2024, this list has extended support to nearly 300 prefecture-level cities (General Office of the State Council, 2023). In 2025, the central authorities further designated the renewal of urban villages as a priority task within a nationwide urban-renewal action plan aiming for major progress by 2030 (General Office of the CPC Central Committee and General Office of the State Council, 2025).

<p align="center"><img src="figures/SDG11.png" alt="Figure 1" width="220"></p>

***Figure 1.** United Nations Sustainable Development Goal 11 — Sustainable Cities and Communities, the policy framing that motivates this project (United Nations, 2015).*

### 1.1 What is Urban Village

An urban village (*chengzhongcun*, literally 'village within the city') is a formerly rural village that has been physically encircled by the rapid outward expansion of a Chinese city (Cao et al., 2025). China runs a dual urban-rural land system, and under this system, such villages keep their collectively owned rural land and their original household-registration based governance, even after the surrounding farmland has been urbanised or privatised. As a result, they sit largely outside the formal urban planning framework (Wang et al., 2009). The outcome is a dense, informal and often high-rise settlement. It supplies cheap rental housing to low-income rural migrants and the urban poor (Zhang, 2011).

### 1.2 Difference between urban village and slum

Although urban villages superficially resemble the slums, favelas and shantytowns of other developing countries, two key differences distinguish them. The first is tenure. Urban villages sit on collectively owned rural land on which villagers hold recognised property rights and build multi-storey 'handshake' housing, rather than squatting illegally on land they do not own (Wang et al., 2009). The second is origin. Urban villages are a deliberate and legally distinct outcome of China's dual urban-rural land system, not a product of state or housing market failure (Zhang, 2011). As a result, they are physically much closer to formal built-up fabric than classical slums are.

### 1.3 Difference between urban village and normal built-up area

Both urban villages and the surrounding formal built-up area appear as continuous, high-density urban fabric, which makes them hard to separate, especially in coarse remote-sensing data. The difference is one of morphology and planning. Formal built-up areas, such as residential compounds, commercial districts and industrial parks, follow planned street grids with building setbacks, wider roads and deliberate green and open space. Urban villages, built incrementally by villagers without unified planning, are far denser. It usually has tightly packed multi-storey 'handshake' buildings separated by narrow alleys, with minimal vegetation or open space (Wang et al., 2009).

### 1.4 What is the issue with urban village?

The first issue is safety. urban villages typically suffer from high public-health risks, frequent building and fire-safety hazards, poor supporting infrastructure and a degraded living environment (General Office of the State Council, 2023).

The second issue is social. As the cheapest foothold in the city, urban villages concentrate low-income rural migrants who, excluded by the household registration system, remain marginalised from formal urban services and welfare provision (Wang et al., 2009).

The third main issue is informational. Urban villages lack reliable, up-to-date geospatial data, which leaves planners without a clear picture of where they are and hinders evidence-based upgrading.

### 1.5 Why are we using satellite image to monitor urban village?

Field surveys and manual digitisation can identify urban villages accurately, but they are slow, expensive and quickly outdated, so they cannot keep pace with rapid redevelopment across many cities (Cao et al., 2025). Satellite imagery offers a scalable alternative, it covers entire cities consistently, is revisited every few days, and, through free public missions such as Sentinel-1 and Sentinel-2, incurs almost no data cost. Earth observation also has an established record of identifying informal settlements from their distinctive built-up texture and morphology (Kuffer et al., 2016), making it well suited to detecting the dense fabric of urban villages at scale. Specifically, the C-band VV and VH backscatter of Sentinel-1 is sensitive to the rough, volumetric scattering of densely packed buildings, while the short-wave infrared bands of Sentinel-2 (B11 and B12) and the grey-level texture derived from them best capture built-up materials and the fine-grained heterogeneity of the village fabric.

### 1.6 Why is machine learning suitable for analysing these satellite images? 

Separating urban villages from formal built-up areas is not a simple threshold problem. The distinction emerges only from subtle, non-linear combinations of many weak signals, such as SAR backscatter, SWIR texture, vegetation indices and urban context, which also vary from city to city. Machine learning is well suited to this setting because it can learn these high-dimensional, non-linear feature interactions directly from data, rather than relying on hand-crafted rules. It also supports cross-city generalisation, precisely the scalability and transferability that current urban-village identification still lacks.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **2. Existing Literature Review:**

Although urban villages are an established topic in urban planning and social science, the remote-sensing literature devoted to identifying them is surprisingly thin. The most recent systematic review, by Cao et al. (2025), records fewer than fifty peer-reviewed remote-sensing studies on Chinese urban villages across roughly two decades, with annual publication volume rarely exceeding five papers per year (Cao et al., 2025). 

![Figure 2](figures/cao_et_al_2025.jpg)

***Figure 2.** Annual number of peer-reviewed remote-sensing publications on Chinese urban villages (2011–2024), coloured by disciplinary field. Reproduced from Cao et al. (2025). The chart shows how thin and recent this literature is: annual output stays at or below two papers until 2019 and only reaches a modest peak of seven in 2022, with GIS & Remote Sensing (orange) dominating throughout.*

By way of contrast, the broader literature on global slum and informal-settlement detection reviewed by Kuffer et al. (2016) holds an order of magnitude more entries, and the follow-up critical review by Mahabir et al. (2018) extends this analysis to high- and very-high-resolution approaches. This volume gap is diagnostic rather than coincidental. The Chinese urban village is morphologically much closer to formal built-up fabric than to favela-style slums, so the standard slum-detection pipelines transfer to it only partially, and a parallel domain-specific literature has therefore developed slowly.

The urban-village-specific literature itself clusters into three methodological families.

**(i) Object-based image analysis (OBIA) on very-high-resolution (VHR) optical imagery.** 

The earliest and longest-running family, originating in the late 2000s as VHR Chinese imagery became available. A representative recent contribution is Zhao et al. (2020), who proposed a partition-based OBIA framework integrating a Random Forest classifier with GaoFen-2 imagery for Guangzhou, reporting an overall accuracy of 90.2% and Cohen’s κ = 0.80 across four re-divided administrative zones. However, these studies operate on commercial sub-metre imagery (typically GaoFen-2 or WorldView, with raw-data costs of roughly USD 10–30 per km²) and their segmentation rules and partition schemes require per-city manual tuning. None of the reviewed OBIA studies tests cross-city transfer, and the partition-based design in particular is explicitly tailored to the administrative geometry of a single region.

**(ii) Single-city semantic-segmentation with hand-labelled training data.** 

Since approximately 2020, the dominant approach has shifted to pixel-wise deep learning. Pan et al. (2020) trained a U-Net on WorldView satellite imagery for Guangzhou and reported strong building-level segmentation. Owusu et al. (2023) extended this line by inserting super-resolution generative adversarial networks (SR-GAN) as a preprocessor for multi-resolution chengzhongcun imagery. However, every paper in this family requires 200–2,000 hand-traced training polygons in a single study city, the annotation cost is rarely reported in the methods but is consistently the dominant cost of the work, and again no cross-city held-out evaluation is performed. The resulting models are therefore best understood as city-specific tools rather than transferable methods.

**(iii) The recent multi-source / multi-city frontier (2024–2025).** 

A new wave of work has begun to push past the single-city bottleneck. Zhang et al. (2024) introduced UV-SAM, an adaptation of Meta’s Segment Anything Model that produces fine-grained boundary masks across two Chinese cities and outperforms earlier baselines on both datasets. A flexible Sentinel-2-only deep-learning framework (Lin et al., 2025) employs a dual-branch encoder that fuses annual and seasonal imagery and reports mIoU = 0.83 with cross-city transfer tested within the Pearl River Delta. Most ambitiously, the CUGUV benchmark (Yao et al., 2025) released the first multi-city hand-curated urban-village dataset covering 15 Chinese cities for two years (2010 and 2020), and the accompanying Segformer model claims robust large-scale identification performance. Similiarly, every paper in this frontier still depends on **hand-curated training labels**. CUGUV in particular reports thousands of hand-traced polygons across its 15-city sample. Although these works demonstrate transfer across handfuls of cities, the labelling effort scales linearly with the number of cities studied, so the fundamental cost barrier of manual digitisation is reduced but not removed.

Looking across all three families, and at the wider slum literature (Kuffer et al., 2016; Mahabir et al., 2018), four limitations comes up. First, **cross-city held-out evaluation is rare**. Many studies still report aggregate accuracy from within-city or weakly separated splits, rather than testing whether the model transfers to entirely unseen cities. Second, most studies still **classify each pixel or grid on its own**. This ignores the fact that urban villages cluster together in space, and spatial graph models that could use this clustering are absent from the reviewed work. Thirdly, **the imbalanced original data** is always ignored. Urban villages usually cover only 1–2% of the urban area. This is rarely treated as a core problem, so a high overall accuracy can still hide poor recall on the village class. Fourth, even the latest cross-city deep-learning methods (Zhang et al., 2024; Lin et al., 2025; Yao et al., 2025) **still need hand-labelled training data for every new city**. It remains unclear whether free, public weak signals could replace manual annotation altogether.

These four structural gaps—single-city evaluation, missing spatial inductive bias, unhandled data imbalance, and persistent dependence on manual labelling—together define the methodological space this project occupies, and motivate the specific research gaps set out below.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **3. Current research gap**

i.  Despite the scale of the phenomenon, urban villages remain comparatively understudied, and the first comprehensive review of urban-village identification in China appeared only recently.

ii.  Chinese urban villages differ fundamentally from the classical slums of other developing countries: built on collectively owned rural land, they are a deliberate product of the dual land system rather than illegal squatting, which makes them an institutionally distinct and more complex phenomenon (Zhang, 2011).

iii.  Methods developed for this setting have wider relevance, because informal and inadequate settlements are growing fastest across the rapidly urbanising Global South, particularly in Asia and sub-Saharan Africa (UN-Habitat, 2022).

iv.  There is still no unified definition of, or consensus method for, identifying urban villages, owing to persistent conceptual fuzziness and their spatial heterogeneity across cities.

v.  Existing identification studies are largely confined to single cities or single images (e.g. Pan et al., 2020), and large-scale, cross-city benchmarks have only begun to emerge (Yao et al., 2025); the cross-city transferability of identification approaches therefore remains largely untested.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **4. Research Question**

Building on the gaps above, this project asks one main question:

> **Can free, public satellite data combined with weak OSM supervision deliver a transferable urban-village classifier across 10 Chinese cities, without any manual annotation?**

This is broken down into two sub-questions:

1. **Signal** — Can multi-source Earth observation (Sentinel-1 SAR, Sentinel-2 optical and texture, and built-up context) separate urban villages from formal built-up areas and urban green at the 250 m grid scale?
2. **Models** — How do four machine-learning families (Random Forest, XGBoost, GraphSAGE, Gaussian Process) compare on this task, and does adding spatial structure (a graph) improve detection?

<p align="right"><a href="#top">↑ Back to top</a></p>

## **5. Data, Justification of the AOI selection and Platform Used**

### 5.1 City selection

The study covers 10 Chinese cities, organised into a 6-city training group and a 4-city held-out test group. The four first-tier cities (Beijing, Shanghai, Guangzhou, Shenzhen) and six second-tier cities (Tianjin, Chongqing, Wuhan, Chengdu, Xi'an, Dongguan) are listed below.

| Group | Cities | Role |
|---|---|---|
| Train | Guangzhou, Shenzhen, Tianjin, Beijing, Xi'an, Chongqing | Models fit on these |
| Test  | Dongguan, Shanghai, Chengdu, Wuhan | Never seen during training |

Beijing additionally serves as the inner validation city for GraphSAGE (Book 3), so the GNN sees only five training cities. The split is fixed across all four models so that headline metrics are directly comparable.

Three reasons motivate this 6/4 partition.

First, the first-tier cities (Beijing, Shanghai, Guangzhou, Shenzhen) are over-represented in existing research, while urban-village formation in fast-growing second-tier cities receives much less attention. Putting Dongguan, Chengdu and Wuhan into the test group forces the model to generalise into precisely the cities where new urban-village data are most needed.

Second, the ten cities collectively span the dominant Chinese settlement geographies: the Pearl River Delta (GZ, SZ, DG), the Yangtze River Delta (SH), the Beijing-Tianjin-Hebei region (BJ, TJ), the upper Yangtze (CQ, CD), the middle Yangtze (WH) and the inland north-west (XA). A model that transfers across these geographies has a reasonable claim to capture the general signature of urban villages rather than one city's idiosyncrasies.

Third, the train/test split is spatial (whole cities) rather than random within one city. This is the strict generalisation setting recommended for remote-sensing classification (Roberts et al., 2017), and is the setting in which most published urban-village models have not been benchmarked.

### 5.2 Data Sources

Five free, public data sources are used. All five are accessed at no cost.

| Source | Role | Resolution / coverage | Used in |
|---|---|---|---|
| **Sentinel-2 L2A** (Harmonised) | Optical reflectance + spectral indices | 10 m, 5-day revisit | Book 1 |
| **Sentinel-1 GRD** (IW mode, VV + VH) | C-band SAR backscatter + texture | 10 m, 12-day revisit | Book 1 |
| **ESA WorldCover v200** (2021) | Built-up class + 1 km urban-context smoothing | 10 m | Preprocess + Book 1 |
| **OpenStreetMap** (Geofabrik China PBF) | Weak supervision: `place=village/hamlet/locality`, `residential=rural`, `highway=*` | Variable, 2026 vintage | Book 0 |
| **China county/district administrative boundaries** (`CHN_level3_2024.shp`, 2853 districts) | Administrative AOI boundaries | National Bureau of Statistics 2024 vintage | Preprocess |

OSM is used only as the *source of weak labels*; it is never an input feature at inference time. This is the design choice that distinguishes this project from the methodological families surveyed in §2.

### 5.3 Area of Interest (AOI)

A naive city-administrative AOI would push the project far outside its intended remit. Full municipality boundaries include large mountain, farmland and peri-rural areas that are irrelevant to urban-village morphology. A preprocessing notebook (`pre-process/prepare_urban_aoi_from_gadm_l3.ipynb`) therefore filters the 2853 nationwide county/district polygons by an objective criterion: any district whose ESA WorldCover built-up fraction is at least 0.20 is retained, and the rest are dropped. The 0.20 threshold was selected after inspection so that urban-fringe districts where urban villages cluster (e.g. Tongzhou in Beijing, Binhai Xinqu in Tianjin, Lintong in Xi'an) are kept, while pure mountain or farmland districts are excluded. The script outputs a shared urban-core AOI containing **84 retained district polygons** across the ten cities, which is then used by both Book 0 (OSM extraction) and Book 1 (GEE feature export). Using a single shared AOI guarantees that every grid generated in Book 1 has matching OSM features within reach in Book 0, eliminating the spatial-join gaps that plagued earlier iterations.

### 5.4 Platforms

The project uses three free-tier platforms.

| Platform | Role |
|---|---|
| **Google Earth Engine (GEE)** | Sentinel-1, Sentinel-2 and WorldCover composite + grid-feature export (Book 1) |
| **Google Colab** (T4 GPU) | All Python notebooks: preprocessing, Book 0, Book 2, Book 3, Book 4 |
| **Geofabrik** | One-time China-wide OSM PBF download (~1.5 GB), cached on Drive |

No commercial imagery, no paid compute, and no manual annotation is used at any stage.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **6. Methods**

The pipeline is organised as one preprocessing notebook and five computational books. The relationship between them is shown in the master flow chart below.

```mermaid
flowchart LR
    PP["**Pre-process**<br/>district-level<br/>AOI selection"]
    B0["**Book 0**<br/>OSM weak-label<br/>source extraction"]
    B1["**Book 1**<br/>GEE feature<br/>export · 250 m grid"]
    B2["**Book 2**<br/>RF + XGBoost<br/>3-class · SMOTE · SHAP"]
    B3["**Book 3**<br/>GraphSAGE<br/>kNN graph · k=8"]
    B4["**Book 4**<br/>Gaussian Process<br/>uncertainty analysis"]
    OUT(["**Cross-model results**<br/>metrics · figures"])

    PP --> B0
    PP --> B1
    B0 --> B2
    B1 --> B2
    B2 --> B3
    B2 --> B4
    B3 --> OUT
    B4 --> OUT

    classDef data fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef model fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C
    classDef out fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    class PP,B0,B1 data
    class B2,B3,B4 model
    class OUT out
```
***Figure: Master flow chart of the five-book pipeline.** Pre-process plus Book 0 and Book 1 build the data layer. Book 2 produces the three-class weak labels and tabular baselines. Book 3 (GraphSAGE) adds the spatial model, and Book 4 (Gaussian Process) adds probabilistic uncertainty analysis.*

The first three stages (preprocess, Book 0, Book 1) build the data layer. The remaining three (Book 2, Book 3, Book 4) train and interpret the four machine-learning models.

### 6.1 Preprocessing — district-level AOI selection

```mermaid
flowchart TB
    A[("**CHN_level3_2024.shp**<br/>2853 districts nationwide")]
    A --> B["Filter by 6-digit administrative code<br/>retain 10 target prefectures"]
    B --> C(["**153 candidate districts**"])
    C --> D["Sample ESA WorldCover v200<br/>via GDAL /vsicurl/ (no GEE auth)"]
    D --> E["Compute built-up fraction<br/>class 50 / valid pixels per district"]
    E --> F{"built-up fraction<br/>≥ 0.20 ?"}
    F -- yes --> G(["**84 retained districts**"])
    F -- no --> X["drop district"]
    G --> O[("**urban-core AOI file**<br/>+ summary CSV")]

    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef process fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#212121
    classDef decision fill:#FFF9C4,stroke:#F57F17,stroke-width:2px,color:#E65100
    classDef interim fill:#E1F5FE,stroke:#0277BD,stroke-width:2px,color:#01579B
    classDef output fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20

    class A input
    class B,D,E,X process
    class C,G interim
    class F decision
    class O output
```
***Figure: Pre-process flow chart.** China Level-3 county/district boundaries are filtered down to districts with WorldCover built-up fraction ≥ 0.20, producing a tight AOI that both Book 0 and Book 1 then share.*

#### 6.1.1 Procedure

Not all districts within a city's administrative range are well developed or dominated by built-up land. The notebook `pre-process/prepare_urban_aoi_from_gadm_l3.ipynb` performs four steps. It loads the `CHN_level3_2024.shp` county/district boundary file (2853 districts nationwide), filters by the 6-digit national administrative code to retain only districts inside the ten target prefectures, samples ESA WorldCover v200 directly from its public S3 bucket via GDAL `/vsicurl/` (no Earth Engine authentication required), and computes the built-up fraction (WorldCover class 50 over total valid pixels) for each district. Districts with built-up fraction >= 0.20 are retained. The 0.20 threshold deliberately includes urban-fringe districts where urban villages tend to cluster.

#### 6.1.2 Outcome

Of 153 districts in the ten target cities, **84 districts are retained** after filtering. The breakdown by city is given in `data/processed/builtup_filtered/urban_cores_by_builtup_summary.csv`.

![Figure 3](figures/fig01_urban_core_aoi.png)

***Figure 3.** Outcome of the district-level AOI filtering step. The retained urban-core districts define the shared analysis extent used by Book 0 and Book 1.*

The script writes the kept districts to an AOI file consumed by Book 0, a zipped Shapefile uploadable to Google Earth Engine as an asset and consumed by Book 1, and a summary CSV for the report.

### 6.2 Book 0 — OSM weak-label source extraction

```mermaid
flowchart TB
    A1[("**urban-core AOI**<br/>84 polygons · 10 cities")]
    A2[("**Geofabrik snapshot**<br/>china-latest.osm.pbf · ~1.5 GB<br/>downloaded once, cached on Drive")]

    A1 --> C["**osmium-tool extract**<br/>per-city PBF using polygon bbox"]
    A2 --> C
    C --> D["**pyrosm** parses per-city PBF"]

    D --> Q1["Query 1<br/>place = village / hamlet / locality"]
    D --> Q2["Query 2<br/>residential = rural"]
    D --> Q3["Query 3<br/>highway = *"]

    Q1 --> E["Spatial clip<br/>to urban-core polygon"]
    Q2 --> E
    Q3 --> E

    E --> O1[("**OSM candidate outputs**<br/>one file per city")]
    E --> O2[("**OSM road outputs**<br/>one file per city")]

    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef process fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#212121
    classDef query fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    classDef output fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20

    class A1,A2 input
    class C,D,E process
    class Q1,Q2,Q3 query
    class O1,O2 output
```
***Figure: Book 0 flow chart.** A single offline PBF feeds three OSM queries per city. The polygon-clip step at the end matches Book 0 coverage exactly to Book 1's grid coverage, so that no Book 1 grid in Book 2 will be left without nearby OSM.*

#### 6.2.1 Procedure

The notebook `book0_osm/book0_final.ipynb` extracts OpenStreetMap features for use as weak supervision in Book 2. It downloads the daily Geofabrik snapshot `china-latest.osm.pbf` (about 1.5 GB) to Google Drive once and caches it. For each of the ten cities, the native command-line tool `osmium-tool` extracts a per-city PBF using the bounding box of that city's urban-core polygon, and the Python library `pyrosm` parses the per-city PBF in seconds. Three queries are run per city: `place = village / hamlet / locality` (the high-yield candidate signal), `residential = rural` (rare polygon hint) and the full highway network. All outputs are spatially clipped to the urban-core polygon so that OSM coverage matches Book 1 grid coverage exactly.

The design choice to use offline PBF parsing rather than the Overpass API was forced by experience. The public Overpass endpoints rate-limit and ban the Colab IP within minutes, which made earlier Overpass-based versions of Book 0 fail on 7 of 10 cities. The PBF route runs all ten cities in roughly 15 minutes end-to-end without any rate-limit risk.

#### 6.2.2 Outcome

For each city, two OSM-derived outputs are written: one candidate-set file and one road-network file. The final candidate counts range from 57 (Wuhan) to 1 750 (Beijing), and the road counts range from 27,875 (Xi'an) to 192,008 (Shanghai). The counts are sensible because Book 2's weak-label rules add built-up, urban-context and NDVI constraints that drop rural candidates downstream.

### 6.3 Book 1 — Google Earth Engine feature export

```mermaid
flowchart TB
    A[("**urban_cores_by_builtup**<br/>uploaded as GEE asset")]
    A --> B["For each city · local UTM<br/>EPSG 32648 / 32649 / 32650 / 32651"]
    B --> C["Build 250 m grid<br/>over urban-core polygon"]

    C --> F1["**Sentinel-2 annual**<br/>B2,3,4,8,11,12<br/>+ NDVI / NDBI / NDWI / BSI"]
    C --> F2["**Sentinel-2 quarterly**<br/>NDVI & NDBI · Q1–Q4<br/>+ quarterly std"]
    C --> F3["**Sentinel-2 SWIR GLCM**<br/>contrast + entropy · 90 m"]
    C --> F4["**Sentinel-1 ASC + DESC**<br/>VV / VH + texture<br/>+ has_data flag"]
    C --> F5["**ESA WorldCover**<br/>built-up · veg · water<br/>+ 1 km circular context"]

    F1 --> G["**reduceRegions** per grid cell<br/>mean + standard deviation"]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G

    G --> H{"builtup_mean ≥ 0.02<br/>OR urban_context_1km ≥ 0.40 ?"}
    H -- yes --> K[("**book1_grid_features_&lcub;city&rcub;_2025.csv**<br/>~80 columns · 333 674 grids · 10 CSVs")]
    H -- no --> X["drop cell"]

    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef process fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#212121
    classDef feature fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    classDef decision fill:#FFF9C4,stroke:#F57F17,stroke-width:2px,color:#E65100
    classDef output fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20

    class A input
    class B,C,G,X process
    class F1,F2,F3,F4,F5 feature
    class H decision
    class K output
```
***Figure: Book 1 flow chart.** Five feature groups are computed inside Google Earth Engine, reduced to a 250 m grid in each city's local UTM projection, and exported as one CSV per city.*

#### 6.3.1 Procedure

The script `book1_gee/book1_gee_export_grid_features.js` runs inside the Google Earth Engine Code Editor. For each city, it uses the urban-core polygon as the AOI and constructs a 250 m grid in the city's local UTM projection (EPSG 32648–32651, automatically dispatched to avoid Web-Mercator scale distortion at higher latitudes). For each grid cell it then reduces the following inputs to a per-cell `mean` and `stdDev` summary:

- annual Sentinel-2 surface reflectance for bands B2, B3, B4, B8, B11, B12;
- annual Sentinel-2 spectral indices NDVI, NDBI, NDWI, BSI;
- quarterly NDVI and NDBI for Q1–Q4 and the quarterly standard deviation, capturing seasonal variation that distinguishes year-stable urban-village rooftops from agricultural land;
- Sentinel-1 backscatter VV and VH for ascending and descending passes separately, with a 90 m kernel VV-texture for each pass and an `has_data` flag (some Chinese cities have only one S1 pass in 2025);
- a SWIR (B11) grey-level co-occurrence matrix at 90 m kernel for contrast and entropy;
- WorldCover classes built-up, vegetation, water, and a 1 km circular built-up smoothing as urban-context.

A grid cell is exported if either `builtup_mean ≥ 0.02` or `urban_surrounding_1km_mean ≥ 0.40`, so that urban-green cells (which have low built-up but high urban-context) are retained for the three-class problem in Book 2.

#### 6.3.2 Outcome

One CSV per city is exported to Google Drive, named `book1_grid_features_<city>_2025.csv`, with roughly 80 columns and a number of rows ranging from 10 860 (Wuhan) to 74 353 (Shanghai). Across the ten cities, **333 674 grid cells** are produced.

### 6.4 Book 2 — Three-class weak labels, RF and XGBoost

```mermaid
flowchart TB
    A1[("**Book 0**<br/>OSM candidates + roads<br/>10 cities")]
    A2[("**Book 1**<br/>EO feature CSVs<br/>10 cities · 333 674 grids")]

    A1 --> B["Spatial join<br/>OSM → 250 m grid centroids"]
    A2 --> B
    B --> R["**3-class weak-label rule**<br/>0 green · 1 normal · 2 village · −1 ambiguous"]
    R --> CC(["Class proportions<br/>**green 8.8% · normal 90.1% · village 1.1%**<br/>82 : 1 imbalance"])

    CC --> D["**Drop leaky OSM columns**<br/>dist_to_osm_candidate_m<br/>+ osm_candidate_overlap"]
    D --> E["Cross-city split<br/>**6 train · 4 test**"]

    E --> F["**SMOTE on training only**<br/>village 1 043 → 10 000<br/>green 6 229 → 15 000"]

    F --> G1["**Random Forest**<br/>class_weight = balanced"]
    F --> G2["**XGBoost**<br/>objective = multi:softprob"]

    G1 --> H["Evaluate on 4 test cities"]
    G2 --> H

    H --> O1[("metrics tables · 3×3 confusion<br/>**7-step feature ablation**<br/>**per-class SHAP**")]
    H --> O2[("prediction tables<br/>for downstream comparison")]

    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef process fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#212121
    classDef rule fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    classDef interim fill:#E1F5FE,stroke:#0277BD,stroke-width:2px,color:#01579B
    classDef model fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C
    classDef output fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20

    class A1,A2 input
    class B,D,E,F,H process
    class R rule
    class CC interim
    class G1,G2 model
    class O1,O2 output
```
***Figure: Book 2 flow chart.** Three-class weak labels are constructed from the joined OSM + EO features, leaky OSM columns are explicitly removed from X to prevent label leakage, SMOTE rebalances the training cities, and the two tabular models are trained and evaluated cross-city.*

#### 6.4.1 Procedure

The notebook `book2_colab/book2_final.ipynb` joins Book 0 OSM and Book 1 EO features and produces the main classification result. Three steps are central.

First, a three-class weak-label rule is applied. The rule uses OSM proximity (distance to candidate, overlap) together with EO statistics (`builtup_mean`, `urban_surrounding_1km_mean`, `s2_ndvi_mean`) to assign each grid cell one of: 0 = urban green (low built-up, high NDVI, in urban context), 1 = normal built-up (high built-up, far from any OSM village candidate), 2 = urban village (high built-up, close to an OSM village candidate, very low NDVI), or −1 = ambiguous (any grid that does not satisfy any of the three rules). All ambiguous grids are dropped before training. The resulting class distribution is 8.8 % green, 90.1 % normal, 1.1 % village, a severe 82 : 1 imbalance between the majority and minority class.

Second, label leakage is explicitly prevented. The two OSM-derived columns used in the label rule (`dist_to_osm_candidate_m`, `osm_candidate_overlap`) are removed from the feature matrix X before training; otherwise the model can recover the rule trivially and achieve an artificially perfect F1. A defensive assertion in the ablation cell verifies that these columns are not in X.

Third, class imbalance is handled explicitly. The training set is resampled with SMOTE (Chawla et al., 2002), raising the village count from 1 043 to 10 000 and the green count from 6 229 to 15 000 while leaving the normal class at 59 900. SMOTE is fitted on training cities only; test cities are kept pristine. Random Forest (`class_weight = 'balanced'`) and XGBoost (`objective = 'multi:softprob'`) are then fitted on the resampled training set. Per-class precision, recall, F1, macro-F1, balanced accuracy, a 7-step feature ablation and per-class SHAP explanations are computed on the held-out test cities.

#### 6.4.2 Outcome

The headline metrics on the four held-out test cities are macro-F1 = 0.721 for Random Forest and macro-F1 = 0.719 for XGBoost, with urban-village F1 of 0.169 and 0.163 respectively. These results use SMOTE on the training cities only, while the held-out test cities remain untouched. The ablation results, the SHAP plots and the 3 × 3 confusion matrices are saved under `output/` and are shown in Section 7.

### 6.5 Book 3 — GraphSAGE graph neural network

```mermaid
flowchart TB
    A[("**Book 2 labelled grids**<br/>127 899 nodes · 10 cities<br/>+ feature_cols.json")]
    A --> B["**Build kNN graph per city**<br/>k = 8 · intra-city edges only<br/>≈ 2.05 M directed edges"]
    B --> C["Cross-city split<br/>**5 train · Beijing val · 4 test**"]

    C --> D["**2-layer GraphSAGE** (pure PyTorch)<br/>x → SAGE₁ → ReLU + dropout<br/>→ SAGE₂ → ReLU + dropout<br/>→ Linear → 3-class logits"]

    D --> E["**Train 300 epochs**<br/>CrossEntropy · class weights [13.0, 1.1, 75.8]<br/>AdamW lr = 3 × 10⁻³<br/>best-on-val checkpoint"]

    E --> G["Evaluate on 4 test cities"]

    G --> O1[("**macro F1 = 0.858**<br/>**urban-village F1 = 0.580**<br/>3×3 confusion matrix")]
    G --> O2[("training history CSV<br/>node-level prediction table")]

    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef process fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#212121
    classDef model fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C
    classDef output fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20

    class A input
    class B,C,G process
    class D,E model
    class O1,O2 output
```
***Figure: Book 3 flow chart.** A per-city kNN graph carries the spatial signal that the tabular models cannot use. Class weights handle the 1.1% village prevalence inside the loss rather than via resampling.*

#### 6.5.1 Procedure

The notebook `book3_gnn/book3_final.ipynb` reads the labelled grids produced by Book 2 and constructs a per-city k-nearest-neighbour graph (k = 8) using each grid's centroid in longitude / latitude. Edges are built within a city only, and no edge crosses a city boundary, which would otherwise leak test-city structure into the training cities. The result is roughly 2.05 million directed edges on 127,899 nodes.

The model is a two-layer GraphSAGE implemented in pure PyTorch. Each layer concatenates a node's own features with the mean of its k = 8 neighbours' features and applies a linear projection. The classifier head is a linear layer producing three logits. Training uses cross-entropy with inverse-frequency class weights [13.0, 1.1, 75.8] for [green, normal, village], the AdamW optimiser (lr = 3 × 10⁻³), 300 epochs, and best-on-validation checkpoint selection using Beijing as the inner validation city.

#### 6.5.2 Outcome

GraphSAGE achieves macro-F1 = 0.858 and urban-village F1 = 0.580 on the four held-out test cities, with urban-village precision at 0.489 and recall at 0.715. This is a 3.4x improvement over the Random Forest baseline. The trained model, the training history, the 3 × 3 confusion matrix and the node-level prediction table are written to `output/`.

### 6.6 Book 4 — Gaussian Process 

```mermaid
flowchart TB
    A1[("**Book 2 labelled grids**<br/>+ feature_cols.json")]

    A1 --> B["**Stratified subsample**<br/>700 per class × 3 = 2 100 grids"]

    B --> C["**Gaussian Process Classifier**<br/>fixed RBF kernel<br/>one-vs-rest multi-class"]

    C --> D["Predict on 4 test cities<br/>per-class probability<br/>+ predictive entropy"]

    D --> E1["Evaluate GP point classification<br/>macro F1 · village precision · village recall"]
    D --> E2["Summarise uncertainty<br/>high entropy = low model confidence"]

    E1 --> O1[("**GP metric table**<br/>reported in Section 7")]
    E2 --> O2[("probability + entropy outputs<br/>for uncertainty analysis")]

    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef process fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#212121
    classDef model fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C
    classDef output fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20

    class A1 input
    class B,D,E1,E2 process
    class C model
    class O1,O2 output
```
***Figure: Book 4 flow chart.** The Gaussian Process is trained on a balanced subsample to produce both point predictions and per-grid predictive entropy. Its main role is uncertainty analysis rather than outperforming GraphSAGE.*

#### 6.6.1 Procedure

The notebook `book4_gp/book4_final.ipynb` adds a fourth model family — a Gaussian Process Classifier — and evaluates predictive uncertainty. Because the GP cost is O(n³), training is performed on a stratified subsample of 700 grids per class (2 100 grids total) using a fixed RBF kernel and one-versus-rest multi-class formulation. The trained GP outputs per-class probabilities and predictive entropy for the held-out test grids.

#### 6.6.2 Outcome

The Gaussian Process subsample classifier achieves macro-F1 = 0.527 and urban-village F1 = 0.061. Its urban-village recall is very high (0.825) but at the cost of an extremely low precision (0.032) — almost every cell the model nominates as a village is wrong. The honest interpretation is that the small balanced subsample forces the GP to predict the minority class confidently in many places where it should not. The GP's value is therefore not its point predictions but the per-grid predictive entropy, which highlights the cells the model is least sure about (see section 7.4).

<p align="right"><a href="#top">↑ Back to top</a></p>

## **7. Results**

### 7.1 Headline cross-model comparison

The table below summarises the four model families on the same four held-out test cities (Dongguan, Shanghai, Chengdu, Wuhan, totalling 60 727 labelled test grids: 5 020 green, 55 346 normal, 361 village).

| Model | macro F1 | green F1 | normal F1 | **village F1** | village P | village R |
|---|---|---|---|---|---|---|
| Random Forest (with SMOTE) | 0.721 | 1.000 | 0.994 | 0.169 | 0.166 | 0.172 |
| XGBoost (with SMOTE) | 0.719 | 1.000 | 0.994 | 0.163 | 0.146 | 0.183 |
| **GraphSAGE (k = 8)** | **0.858** | 0.998 | 0.996 | **0.580** | 0.489 | 0.715 |
| Gaussian Process (subsample) | 0.527 | 0.661 | 0.860 | 0.061 | 0.032 | 0.825 |

Two observations stand out. First, the village-class F1 spans a factor of nearly ten across the four models, even though all four were trained on the same labels and features. The choice of model family matters more than any other single design decision. Second, the green and normal classes are handled equally well by every model that can be trained on them — the *only* meaningful contest is on the urban-village class, which is exactly the class the project sets out to detect.

The headline comparison is reported as a table because the corresponding CSV metric outputs were not copied into this repository.

### 7.2 Random Forest and XGBoost results (Book 2)

The two tabular baselines behave almost identically (macro-F1 within 0.003, village-F1 within 0.006). Both reach near-perfect green-class F1 because the green-class rule is the easiest of the three to recover from EO features alone. The bottleneck is the urban-village class. The figures below are read in three groups. First the confusion matrices (where the errors fall), then the per-class SHAP summaries (why the model decides as it does), and finally the feature ablation (which input groups actually carry the signal).

**Where the errors fall — confusion matrices.** Figures 4 and 5 show the 3×3 confusion matrices on the four held-out test cities.

![Figure 4](figures/fig02_rf_confusion_matrix.png)

***Figure 4.** Random Forest confusion matrix on the held-out test cities.*

Random Forest recovers the urban-green class perfectly (5,020 / 5,020) and the normal-built-up class almost perfectly. The whole difficulty sits in the bottom row: of 361 true urban-village grids only 62 are correctly recovered (recall 0.17), 299 leak into normal built-up, and a comparable 311 normal grids are misread as village. The minority class is therefore confused almost exclusively with normal built-up, never with green.

![Figure 5](figures/fig03_xgb_confusion_matrix.png)

***Figure 5.** XGBoost confusion matrix on the held-out test cities.*

XGBoost behaves almost identically: 66 of 361 villages recovered (recall 0.18), 295 lost to normal built-up, and 385 normal grids over-predicted as village. The near-identical pattern confirms that the tabular ceiling is set by the features and weak labels, not by the bagging-versus-boosting choice.

**Why the model decides as it does — SHAP summaries.** Figures 6–8 give the per-class SHAP summaries, with each dot a test grid, its horizontal position the push towards (right) or away from (left) that class, and its colour the underlying feature value (red = high, blue = low).

![Figure 6](figures/fig05_shap_urban_green.png)

***Figure 6.** SHAP summary for the urban-green class in the Random Forest/XGBoost tabular workflow.*

For the urban-green class, `builtup_mean` is by far the strongest driver: a low built-up fraction (blue) gives a large positive push towards green, while a high fraction (red) pushes strongly against it. `builtup_stdDev`, `vegetation_mean` and `s2_ndvi_mean` follow, all consistent with green cells being low-density and well vegetated. This is the cleanest, most separable decision of the three.

![Figure 7](figures/fig06_shap_normal_built_up.png)

***Figure 7.** SHAP summary for the normal built-up class in the Random Forest/XGBoost tabular workflow.*

For the normal-built-up class the decision is led by `s2_ndvi_mean`, `builtup_stdDev` and `builtup_mean`, but the SHAP values are smaller and more symmetric than for the green class. This reflects its role as the heterogeneous majority class against which both other classes are contrasted, so no single feature dominates.

![Figure 8](figures/fig07_shap_urban_village.png)

***Figure 8.** SHAP summary for the urban-village class in the Random Forest/XGBoost tabular workflow.*

For the urban-village class `s2_ndvi_mean` dominates again, but in the opposite direction: a very low annual NDVI (blue) is the single strongest signal pushing a cell towards village, reinforced by low quarterly NDVI (`q1`, `q2`). Built-up density and Sentinel-1 descending backscatter (`s1_desc_*`) add secondary support, with `road_density_m_per_km2` contributing further down. This matches the morphological intuition that urban villages are the most vegetation-free, most densely built fabric in the city.

**Which inputs carry the signal — feature ablation.** Figure 9 reports macro-F1 (left) and urban-village F1 (right) as feature groups are added cumulatively.

![Figure 9](figures/fig04_feature_ablation.png)

***Figure 9.** Feature-ablation result for the tabular model feature groups.*

The full feature set (S2 annual + quarterly + GLCM SWIR texture + S1 ASC and DESC + WorldCover context + OSM road density) reaches macro-F1 = 0.720, whereas S2 annual alone reaches only 0.609. Macro-F1 rises monotonically as sources are added, with the largest jump coming from WorldCover urban context. The village-class panel is flatter and noisier and every configuration stays near 0.15–0.18, which again shows that the minority class is the binding constraint and that no single feature group rescues it on its own. These two model both performed badly.

### 7.3 GraphSAGE results (Book 3)

![Figure 10](figures/fig08_graphsage_confusion_matrix.png)

***Figure 10.** GraphSAGE confusion matrix on the held-out test cities.*

GraphSAGE recovers 258 of the 361 true urban-village grids (village recall 0.715) at a precision of 0.489, which is a 3.4x lift in F1 over Random Forest. The confusion matrix shows that the dominant error mode is false positives in the normal-built-up class. This is the right direction of error for a deployment scenario in which a human verifier is willing to inspect candidate grids in the field, false positives can be filtered downstream, false negatives cannot.

The training history (saved in the Colab output directory as `book3_graphsage_training_history_*.csv`) shows the model converging steadily. Village-class F1 rises from 0.135 at epoch 25 to 0.580 at epoch 300, with validation-city macro-F1 plateauing around epoch 275. The inner validation city (Beijing) prevents the test cities from being used for early stopping.

### 7.4 Gaussian Process result and uncertainty (Book 4)

The Gaussian Process classifier trained on a stratified 2 100-grid subsample produces a probability and entropy estimate for every test grid. Its point-classification performance is weak (macro-F1 = 0.527, village-F1 = 0.061), because the balanced subsample inflates the prior on the village class and the model accordingly over-predicts villages everywhere. This is, however, the expected trade-off when an algorithm with O(n³) cost is forced to use a tiny stratified sample.

The useful GP output is therefore not the point classification but the uncertainty summary. High predictive entropy marks grids where the model is least confident, which is valuable as a diagnostic complement to the stronger GraphSAGE classifier.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **8. How to Use this and Repository Structure**

### 8.1 How to reproduce

The pipeline is designed so that a reader with a Google account and ~3 GB of Drive can reproduce every result from scratch in roughly four hours. The order is fixed.

1. **Clone the repository** and upload its `data/raw/CHN_level3_2024.{shp,shx,dbf,prj,cpg}` files to `/content/drive/MyDrive/0069/week10/data/raw/`.
2. **Run** `pre-process/prepare_urban_aoi_from_gadm_l3.ipynb` in Colab. Expected runtime: 5–15 minutes (one-time WorldCover sampling). Output: a shared urban-core AOI and a zipped Shapefile.
3. **Upload** the zipped Shapefile produced in step 2 to **Google Earth Engine Assets**. Note the asset id (e.g. `projects/<your-project>/assets/0069/urban_cores_by_builtup`) must stay aligned with your own id.
4. **Run** `book0_osm/book0_final.ipynb` in Colab. First run downloads `china-latest.osm.pbf` (~1.5 GB, ~10 minutes). Subsequent runs reuse the cached PBF. Expected total: 15–25 minutes.
5. **Open** `book1_gee/book1_gee_export_grid_features.js` in the Google Earth Engine Code Editor, set `CITY_BOUNDARY_ASSET` to the asset id from step 3, set `EXPORT_ALL_CITIES = true`, and click Run → Run All Tasks. Expected total: 5–20 minutes per city, run in parallel.
6. **Run** `book2_colab/book2_final.ipynb` in Colab. Expected runtime: 25–40 minutes.
7. **Run** `book3_gnn/book3_final.ipynb` in Colab (T4 GPU required). Expected runtime: 10–20 minutes.
8. **Run** `book4_gp/book4_final.ipynb` in Colab. Expected runtime: 5–15 minutes.

Colab writes full outputs to `/content/drive/MyDrive/0069/week10/outputs/`. The GitHub repository keeps only lightweight report figures under `figures/`; large generated intermediate outputs are excluded from version control.

### 8.2 Repository structure

```text
0069_coursework/
├── README.md                         this file
├── COURSEWORK_PLAN.md                project planning notes
│
├── pre-process/
│   └── prepare_urban_aoi_from_gadm_l3.ipynb         preprocess
│
├── book0_osm/
│   └── book0_final.ipynb
│
├── book1_gee/
│   └── book1_gee_export_grid_features.js
│
├── book2_colab/
│   └── book2_final.ipynb                            RF + XGB
│
├── book3_gnn/
│   └── book3_final.ipynb                            GraphSAGE
│
├── book4_gp/
│   └── book4_final.ipynb                            Gaussian Process uncertainty
│
├── figures/
│   ├── fig02_rf_confusion_matrix.png
│   ├── fig03_xgb_confusion_matrix.png
│   ├── fig04_feature_ablation.png
│   ├── fig05_shap_urban_green.png
│   ├── fig06_shap_normal_built_up.png
│   ├── fig07_shap_urban_village.png
│   └── fig08_graphsage_confusion_matrix.png
│
├── data/
│   ├── raw/
│   │   └── CHN_level3_2024.{shp,shx,dbf,prj,cpg}    national-level admin boundary
│   └── processed/
│       └── builtup_filtered/
│           ├── urban_cores_by_builtup_summary.csv
│           └── urban_cores_by_builtup.zip           GEE-uploadable AOI
│
└── output/                                           local generated outputs
                                                      (ignored by Git)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

## **9. Environmental Cost Assessment**

### 9.1 This project

A project that classifies urban villages in service of UN Sustainable Development Goal 11 (sustainable cities and human settlements) should not be silent about its own carbon cost. SDG 13 (climate action) places this responsibility on any researcher who runs compute. This section therefore reports three things separately: measured model-training emissions, approximate runtime and storage demand, and the unmeasured cloud components, especially Google Earth Engine.

<p align="center"><img src="figures/SDG13.png" alt="Figure 11" width="220"></p>

***Figure 11.** United Nations Sustainable Development Goal 13 — Climate Action, the rationale for reporting this project's own carbon cost (United Nations, 2015).*

### 9.2 Running time and compute boundary

The full workflow is lightweight by AI standards, but it is not a zero-cost workflow. A clean run uses both Google Colab and Google Earth Engine. Colab runtime is directly visible to the user, but Earth Engine runtime is only visible as task duration and is not directly measurable with CodeCarbon.

| Phase | Platform | Approximate clean runtime | Included in CodeCarbon? | Notes |
|---|---:|---:|:---:|---|
| Preprocess AOI filtering | Colab CPU | 5–15 min | No | WorldCover sampling and district filtering |
| Book 0 OSM extraction | Colab CPU / I/O | 15–25 min | No | One-time 1.5 GB Geofabrik PBF download, then cached |
| Book 1 feature export | Google Earth Engine | 30 min–4 h per city; about 6–8 h cumulative task time | No | Ten city exports can run in parallel; Shanghai was the slowest export |
| Book 2 RF + XGBoost | Colab CPU | 25–40 min | Yes | Tabular models, SMOTE, SHAP and ablation |
| Book 3 GraphSAGE | Colab T4 GPU | 10–20 min | Yes | 300-epoch graph neural network training |
| Book 4 Gaussian Process | Colab CPU | 5–15 min | Yes | Subsampled GP uncertainty experiment |

In wall-clock terms, one clean end-to-end reproduction takes roughly four hours because the Earth Engine tasks can be queued in parallel. In cumulative compute terms, the workflow is larger than this, because the ten Earth Engine exports run as separate backend jobs.

### 9.3 Measured Colab model-training emissions

CodeCarbon (Schmidt et al., 2021) was embedded inside Books 2, 3 and 4 to log the energy used by model training on Google Colab and to convert that energy into kilograms of CO₂-equivalent. These are the only emissions directly measured in this project.

| Notebook | Activity | kg CO₂e |
|---|---|---|
| Book 2 | Random Forest + XGBoost fit | 1.9 × 10⁻³ |
| Book 3 | GraphSAGE training (300 epochs, GPU) | 1.1 × 10⁻³ |
| Book 4 | Gaussian Process fit (n = 2 100) | 6.0 × 10⁻⁵ |
| **Total measured model training** | | **≈ 3.1 × 10⁻³ kg CO₂e** |

This is roughly **3 grams of CO₂-equivalent** for the measured model-training stage. Three everyday equivalents are:

- about **0.4 smartphone charges** (at 8 g CO₂e per full charge, Belkhir & Elmeligi 2018);
- about **20 metres** of driving an average European petrol car (at 130 g CO₂e per km, EEA 2024);
- about **75 minutes** of one mature broad-leaf tree's annual CO₂ uptake (at 22 kg CO₂ per tree per year, UN-REDD 2017).

### 9.4 Google Earth Engine estimate

Book 1 is the largest untracked component. It builds Sentinel-1, Sentinel-2, WorldCover and texture features for ten cities inside Google Earth Engine. CodeCarbon cannot observe this backend computation, and GEE does not expose job-level energy use. For transparency, I therefore report it as an uncertainty range rather than a measured value.

Using a conservative illustrative assumption of a 100 W equivalent backend workload, Google's reported data-centre PUE of about 1.09 (Google Data Centers, 2025), a 6–8 h cumulative GEE task runtime, and a 0.177 kg CO₂e/kWh grid-carbon factor based on the UK Government's 2025 reporting factors (UK Government DESNZ, 2025), the GEE feature-export stage would be on the order of:

$$
\mathrm{Energy} = \mathrm{runtime} \times \mathrm{power} \times \mathrm{PUE}
$$

$$
6\text{--}8 \,\mathrm{h} \times 0.1 \,\mathrm{kW} \times 1.09
= 0.65\text{--}0.87 \,\mathrm{kWh}
$$

$$
0.65\text{--}0.87 \,\mathrm{kWh} \times 0.177
= 0.12\text{--}0.15 \,\mathrm{kg\,CO_2e}
$$

This is not a measured value. It is a scale estimate to acknowledge that the GEE backend likely dominates the carbon footprint relative to the measured Colab model training. If the true backend power were 50–200 W equivalent rather than 100 W, the GEE estimate would widen to roughly 0.06–0.31 kg CO₂e.

### 9.5 Data transfer and storage

The workflow also has a storage and data-transfer footprint. These values were measured from the local repository after outputs were copied back from Drive, plus the known size of the cached Geofabrik PBF.

| Data component | Approximate size | Treatment in this assessment |
|---|---:|---|
| Geofabrik China OSM PBF | ~1.5 GB | Downloaded once in Book 0, cached on Google Drive |
| Local `output/` directory | ~1.4 GB | Mainly prediction tables and generated analysis outputs |
| Local `data/` directory | ~127 MB | Raw and processed administrative/AOI data |
| Report `figures/` directory | ~1.9 MB | Lightweight figures kept in GitHub |
| Full repository after outputs | ~1.5 GB | Local copy after generated outputs were added |
| Estimated peak Google Drive working storage | ~3–5 GB | PBF cache, GEE exports, attached OSM outputs and notebook outputs |

The storage cost is not converted into a single CO₂e value because it depends on storage duration, data-centre location, replication policy and whether cached files are deleted after assessment. However, the estimate is still important: for this project, storage and GEE processing are probably larger environmental contributors than the actual ML model fitting.

### 9.6 What is not included

The measured 3 g CO₂e and the GEE sensitivity estimate above do not include:

1. repeated development and debugging runs;
2. exact Google Earth Engine backend energy, because job-level energy is not exposed;
3. network routing and data-transfer emissions for the Geofabrik PBF and Drive synchronisation;
4. long-term Google Drive storage, replication and backup energy;
5. embodied emissions of Google's hardware and the satellite missions themselves;
6. water use for data-centre cooling;
7. AI-assisted development queries used during debugging, writing and editing.

Therefore, the defensible statement is not that the project has a precisely known footprint. The defensible statement is that **measured model training is very small (~3 g CO₂e), while the realistic total workflow footprint is dominated by GEE processing and cloud storage, probably at the scale of tens to hundreds of grams rather than kilograms for one clean run**.

### 9.7 Environmental and social benefits

The headline counter-balance is methodological. The pipeline produces a multi-city urban-village classifier without commissioning any new sub-metre satellite imagery and without commissioning any manual annotation work. To the extent that the method scales, it substitutes free public data, weak supervision and a small amount of cloud computation for what would otherwise be an open-ended manual labelling exercise across hundreds of Chinese cities.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **10. Discussion and Limitations**

### 10.1 What the project shows, and what it does not

The project shows three concrete things. First, that EO-only weak supervision can produce a non-trivial cross-city urban-village classifier, GraphSAGE on four held-out cities reaches village-F1 = 0.58, a number that has no published cross-city baseline to compare against. Second, that the choice of model family matters far more than the choice of features once class imbalance is handled. The same Sentinel-1 + Sentinel-2 + WorldCover + OSM-road-density feature stack moves village-F1 from 0.17 (Random Forest) to 0.58 (GraphSAGE) — a 3.4x swing driven entirely by the spatial inductive bias of message-passing on a k-nearest-neighbour graph. Third, that the Gaussian Process experiment is more useful as an uncertainty screen than as a point classifier: it has high recall but poor precision.

GNN (GraphSAGE) is the best model and future analysis should consider improve on this model.

What the project does not show is the absolute accuracy of urban-village identification against hand-verified ground truth, because no such ground truth was produced. The reported numbers measure agreement against weak labels, not against reality. Section 10.3 discusses this limitation in detail.

### 10.2 Comparison with existing literature

Directly comparable numbers are not available, because, as section 2 noted, there are no published study evaluates urban-village identification with EO-only weak supervision. The closest reference points are: Pan et al. (2020) reaching ~0.88 building-level F1 on hand-labelled WorldView data in **one single** city, Zhao et al. (2020) reaching overall accuracy 0.90 on hand-labelled GaoFen-2 data in **one** city as well, and Zhang et al. (2024) UV-SAM reaching strong boundary IoU on **two** cities. All three use manual annotation and all three either evaluate on the same city used for training or report aggregate accuracy that does not isolate per-test-city performance. The GraphSAGE result here trades absolute accuracy for the absence of any hand-labelling step and the presence of strict cross-city held-out evaluation.

### 10.3 Limitations

1. **Weak labels are not ground truth.** Every reported F1 is agreement with the weak-label rule, not agreement with reality. The rule combines OSM proximity, ESA WorldCover built-up density, and Sentinel-2 NDVI; a grid that satisfies these constraints is labelled urban village, but a human inspector might disagree, especially at the boundary with normal built-up.
2. **The class-imbalance treatment is partial.** SMOTE (Book 2) and inverse-frequency class weights (Book 3) both compensate for the 1.1 % urban-village prevalence, but neither *solves* it. An additional 30 % of urban-village grids on the test cities are predicted as normal built-up by GraphSAGE.
3. **The "ambiguous" class is large.** About 60 % of all grids in the urban-context AOI fail to meet any of the three rule clauses and are dropped from training. Some of these are real urban villages that the rule simply cannot describe; the project does not quantify how many.


<p align="right"><a href="#top">↑ Back to top</a></p>

## **11. Future Work**

Three directions would extend this project usefully.

The first is **manual validation**. A subset of 200–500 grids drawn from the held-out test predictions, hand-labelled by a single trained annotator using high-resolution Esri or Tianditu imagery, would convert the project's weak-label F1 into a defensible absolute accuracy figure. The cost is a few hours of annotation per test city.

The second is **graph-aware sampling under imbalance**. GraphSMOTE (Zhao et al., 2021) and BorderlineSMOTE for graphs interpolate synthetic minority nodes *and* synthesise their edges, which is the operation SMOTE-on-features alone cannot do. Applying GraphSMOTE to the kNN graph used in Book 3 would test whether spatial oversampling of the minority class lifts GraphSAGE precision without sacrificing its high village-class recall.

The third is **finer-resolution boundary refinement**. Due to the computational limit of the free Google Earth Engine and Colab accounts used here, the current pipeline classifies at 250 m, which is appropriate for *location* but too coarse for *boundary*. A second-stage segmentation step using S2 10 m NDBI and SAR texture would make the detected locations more operationally useful.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **12. Conclusion**

The motivating question of this project was whether free public satellite data combined with weak OpenStreetMap supervision can produce a transferable urban-village classifier across Chinese cities without commissioning any new imagery, any new annotation, or any paid compute. The answer is a qualified *yes*. GraphSAGE on a 250 m grid graph trained against a three-class weak-label rule reaches macro-F1 = 0.858 and urban-village F1 = 0.580 on four held-out cities (Dongguan, Shanghai, Chengdu, Wuhan), a 3.4× lift over the tabular Random Forest and XGBoost baselines on the same data. The result is not deployment-ready, the absolute accuracy is bounded by weak-label noise and the residual class imbalance. But it is the first reproducible cross-city evidence in this literature that spatial message-passing on a graph is the inductive bias the urban-village classification problem actually needs, and that this signal can be captured without any manual annotation step. It stress that future similiar work could use GNN methods like GraphSAGE to imporve accuracy and recall. The wider methodological lesson, that the four recurrent gaps identified in §2 (single-city evaluation, missing spatial inductive bias, unhandled class imbalance, and dependence on manual labelling) can be addressed simultaneously rather than one at a time, is the principal contribution of this work.

<p align="right"><a href="#top">↑ Back to top</a></p>

## **References**

Belkhir, L. and Elmeligi, A. (2018) 'Assessing ICT global emissions footprint: Trends to 2040 & recommendations', *Journal of Cleaner Production*, 177, pp. 448–463. Available at: https://doi.org/10.1016/j.jclepro.2017.12.239 (Accessed: 31 May 2026).

Burgess, E.W. (1925) 'The Growth of the City: An Introduction to a Research Project', in Park, R.E., Burgess, E.W. and McKenzie, R.D. *The City*. Chicago: University of Chicago Press, pp. 47-62. Available at: https://www.taylorfrancis.com/chapters/edit/10.4324/9780429261732-24/growth-city-introduction-research-project-ernest-burgess.

Cao, R., Tu, W., Chen, D. and Zhang, W. (2025) 'Mapping urban villages in China: Progress and challenges', *Computers, Environment and Urban Systems*, 119, 102282. Available at: https://doi.org/10.1016/j.compenvurbsys.2025.102282 (Accessed: 31 May 2026).

Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', *Journal of Artificial Intelligence Research*, 16, pp. 321–357. Available at: https://doi.org/10.1613/jair.953 (Accessed: 31 May 2026).

European Environment Agency (2024) *CO₂ emission performance of new passenger cars in Europe*. Copenhagen: European Environment Agency. Available at: https://www.eea.europa.eu/en/analysis/indicators/co2-performance-of-new-passenger (Accessed: 31 May 2026).

General Office of the CPC Central Committee and General Office of the State Council (2025) *Opinions on Continuously Advancing the Urban Renewal Action*. Beijing: Xinhua News Agency / The State Council of the People's Republic of China. Available at: https://www.gov.cn/zhengce/202505/content_7024574.htm (Accessed: 30 May 2026).

General Office of the State Council (2023) *Guiding Opinions on Actively and Steadily Advancing the Redevelopment of Urban Villages in Mega- and Super-Large Cities (Guo Ban Fa [2023] No. 25)*. Beijing: The State Council of the People's Republic of China. Available at: https://www.gov.cn/yaowen/liebiao/202309/content_6904844.htm (Accessed: 30 May 2026).

Google Data Centers (2025) *Power usage effectiveness*. Available at: https://datacenters.google/efficiency/ (Accessed: 1 June 2026).

Kuffer, M., Pfeffer, K. and Sliuzas, R. (2016) 'Slums from space – 15 years of slum mapping using remote sensing', *Remote Sensing*, 8(6), 455. Available at: https://doi.org/10.3390/rs8060455 (Accessed: 31 May 2026).

Lin, Y., Chen, Q. and Li, S. (2025) 'A flexible framework for identifying urban villages using Sentinel-2 observations and deep learning', *International Journal of Applied Earth Observation and Geoinformation*. Available at: https://www.sciencedirect.com/science/article/pii/S156984322500278X (Accessed: 31 May 2026).

Mahabir, R., Croitoru, A., Crooks, A.T., Agouris, P. and Stefanidis, A. (2018) 'A critical review of high and very high-resolution remote sensing approaches for detecting and mapping slums: Trends, challenges and emerging opportunities', *Urban Science*, 2(1), 8. Available at: https://doi.org/10.3390/urbansci2010008 (Accessed: 31 May 2026).

National Bureau of Statistics of China (2024) *Statistical Communiqué of the People's Republic of China on the 2023 National Economic and Social Development*. Available at: http://www.stats.gov.cn/english/PressRelease/202402/t20240228_1947918.html (Accessed: 30 May 2026).

Owusu, M., Nascetti, A., Ban, Y. and Persello, C. (2023) 'Super-resolution GANs for upscaling unplanned urban settlements from remote sensing satellite imagery – the case of Chinese urban village detection', *International Journal of Digital Earth*, 16(1), pp. 2890–2911. Available at: https://doi.org/10.1080/17538947.2023.2230956 (Accessed: 31 May 2026).

Pan, Z., Xu, J., Guo, Y., Hu, Y. and Wang, G. (2020) 'Deep learning segmentation and classification for urban village using a Worldview satellite image based on U-Net', *Remote Sensing*, 12(10), 1574. Available at: https://doi.org/10.3390/rs12101574 (Accessed: 31 May 2026).

Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J.J., Schröder, B., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F. and Dormann, C.F. (2017) 'Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure', *Ecography*, 40(8), pp. 913–929. Available at: https://doi.org/10.1111/ecog.02881 (Accessed: 31 May 2026).

Schmidt, V. *et al.* (2021) *CodeCarbon: Estimate and track carbon emissions from machine learning computing*. Software. Available at: https://codecarbon.io/ (Accessed: 31 May 2026).

Strubell, E., Ganesh, A. and McCallum, A. (2019) 'Energy and policy considerations for deep learning in NLP', in *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 3645–3650. Available at: https://doi.org/10.18653/v1/P19-1355 (Accessed: 31 May 2026).

UN-REDD Programme (2017) *Forests and trees: The forgotten ingredient of climate change adaptation*. Geneva: UN-REDD Secretariat. Available at: https://www.un-redd.org/ (Accessed: 31 May 2026).

UN-Habitat (2022) *World Cities Report 2022: Envisaging the Future of Cities*. Nairobi: United Nations Human Settlements Programme (UN-Habitat). Available at: https://unhabitat.org/wcr/ (Accessed: 31 May 2026).

UK Government Department for Energy Security and Net Zero (2025) *Greenhouse gas reporting: conversion factors 2025*. Available at: https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2025 (Accessed: 1 June 2026).

United Nations (2015) *Transforming our world: the 2030 Agenda for Sustainable Development (A/RES/70/1)*. New York: United Nations. Available at: https://sdgs.un.org/goals/goal11 .

Wang, Y.P., Wang, Y. and Wu, J. (2009) 'Urbanization and informal development in China: urban villages in Shenzhen', *International Journal of Urban and Regional Research*, 33(4), pp. 957-973. Available at: https://doi.org/10.1111/j.1468-2427.2009.00891.x (Accessed: 31 May 2026).

World Bank (2022) *Lifting 800 Million People Out of Poverty – New Report Looks at Lessons from China's Experience*. Washington, DC: World Bank. Available at: https://www.worldbank.org/en/news/press-release/2022/04/01/lifting-800-million-people-out-of-poverty-new-report-looks-at-lessons-from-china-s-experience .

Yao, Y., Cao, R., Tu, W., Zhang, J. and Liu, P. (2025) 'CUGUV: A benchmark dataset for promoting large-scale urban village mapping with deep learning models', *Scientific Data*, 12, 390. Available at: https://doi.org/10.1038/s41597-025-04701-w (Accessed: 31 May 2026).

Zhang, L. (2011) 'The political economy of informal settlements in post-socialist China: the case of chengzhongcun(s)', *Geoforum*, 42(4), pp. 473-483. Available at: https://doi.org/10.1016/j.geoforum.2011.03.010 (Accessed: 31 May 2026).

Zhang, X., Liu, Y., Lin, Y., Liao, Q. and Li, Y. (2024) 'UV-SAM: Adapting Segment Anything Model for urban village identification', *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(20), pp. 22520-22528. Available at: https://doi.org/10.1609/aaai.v38i20.30260 (Accessed: 31 May 2026).

Zhao, L., Ren, H., Cui, C. and Huang, Y. (2020) 'A partition-based detection of urban villages using high-resolution remote sensing imagery in Guangzhou, China', *Remote Sensing*, 12(14), 2334. Available at: https://doi.org/10.3390/rs12142334 (Accessed: 31 May 2026).

Zhao, T., Zhang, X. and Wang, S. (2021) 'GraphSMOTE: Imbalanced node classification on graphs with graph neural networks', in *Proceedings of the 14th ACM International Conference on Web Search and Data Mining (WSDM '21)*, pp. 833–841. Available at: https://doi.org/10.1145/3437963.3441720 (Accessed: 31 May 2026).

<p align="right"><a href="#top">↑ Back to top</a></p>

## **Contact**
Author: Entao Wang
Email: zcfaew0@ucl.ac.uk
Institution: Centre for Advanced Spatial Analysis, UCL Bartlett Faculty of the Built Environment

<p align="right"><a href="#top">↑ Back to top</a></p>

## **Acknowledgement**
Many thanks to everyone who contributed to this course. Thanks to Dr. Michel for leading this module, giving me this opportunity to participate and thanks to Weibin and Shambhu for their hard work. 

<p align="right"><a href="#top">↑ Back to top</a></p>

## **Generative AI Statement**
Cursor (https://cursor.com/) was used as an assistive tool during this project. It supported code debugging, helped refine parts of the code, and was also used for wording improvements, grammar checking, and typo correction.

All research design, literature review, methodological development, notebook construction, and result interpretation were completed independently.

<p align="right"><a href="#top">↑ Back to top</a></p>
