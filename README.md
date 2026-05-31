# **Evaluating the potential of multiple machine learning approaches for low-cost remote-sensing-only detection of Chinese urban villages**

## **1. Project Overview and Background Information**

Accompanied by the rapid urbanization process across the world, the inequality of development within modern cities is outstanding. The classical urban model, in which a central business district (CBD) is surrounded by a deteriorating inner 'zone of transition' or slum, was first set out by Burgess (1925). However, after the turn of the 21st century, the skyrocketing growth rate of countless Global South countries has added new layers to that model. This is especially true in China, one of the fastest growing countries, which has lifted close to 800 million people out of extreme poverty over the past four decades (World Bank, 2022) and where 66.16% of the total population was living in cities by the end of 2023 (National Bureau of Statistics of China, 2024). Throughout this transformation, the inequality within cities has been comparatively neglected and not focused on.

In 2015, the United Nations adopted the 2030 Agenda for Sustainable Development, whose Goal 11 calls on countries to 'make cities and human settlements inclusive, safe, resilient and sustainable' (United Nations, 2015). After mass urbanisation, the inequality within fast developing cities cannot be ignored anymore. In China, the State Council issued its first dedicated guideline on urban-village redevelopment in 2023, initially targeting 35 mega- and super-large cities and, since late 2024, extending support to nearly 300 prefecture-level cities (General Office of the State Council, 2023); in 2025 the central authorities further designated the renewal of urban villages as a priority task within a nationwide urban-renewal action plan aiming for major progress by 2030 (General Office of the CPC Central Committee and General Office of the State Council, 2025).

### 1.1 What is Urban Village

An urban village (城中村, *chengzhongcun*, literally 'village within the city') is a formerly rural village that has been physically encircled by the rapid outward expansion of a Chinese city (Cao et al., 2025). Under China's dual urban-rural land system, such villages retain collectively owned rural land and their original household-registration (*hukou*) based governance even after the surrounding farmland has been urbanised, which leaves them largely outside the formal urban planning framework (Wang et al., 2009). The outcome is a dense, informal, often high-rise settlement that supplies cheap rental housing to low-income rural migrants and the urban poor (Zhang, 2011).

### 1.2 Difference between urban village and slum

Although urban villages superficially resemble the slums, favelas and shantytowns of other developing countries, two key differences distinguish them. The first is tenure: urban villages sit on collectively owned rural land on which villagers hold recognised property rights and build durable, multi-storey 'handshake' housing, rather than squatting illegally on land they do not own (Wang et al., 2009). The second is origin: *chengzhongcun* are a deliberate and legally distinct outcome of China's dual urban-rural land system, not a product of state or market failure (Zhang, 2011). As a result, they are physically much closer to formal built-up fabric than classical slums are (Cao et al., 2025).

### 1.3 Difference between urban village and normal built-up area

Both urban villages and the surrounding formal built-up area appear as continuous, high-density urban fabric, which makes them hard to separate, especially in coarse remote-sensing data (Cao et al., 2025). The difference is one of morphology and planning. Formal built-up areas, such as residential compounds, commercial districts and industrial parks, follow planned street grids with building setbacks, wider roads and deliberate green and open space. Urban villages, built incrementally by villagers without unified planning, are far denser: tightly packed multi-storey 'handshake' buildings separated by narrow alleys, with minimal vegetation or open space (Wang et al., 2009).

### 1.4 What is the issue with urban village?

The first issue is physical and safety-related: urban villages typically suffer from high public-health risks, frequent building and fire-safety hazards, poor supporting infrastructure and a degraded living environment (General Office of the State Council, 2023).

The second issue is social: as the cheapest foothold in the city, urban villages concentrate low-income rural migrants who, excluded by the *hukou* system, remain marginalised from formal urban services and welfare (Wang et al., 2009).

The third main issue is informational: urban villages lack reliable, up-to-date geospatial data, which leaves planners without a clear picture of where they are and hinders evidence-based upgrading (Cao et al., 2025).

### 1.5 Why are we using satellite image to monitor urban village?

Field surveys and manual digitisation can identify urban villages accurately, but they are slow, expensive and quickly outdated, so they cannot keep pace with rapid redevelopment across many cities (Cao et al., 2025). Satellite imagery offers a scalable alternative, it covers entire cities consistently, is revisited every few days, and, through free public missions such as Sentinel-1 and Sentinel-2, incurs almost no data cost. Earth observation also has an established record of mapping informal settlements from their distinctive built-up texture and morphology (Kuffer et al., 2016), making it well suited to detecting the dense fabric of urban villages at scale. Specifically, the C-band VV and VH backscatter of Sentinel-1 is sensitive to the rough, volumetric scattering of densely packed buildings, while the short-wave infrared bands of Sentinel-2 (B11 and B12) and the grey-level texture derived from them best capture built-up materials and the fine-grained heterogeneity of the village fabric.

### 1.6 Why is machine learning suitable for analysing these satellite images? 

Separating urban villages from formal built-up areas is not a simple threshold problem: the distinction emerges only from subtle, non-linear combinations of many weak signals, such as SAR backscatter, SWIR texture, vegetation indices and urban context, which also vary from city to city. Machine learning is well suited to this setting because it can learn these high-dimensional, non-linear feature interactions directly from data, rather than relying on hand-crafted rules. It also scales to the hundreds of thousands of grid cells produced across ten cities and supports cross-city generalisation, precisely the scalability and transferability that current urban-village mapping still lacks (Cao et al., 2025).

## **2. Existing Literature Review:**

Although urban villages are an established topic in urban planning and social science, the remote-sensing literature devoted to mapping them is comparatively thin. The most recent systematic review, by Cao et al. (2025), records fewer than fifty peer-reviewed remote-sensing studies on Chinese urban villages across roughly two decades, with annual publication volume rarely exceeding five papers per year (Cao et al., 2025, Figure 2). 

By way of contrast, the broader literature on global slum and informal-settlement mapping reviewed by Kuffer et al. (2016) holds an order of magnitude more entries, and the follow-up critical review by Mahabir et al. (2018) extends this analysis to high- and very-high-resolution approaches. This volume gap is diagnostic rather than coincidental. The Chinese urban village is morphologically much closer to formal built-up fabric than to favela-style slums, so the standard slum-detection pipelines transfer to it only partially, and a parallel domain-specific literature has therefore developed slowly.

The urban-village-specific literature itself clusters into three methodological families.

**(i) Object-based image analysis (OBIA) on very-high-resolution (VHR) optical imagery.** 

The earliest and longest-running family, originating in the late 2000s as VHR Chinese imagery became available. A representative recent contribution is Zhao et al. (2020), who proposed a partition-based OBIA framework integrating a Random Forest classifier with GaoFen-2 imagery for Guangzhou, reporting an overall accuracy of 90.2% and Cohen’s κ = 0.80 across four re-divided administrative zones. However, these studies operate on commercial sub-metre imagery (typically GaoFen-2 or WorldView, with raw-data costs of roughly USD 10–30 per km²) and their segmentation rules and partition schemes require per-city manual tuning. None of the reviewed OBIA studies tests cross-city transfer, and the partition-based design in particular is explicitly tailored to the administrative geometry of a single region.

**(ii) Single-city semantic-segmentation with hand-labelled training data.** 

Since approximately 2020, the dominant approach has shifted to pixel-wise deep learning. Pan et al. (2020) trained a U-Net on WorldView satellite imagery for Guangzhou and reported strong building-level segmentation. Owusu et al. (2023) extended this line by inserting super-resolution generative adversarial networks (SR-GAN) as a preprocessor for multi-resolution chengzhongcun imagery. However, every paper in this family requires 200–2,000 hand-traced training polygons in a single study city, the annotation cost is rarely reported in the methods but is consistently the dominant cost of the work, and again no cross-city held-out evaluation is performed. The resulting models are therefore best understood as city-specific tools rather than transferable methods.

**(iii) The recent multi-source / multi-city frontier (2024–2025).** 

A new wave of work has begun to push past the single-city bottleneck. Zhang et al. (2024) introduced UV-SAM, an adaptation of Meta’s Segment Anything Model that produces fine-grained boundary masks across two Chinese cities and outperforms earlier baselines on both datasets. A flexible Sentinel-2-only deep-learning framework (Lin et al., 2025) employs a dual-branch encoder that fuses annual and seasonal imagery and reports mIoU = 0.83 with cross-city transfer tested within the Pearl River Delta. Most ambitiously, the CUGUV benchmark (Yao et al., 2025) released the first multi-city hand-curated urban-village dataset covering 15 Chinese cities for two years (2010 and 2020), and the accompanying Segformer model claims robust large-scale mapping performance. Similiarly, every paper in this frontier still depends on **hand-curated training labels**. CUGUV in particular reports thousands of hand-traced polygons across its 15-city sample. Although these works demonstrate transfer across handfuls of cities, the labelling effort scales linearly with the number of cities studied, so the fundamental cost barrier of manual digitisation is reduced but not removed.

Looking across all three families, and at the wider slum literature (Kuffer et al., 2016; Mahabir et al., 2018), four limitations comes up. First, **cross-city held-out evaluation is rare**. Even the recent CUGUV dataset reports one aggregate accuracy figure, rather than results for each test city.Second, most studies still **classify each pixel or grid on its own**. This ignores the fact that urban villages cluster together in space, and spatial graph models that could use this clustering are absent from the reviewed work. Thirdly, **the imbalanced original data** is always ignored. Urban villages usually cover only 1–2% of the urban area. This is rarely treated as a core problem, so a high overall accuracy can still hide poor recall on the village class. Fourth, even the latest cross-city deep-learning methods (Zhang et al., 2024; Lin et al., 2025; Yao et al., 2025) **still need hand-labelled training data for every new city**. It remains unclear whether free, public weak signals could replace manual annotation altogether.

These four structural gaps—single-city evaluation, missing spatial inductive bias, unhandled class imbalance, and persistent dependence on manual labelling—together define the methodological space this project occupies, and motivate the specific research gaps set out below.

## **3. Current research gap**

1.  Despite the scale of the phenomenon, urban villages remain comparatively understudied, and the first comprehensive review of urban-village mapping in China appeared only recently (Cao et al., 2025).
2.  Chinese urban villages differ fundamentally from the classical slums of other developing countries: built on collectively owned rural land, they are a deliberate product of the dual land system rather than illegal squatting, which makes them an institutionally distinct and more complex phenomenon (Zhang, 2011).
3.  Methods developed for this setting have wider relevance, because informal and inadequate settlements are growing fastest across the rapidly urbanising Global South, particularly in Asia and sub-Saharan Africa (UN-Habitat, 2022).
4.  There is still no unified definition of, or consensus method for, identifying urban villages, owing to persistent conceptual fuzziness and their spatial heterogeneity across cities (Cao et al., 2025).
5.  Existing mapping studies are largely confined to single cities or single images (e.g. Pan et al., 2020), and large-scale, cross-city benchmarks have only begun to emerge (Yao et al., 2025); the cross-city transferability of identification approaches therefore remains largely untested.

## **4. Research Question**

Building on the gaps above, this project asks one main question:

> **Can free, public satellite data combined with weak OSM supervision deliver a transferable urban-village classifier across multiple Chinese cities, without any manual annotation?**

This is broken down into four sub-questions:

1. **Signal** — Can multi-source Earth observation (Sentinel-1 SAR, Sentinel-2 optical and texture, and built-up context) separate urban villages from formal built-up areas and urban green at the 250 m grid scale?
2. **Models** — How do four machine-learning families (Random Forest, XGBoost, GraphSAGE, Gaussian Process) compare on this task, and does adding spatial structure (a graph) improve detection?
3. **Transferability** — Does a classifier trained on a subset of cities generalise to unseen, held-out cities?
4. **Trust** — How well-calibrated and interpretable are the resulting predictions?

## **5. Data, Justify of the AOI selection and Platform Used**

### 5.1 City selection:

10 Cities: 4 first tier cities: Beijing, Shanghai, Guangzhou, Shenzhen. 6 second tier cities: Tianjin, Chongqing, Wuhan, Chengdu, Xi'an, Dongguan.

Train Test Split: 60%- 40% Test city includes different cities.

### Reason of selection these cities:

1.  There are a lot of research on first tier cities but not enough research on second tier cities. And as second tier cities grow these years, it urban village problem cannot be neglected and needs to be studied.
2.  These cities are representative of the urban village problem in China, usually due to its faster development, large population, and rapid urbanisation.
3.  These cities are located in different regions in China, it represents the model could learn the general pattern of the urban village problem in China.

### 5.2 Data Source:

i). Level-3 shapefile for 10 interested chinese cities.

### 5.3 Platform Used:

Google Earth Engine (GEE) and Google Colab.

## **6. Methods**

Overall Flow Chart:

Overall Methodology:

Data processing, cleaning and preparation: Section 6.1, 6.2, 6.3. Model training, evaluation and interpretation: Section 6.4, 6.5, 6.6.

### 6.1 Preprocessing:

6.1.1 Preprocess notebook flow chart:

6.1.2 Preprocessing procedure and AOI selection criteria:

6.1.3 Preprocess Outcome:

### 6.2 Book 0: OSM weak-label source extraction

6.2.1 Book 0 (OSM weak-label source extraction) flow chart:

6.2.2 Book 0 (OSM weak-label source extraction) procedure:

6.2.3 Book 0 (OSM weak-label source extraction) Outcome:

### 6.3 Book 1: GEE Feature Export

6.3.1 Book 1 (GEE Feature Export) flow chart:

6.3.2 Book 1 (GEE Feature Export) procedure:

6.3.3 Book 1 (GEE Feature Export) Outcome:

### 6.4 Book 2: RF / XGBoost Main Model

6.4.1 Book 2 flow chart:

6.4.2 Book 2 procedure:

### 6.5 Book 3: GNN Extension

6.5.1 Book 3 flow chart:

6.5.2 Book 3 procedure:

### 6.6 Book 4: Gaussian Process Uncertainty and Calibration

6.6.1 Book 4 flow chart:

6.6.2 Book 4 procedure:

## **7. Results**

### 7.1 Random Forest and XGBoost Model Result:

### 7.2 GNN (GraphSAGE Model) Result:

### 7.3 Gaussian Process Result:

## **8. How to Use this and Repository Structure:**

### 8.1 How to Use this:

### 8.2 Repository Structure:

## **9. Environmental Cost Assessment:**

### 9.1 Overview and UNSDGs 2030:

### 9.2 Environmental Cost Assessment:

### 9.3 Environmental Benefits:

## **10. Discussion and Limitations:**

### 10.1 Discussion and Comparasion with other research

### 10.2 Limitations:

## **11. Future Work**

## **12. Conclusion**

## **13. References**

Burgess, E.W. (1925) 'The Growth of the City: An Introduction to a Research Project', in Park, R.E., Burgess, E.W. and McKenzie, R.D. *The City*. Chicago: University of Chicago Press, pp. 47-62. Available at: https://www.taylorfrancis.com/chapters/edit/10.4324/9780429261732-24/growth-city-introduction-research-project-ernest-burgess.

Cao, R., Tu, W., Chen, D. and Zhang, W. (2025) 'Mapping urban villages in China: Progress and challenges', *Computers, Environment and Urban Systems*, 119, 102282. Available at: https://doi.org/10.1016/j.compenvurbsys.2025.102282 (Accessed: 31 May 2026).

General Office of the CPC Central Committee and General Office of the State Council (2025) *Opinions on Continuously Advancing the Urban Renewal Action*. Beijing: Xinhua News Agency / The State Council of the People's Republic of China. Available at: https://www.gov.cn/zhengce/202505/content_7024574.htm (Accessed: 30 May 2026).

General Office of the State Council (2023) *Guiding Opinions on Actively and Steadily Advancing the Redevelopment of Urban Villages in Mega- and Super-Large Cities (Guo Ban Fa [2023] No. 25)*. Beijing: The State Council of the People's Republic of China. Available at: https://www.gov.cn/yaowen/liebiao/202309/content_6904844.htm (Accessed: 30 May 2026).

Kuffer, M., Pfeffer, K. and Sliuzas, R. (2016) 'Slums from space – 15 years of slum mapping using remote sensing', *Remote Sensing*, 8(6), 455. Available at: https://doi.org/10.3390/rs8060455 (Accessed: 31 May 2026).

Lin, Y., Chen, Q. and Li, S. (2025) 'A flexible framework for identifying urban villages using Sentinel-2 observations and deep learning', *International Journal of Applied Earth Observation and Geoinformation*. Available at: https://www.sciencedirect.com/science/article/pii/S156984322500278X (Accessed: 31 May 2026).

Mahabir, R., Croitoru, A., Crooks, A.T., Agouris, P. and Stefanidis, A. (2018) 'A critical review of high and very high-resolution remote sensing approaches for detecting and mapping slums: Trends, challenges and emerging opportunities', *Urban Science*, 2(1), 8. Available at: https://doi.org/10.3390/urbansci2010008 (Accessed: 31 May 2026).

National Bureau of Statistics of China (2024) *Statistical Communiqué of the People's Republic of China on the 2023 National Economic and Social Development*. Available at: http://www.stats.gov.cn/english/PressRelease/202402/t20240228_1947918.html (Accessed: 30 May 2026).

Owusu, M., Nascetti, A., Ban, Y. and Persello, C. (2023) 'Super-resolution GANs for upscaling unplanned urban settlements from remote sensing satellite imagery – the case of Chinese urban village detection', *International Journal of Digital Earth*, 16(1), pp. 2890–2911. Available at: https://doi.org/10.1080/17538947.2023.2230956 (Accessed: 31 May 2026).

Pan, Z., Xu, J., Guo, Y., Hu, Y. and Wang, G. (2020) 'Deep learning segmentation and classification for urban village using a Worldview satellite image based on U-Net', *Remote Sensing*, 12(10), 1574. Available at: https://doi.org/10.3390/rs12101574 (Accessed: 31 May 2026).

UN-Habitat (2022) *World Cities Report 2022: Envisaging the Future of Cities*. Nairobi: United Nations Human Settlements Programme (UN-Habitat). Available at: https://unhabitat.org/wcr/ (Accessed: 31 May 2026).

United Nations (2015) *Transforming our world: the 2030 Agenda for Sustainable Development (A/RES/70/1)*. New York: United Nations. Available at: https://sdgs.un.org/goals/goal11 .

Wang, Y.P., Wang, Y. and Wu, J. (2009) 'Urbanization and informal development in China: urban villages in Shenzhen', *International Journal of Urban and Regional Research*, 33(4), pp. 957-973. Available at: https://doi.org/10.1111/j.1468-2427.2009.00891.x (Accessed: 31 May 2026).

World Bank (2022) *Lifting 800 Million People Out of Poverty – New Report Looks at Lessons from China's Experience*. Washington, DC: World Bank. Available at: https://www.worldbank.org/en/news/press-release/2022/04/01/lifting-800-million-people-out-of-poverty-new-report-looks-at-lessons-from-china-s-experience .

Yao, Y., Cao, R., Tu, W., Zhang, J. and Liu, P. (2025) 'CUGUV: A benchmark dataset for promoting large-scale urban village mapping with deep learning models', *Scientific Data*, 12, 390. Available at: https://doi.org/10.1038/s41597-025-04701-w (Accessed: 31 May 2026).

Zhang, L. (2011) 'The political economy of informal settlements in post-socialist China: the case of chengzhongcun(s)', *Geoforum*, 42(4), pp. 473-483. Available at: https://doi.org/10.1016/j.geoforum.2011.03.010 (Accessed: 31 May 2026).

Zhang, X., Liu, Y., Lin, Y., Liao, Q. and Li, Y. (2024) 'UV-SAM: Adapting Segment Anything Model for urban village identification', *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(20), pp. 22520-22528. Available at: https://doi.org/10.1609/aaai.v38i20.30260 (Accessed: 31 May 2026).

Zhao, L., Ren, H., Cui, C. and Huang, Y. (2020) 'A partition-based detection of urban villages using high-resolution remote sensing imagery in Guangzhou, China', *Remote Sensing*, 12(14), 2334. Available at: https://doi.org/10.3390/rs12142334 (Accessed: 31 May 2026).
