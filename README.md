# Awesome-AI-for-Materials-Generation


## Deep Generative Models for Materials Design

### Overview

This repository presents a curated and comprehensive overview of deep generative models for **crystal structure generation**. It categorizes recent methods by generation mechanism (e.g., VAE, GAN, Diffusion, and LLM), summarizes key datasets, and provides links to implementations and papers for further exploration.

###  Motivation

The discovery of novel materials with desired physical, chemical, or mechanical properties is a longstanding challenge in materials science. Deep generative models (DGMs) have emerged as a powerful tool to design new materials by learning underlying patterns from existing structure-property databases. This repository serves as:

* A taxonomy of DGM-based crystal generation methods.
* A comparative analysis of architectures, conditioning schemes, and model sizes.
* A collection of open datasets used for training generative models.

---

### Model Summary


| Category      | Model        | Method           | Materials                           | Backbone               | Condition                             | Size             | Code & Year |
|---------------|--------------|------------------|-------------------------------------|------------------------|----------------------------------------|------------------|-------------|
| **VAE**       | iMatGen [2019](https://github.com/kaist-amsg/imatgen) | VAE              | Inorganic Crystals                  | CNN                    | Composition, Property                  | ~7M              | 2019        |
|               | Cond-DFC-VAE [2020](https://github.com/by256/icsg3d) | VAE              | Inorganic Crystals                  | CNN                    | Property                               | -                | 2020        |
|               | FTCP [2022](https://github.com/PV-Lab/FTCP)          | VAE              | Inorganic Crystals                  | CNN                    | Property                               | -                | 2022        |
|               | PCVAE [2023](https://github.com/zjuKeLiu/PCVAE)      | VAE              | Inorganic Crystals                  | MLP                    | Composition                            | ~3M              | 2023        |
|               | WyCryst [2024](https://github.com/RaymondZhurm/WyCryst) | VAE           | Inorganic Crystals                  | CNN                    | Composition, Property                  | -                | 2024        |
|               | MagGen [2024]                                        | VAE              | Permanent Magnets                   | -                      | Property                               | -                | 2024        |


---

| Category      | Model        | Method           | Materials                           | Backbone               | Condition                             | Size             | Code & Year |
|---------------|--------------|------------------|-------------------------------------|------------------------|----------------------------------------|------------------|-------------|
| **GAN**       | GANCSP [2020](https://github.com/kaist-amsg/Composition-Conditioned-Crystal-GAN) | GAN | Inorganic Crystals | CNN | Composition | ~4M | 2020 |
|               | CCDCGAN [2021]                                       | GAN              | Inorganic Crystals                  | CNN                    | Composition, Property                  | -                | 2021        |
|               | ZeoGAN [2020](https://github.com/good4488/ZeoGAN)    | GAN              | Zeolites                            | CNN                    | Property                               | ~39M             | 2020        |
|               | PGCGM [2023](https://github.com/MilesZhao/PGCGM)     | GAN              | Inorganic Crystals                  | CNN                    | Composition, Space Group              | ~5.5M            | 2023        |
|               | GAN-DDLSF [2024]                                     | GAN              | Gallium Nitride                     | -                      | Composition                            | -                | 2024        |
|               | NSGAN [2024](https://github.com/anucecszl/NSGAN_aluminium) | GAN       | Aluminium Alloys                   | MLP                    | Composition, Property                  | ~5K              | 2024        |
|               | MatGAN [2020]                                        | GAN              | Inorganic Crystals                  | CNN                    | Property                               | -                | 2020        |
|               | CubicGAN [2021]                                      | GAN              | Cubic Crystal                       | CNN                    | Composition, Space Group              | -                | 2021        |
|               | DeepCSP [2024]                                       | GAN              | Organic Crystal                     | GCN                    | Composition                            | -                | 2024        |
|               | CGWGAN [2024](https://github.com/WPEM/CGWGAN)        | GAN              | Inorganic Crystals                  | MLP                    | Composition                            | 0.38M            | 2024        |

---

###  Datasets for Material Generation

**Note:** Data were updated as of April 18, 2025.

| **Dataset**           | **#Open Access** | **#Structures**          | **Attribute** | **E or C** | **In/Organic** | **Format**                            | **Link** |
|-----------------------|------------------|---------------------------|---------------|------------|----------------|----------------------------------------|----------|
| **COD**               | ✓                | 523,874                   | ✗             | Both       | Both           | CIF                                    | [COD](https://www.crystallography.net/cod/) |
| **Materials Project** | ✓                | 154,718                   | ✓             | C          | Inorganic      | CIF, API                               | [Materials Project](https://materialsproject.org/) |
| **JARVIS‑DFT**        | ✓                | 40,000 (3D) / 1,000 (2D)  | ✓             | C          | Inorganic      | CIF, JSON, API                         | [JARVIS](https://jarvis.nist.gov/) |
| **ICSD**              | ✗                | 318,901                   | ✗             | E          | Inorganic      | CIF                                    | [ICSD](https://icsd.products.fiz-karlsruhe.de/) |
| **AFLOW**             | ✓                | 3,530,330                 | ✓             | C          | Inorganic      | API (JSON), CIF                        | [AFLOW](https://aflow.org/) |
| **OQMD**              | ✓                | 1,226,781                 | ✓             | C          | Inorganic      | JSON, API                              | [OQMD](https://oqmd.org/) |
| **ICDD (PDF‑5+)**     | ✗                | 1,104,137                 | ✗             | E          | Both           | PDF, TXT, CIF                          | [ICDD](https://www.icdd.com/) |
| **OMat24**            | ✓                | 118,000,000               | ✓             | C          | Inorganic      | ASEDB (LMDB)                           | [OMat24](https://huggingface.co/datasets/facebook/OMAT24) |
| **HKUST-CrystDB**     | ✓                | 718,725                   | ✓             | Both       | Inorganic      | ASEDB                                  | [HKUST-CrystDB](https://huggingface.co/datasets/caobin/HKUST-CrystDB) |
| **Alexandria**        | ✗                | 1,500,000+                | ✗             | C          | Inorganic      | CIF, JSON, DGL, PyG, LMDB              | [Alexandria](https://alexandria.icams.rub.de/) |
| **CSD**               | ✗                | 1,250,000+                | ✗             | E          | Organic        | CIF                                    | [CSD](https://www.ccdc.cam.ac.uk/solutions/about-the-csd/) |
| **NOMAD**             | ✓                | 19,115,490                | ✓             | C          | Inorganic      | Raw I/O, Metainfo (JSON)               | [NOMAD](https://nomad-lab.eu/) |



---



###  License

This repository is licensed under the MIT License.
