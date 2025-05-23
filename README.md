# Awesome-AI-for-Materials-Generation | [arXiv](https://arxiv.org/pdf/2505.16379)


## Deep Generative Models for Materials Design

### Overview

This repository presents a curated and comprehensive overview of deep generative models for **crystal structure generation**. It categorizes recent methods by generation mechanism (e.g., VAE, GAN, Diffusion, and LLM), summarizes key datasets, and provides links to implementations and papers for further exploration.

###  Motivation

The discovery of novel materials with desired physical, chemical, or mechanical properties is a longstanding challenge in materials science. Deep generative models (DGMs) have emerged as a powerful tool to design new materials by learning underlying patterns from existing structure-property databases. This repository serves as:

* A taxonomy of DGM-based crystal generation methods.
* A comparative analysis of architectures, conditioning schemes, and model sizes.
* A collection of open datasets used for training generative models.


![image](https://github.com/user-attachments/assets/294ceb21-52e6-42a2-94c5-08fc09380494)


---

### Model Summary


| Model        | Method | Materials          | Backbone | Condition             | Size | Code & Year                                     |
| ------------ | ------ | ------------------ | -------- | --------------------- | ---- | ----------------------------------------------- |
| iMatGen      | VAE    | Inorganic Crystals | CNN      | Composition, Property | \~7M | [2019](https://github.com/kaist-amsg/imatgen)   |
| Cond-DFC-VAE | VAE    | Inorganic Crystals | CNN      | Property              | -    | [2020](https://github.com/by256/icsg3d)         |
| FTCP         | VAE    | Inorganic Crystals | CNN      | Property              | -    | [2022](https://github.com/PV-Lab/FTCP)          |
| PCVAE        | VAE    | Inorganic Crystals | MLP      | Composition           | \~3M | [2023](https://github.com/zjuKeLiu/PCVAE)       |
| WyCryst      | VAE    | Inorganic Crystals | CNN      | Composition, Property | -    | [2024](https://github.com/RaymondZhurm/WyCryst) |
| MagGen       | VAE    | Permanent Magnets  | -        | Property              | -    | 2024                                            |

---

| Model     | Method | Materials          | Backbone | Condition                | Size   | Code & Year                                                               |
| --------- | ------ | ------------------ | -------- | ------------------------ | ------ | ------------------------------------------------------------------------- |
| GANCSP    | GAN    | Inorganic Crystals | CNN      | Composition              | \~4M   | [2020](https://github.com/kaist-amsg/Composition-Conditioned-Crystal-GAN) |
| CCDCGAN   | GAN    | Inorganic Crystals | CNN      | Composition, Property    | -      | 2021                                                                      |
| ZeoGAN    | GAN    | Zeolites           | CNN      | Property                 | \~39M  | [2020](https://github.com/good4488/ZeoGAN)                                |
| PGCGM     | GAN    | Inorganic Crystals | CNN      | Composition, Space Group | \~5.5M | [2023](https://github.com/MilesZhao/PGCGM)                                |
| GAN-DDLSF | GAN    | Gallium Nitride    | -        | Composition              | -      | 2024                                                                      |
| NSGAN     | GAN    | Aluminium Alloys   | MLP      | Composition, Property    | \~5K   | [2024](https://github.com/anucecszl/NSGAN_aluminium)                      |
| MatGAN    | GAN    | Inorganic Crystals | CNN      | Property                 | -      | 2020                                                                      |
| CubicGAN  | GAN    | Cubic Crystal      | CNN      | Composition, Space Group | -      | 2021                                                                      |
| DeepCSP   | GAN    | Organic Crystal    | GCN      | Composition              | -      | 2024                                                                      |
| CGWGAN    | GAN    | Inorganic Crystals | MLP      | Composition              | 0.38M  | [2024](https://github.com/WPEM/CGWGAN)                                    |

---

| Model         | Method | Materials                           | Backbone             | Condition                          | Size          | Code & Year                                                                |
| ------------- | ------ | ----------------------------------- | -------------------- | ---------------------------------- | ------------- | -------------------------------------------------------------------------- |
| CDVAE         | SMLD   | Inorganic Crystals                  | DimeNet+GemNet       | Property                           | 4.5M          | [2021](https://github.com/txie-93/cdvae)                                   |
| Cond-CDVAE    | SMLD   | Inorganic Crystals                  | DimeNet+GemNet       | Composition, Property              | 4M/86M        | [2024](https://github.com/ixsluo/cond-cdvae)                               |
| Con-CDVAE     | SMLD   | Inorganic Crystals                  | DimeNet+GemNet       | Composition, Property              | \~5M          | [2024](https://github.com/cyye001/Con-CDVAE)                               |
| P-CDVAE       | SMLD   | Inorganic Crystals                  | DimeNet+GemNet       | Composition, Property              | -             | 2023                                                                       |
| LCOMs         | SMLD   | Inorganic Crystals                  | DimeNet+GemNet       | Composition                        | 4.5M          | 2023                                                                       |
| StructRepDiff | DDPM   | Inorganic Crystals                  | U-Net                | -                                  | 1\~10M        | 2024                                                                       |
| DiffCSP       | DDPM   | Inorganic Crystals                  | Periodic GNN         | Composition                        | 12.3M         | [2023](https://github.com/jiaor17/DiffCSP)                                 |
| UniMat        | DDPM   | Inorganic Crystals                  | U-Net                | Composition, Property              | -             | [2023](https://unified-Crystals.github.io/unimat/)                         |
| DiffCSP++     | DDPM   | Inorganic Crystals                  | Periodic GNN         | Composition, Space Group           | 12.3M         | [2024](https://github.com/jiaor17/DiffCSP-PP)                              |
| GemsDiff      | DDPM   | Inorganic Crystals                  | GemsNet              | Composition                        | 2.8M          | [2024](https://github.com/aklipf/gemsdiff)                                 |
| EquiCSP       | DDPM   | Inorganic Crystals                  | Periodic GNN         | Composition                        | 12.3M         | [2024](https://github.com/EmperorJia/EquiCSP)                              |
| FlowMM        | RFM    | Inorganic Crystals                  | Periodic GNN         | Composition                        | 12.3M         | [2024](https://github.com/facebookresearch/flowmm)                         |
| SuperDiff     | DDPM   | Superconductors                     | U-Net                | Composition, Property              | -             | [2024](https://github.com/sdkyuanpanda/SuperDiff)                          |
| SymmCD        | DDPM   | Inorganic Crystals                  | Periodic GNN         | Composition, Space Group           | 12.3M         | [2025](https://github.com/sibasmarak/SymmCD)                               |
| MOFDiff       | SMLD   | Metal-organic Frameworks            | GemNet               | Property                           | 27.2M         | [2023](https://github.com/microsoft/MOFDiff)                               |
| MatterGen     | DDPM   | Inorganic Crystals                  | GemNet               | Composition, Space Group, Property | 46.8M         | [2025](https://github.com/microsoft/mattergen)                             |
| MOFFlow       | RFM    | Metal-organic Frameworks            | EGNN+OpenFold        | Composition                        | 22.5M         | [2024](https://github.com/nayoung10/MOFFlow)                               |
| CrystalFlow   | RFM    | Inorganic Crystals                  | Periodic GNN         | Composition, Property              | 12.3M         | [2024](https://github.com/ixsluo/CrystalFlow)                              |
| ADiT          | LFM    | Atomic Systems                      | Transformer          | -                                  | 32M/130M/450M | [2025](https://github.com/facebookresearch/all-atom-diffusion-transformer) |
| DAO           | DDPM   | Inorganic Crystals, Superconductors | Periodic Transformer | Composition, Property              | -             | 2025                                                                       |
| CrystalGRW    | GRW    | Inorganic Crystals                  | EquiformerV2         | Property                           | 34.9M         | [2025](https://github.com/trachote/crystalgrw)                             |



---

| Model           | Method            | Materials                           | Backbone              | Condition                                 | Size         | Code & Year                                          |
|----------------|-------------------|-------------------------------------|------------------------|-------------------------------------------|--------------|------------------------------------------------------|
| G-SchNet       | NTP               | Atomic Systems                      | CNN                    | Composition                               | -            | [2019](https://github.com/atomistic-machine-learning/G-SchNet) |
| xyztransformer | NTP               | Atomic Systems                      | Transformer            | Composition                               | 1–100M       | [2023](https://github.com/danielflamshep/xyztransformer)       |
| CrystaLLM      | NTP               | Inorganic Crystals                  | GPT-2                  | Composition                               | 25M          | [2024](https://github.com/lantunes/CrystaLLM)                  |
| CrystalLLM     | NTP               | Inorganic Crystals                  | LLaMA-2                | Composition, Text                         | 7B/13B/70B   | [2024](https://github.com/facebookresearch/crystal-text-llm)   |
| SLI2Cry        | NTP               | Inorganic Crystals                  | GRU                    | Composition, Property                     | -            | [2023](https://github.com/xiaohang007/SLICES)                  |
| Mat2Seq        | NTP               | Inorganic Crystals                  | GPT-2                  | Composition                               | 25M/200M     | [2024](https://github.com/YKQ98/Mat2Seq)                       |
| MatExpert      | RAG               | Inorganic Crystals                  | LLaMA-2/3              | Composition, Text                         | 8B/70B       | [2024](https://github.com/BangLab-UdeM-Mila/MatExpert)         |
| NatureLM       | NTP               | Atomic Systems                      | Transformer            | Composition, Space Group, Property        | 1B/8B/46.7B  | [2025](https://naturelm.github.io/)                           |
| MatLLMSearch   | NTP               | Inorganic Crystals                  | LLaMA-3.1              | Composition                               | 70B          | [2025](https://github.com/JingruG/MatLLMSearch)                |
| Uni-3DAR       | MNTP              | Atomic Systems                      | Transformer            | Composition, Text                         | 90M          | [2025](https://github.com/dptech-corp/Uni-3DAR)                |

---

| Model           | Method            | Materials                           | Backbone              | Condition                                 | Size         | Code & Year                                          |
|----------------|-------------------|-------------------------------------|------------------------|-------------------------------------------|--------------|------------------------------------------------------|
| FlowLLM        | LLM+RFM           | Inorganic Crystals                  | LLaMA-2 + GNN          | Composition                               | 70B          | [2024](https://github.com/facebookresearch/flowmm)             |
| GenMS          | LLM+DM            | Inorganic Crystals                  | Gemini + Transformer   | Composition                               | -            | 2024                                                   |
| TGDMat         | LM+DM             | Inorganic Crystals                  | SciBERT + EGNN         | Composition, Space Group                  | -            | [2025](https://github.com/kdmsit/TGDMat)                       |
| UniGenX        | NTP+DDPM          | Atomic Systems                      | Transformer + MLP      | Composition                               | 100M/400M    | 2025                                                   |
| LCMGM          | VAE+GAN           | Inorganic Crystals                  | CNN                    | Crystal System                            | -            | [2024](https://github.com/chenebuah/LCMGM)                     |
| VGD-CG         | VAE+GAN+DDPM      | Inorganic Crystals, Semiconductor   | U-Net                  | Composition                               | -            | [2024](https://github.com/stupidcloud/VGD-CG)                 |
| DP-CDVAE       | DDPM+VAE          | Crystal Structures                  | GemNet                 | Composition, Space Group, Property        | -            | [2024](https://github.com/trachote/dp-cdvae)                   |


---

| Model           | Method            | Materials                           | Backbone              | Condition                                 | Size         | Code & Year                                          |
|----------------|-------------------|-------------------------------------|------------------------|-------------------------------------------|--------------|------------------------------------------------------|
| EMPNN          | MPNN              | Inorganic Crystals                  | MPNN                   | Composition, Noisy Structure              | -            | [2023](https://github.com/aklipf/pegnn)                        |
| CrysBFN        | BFN               | Inorganic Crystals                  | Periodic GNN           | Composition                               | 12.3M        | [2025](https://github.com/wu-han-lin/CrysBFN)                  |
| CHGlownet      | GFlowNet          | Inorganic Crystals                  | GCN + MLP              | Composition, Space Group, Property        | -            | 2023                                                   |




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
