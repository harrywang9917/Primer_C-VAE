# Primer C-VAE: An Interpretable Deep Learning Primer Design Method to Detect Emerging Virus Variants

**CORMSIS External Summer Project (Master Graduation Project)**

**Period:** 2021-06 — 2022-06  
**Updated:** March 2025

**Tutors:**  
- Dr. Alain Zemkoho (University of Southampton, UK)
- Dr. Emmanuel Kagning-Tsinda (Tohoku University, Japan)

## Overview

Primer C-VAE is a novel deep learning-based approach for designing primers to detect emerging virus variants, with particular focus on SARS-CoV-2. The methodology integrates convolutional neural networks and variational autoencoders to generate highly specific primers for variant detection.

## SARS-CoV-2 Virus Gene Sequence Data

### Datasets:
- GISAID (https://www.gisaid.org/)
- NCBI (https://www.ncbi.nlm.nih.gov/)
- ~~NGDC (https://big.ac.cn/ncov/?lang=en)~~


The detailed information about the SARS-CoV-2 sequence data used in this project can be found in:

**`SARS_CoV_2_Gene_sequence_info.md`**

### Data Organization

#### For SARS-CoV-2 virus (Homo Sapiens Host):
- Gene sequence data files downloaded from **GISAID**: If variant classification is completed, move to **`./Dataset/Variant_virus`** with correct variant types. Otherwise, use [Pangolin](https://cov-lineages.org/resources/pangolin.html) to determine variant type before moving.
- Gene sequence data files downloaded from **NCBI**: Use [Pangolin](https://cov-lineages.org/resources/pangolin.html) to determine variant type before moving to **`./Dataset/Variant_virus`**.

#### For SARS-CoV-2 virus (Non-Homo Sapiens Host):  
- Move files to **`./Dataset/other_virus/other_virus_seq`**

#### For other taxa (Homo Sapiens Host):
- Move files to **`./Dataset/other_virus/other_virus_seq`**

> **IMPORTANT NOTES:**
> 
> - The GISAID Dataset requires registration and login to download gene sequence data.
> - The NCBI Dataset does not provide classification of virus variants. If downloading from NCBI, use [Pangolin](https://cov-lineages.org/resources/pangolin.html) to implement dynamic nomenclature of SARS-CoV-2 lineages.
> - The NGDC Dataset is also popular for SARS-CoV-2 virus data, but network issues prevented downloads during project development.

![Pangolin Logo](https://github.com/cov-lineages/pangolin/raw/master/docs/logo.png)

## E.coli and S. flexneri Gene Sequence Data

### Dataset:
- NCBI (https://www.ncbi.nlm.nih.gov/)

### Automated Data Download
For convenience, shell scripts are provided to automatically download E.coli and S. flexneri gene sequences from NCBI:

- For E.coli sequences:
  ```
  ./GitHub/Primer_C-VAE/E.coli and S. flexneri_Data/download_NCBI/NCBI_E.coli/download_sequences.sh
  ```

- For Shigella flexneri sequences:
  ```
  ./GitHub/Primer_C-VAE/E.coli and S. flexneri_Data/download_NCBI/NCBI_Shigella_flexneri/download_sequences.sh
  ```

After downloading, the data organization and processing steps follow similar procedures to those described for SARS-CoV-2 data above. The same directory structure conventions should be followed to ensure compatibility with the analysis pipeline.

## Overall Pipeline and Primer C-VAE Architecture

![Overall Pipeline](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Overall_Pipeline.png?raw=true)

The Primer C-VAE methodology comprises four interconnected computational stages:

1. **Stage I (Data Acquisition and Pre-processing)**: Sequence acquisition from genomic repositories, systematic taxonomic annotation, and strategic data curation to establish high-quality training datasets.

2. **Stage II (Forward Primer Design)**: Implementation of our trained convolutional variational autoencoder architecture to generate initial primer candidates, followed by frequency distribution analysis and thermodynamic property assessment.

3. **Stage III (Reverse Primer Design)**: Analysis of downstream genomic regions adjacent to selected forward primer binding sites, applying the C-VAE model in a second iteration to generate complementary reverse primer candidates.

4. **Stage IV (In-silico PCR and Primer-BLAST Validation)**: Integration of selected forward and reverse primers into functional amplification pairs, evaluation of combinatorial properties, and validation through hierarchical assessment.

![Primer C-VAE Architecture](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Primer_C-VAE_architecture.png?raw=true)

Primer C-VAE architecture implements a specialized convolutional encoder framework for discriminating genomic features between target organisms and their variant populations. This deep learning system integrates three critical functional modules:

1. A multi-layer convolutional encoder that systematically extracts hierarchical sequence features from raw genomic data
2. A variational representation space where latent vectors $z$ are stochastically sampled via the reparameterization technique utilizing the learned distributional parameters $\mu$ and $\log\sigma^2$
3. A bifurcated computational pathway featuring both a classifier component for precise sequence categorization and a reconstruction decoder for generating sequence outputs

The architecture's training protocol optimizes these components simultaneously to maximize feature discrimination while preserving biological sequence integrity.

## Experiment 1: SARS-CoV-2 Emerging Variant Primer Design

### Forward Primer Design

#### Flowchart:

![Forward Primer Design](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Flowchart_Forward.jpg?raw=true)

After training the Primer C-VAE model for Forward Primer Design, you can use the **`other_code/confusion_matrix.py`** file to generate a confusion matrix and plot images to determine the accuracy of the model's classification results.

![Confusion Matrix](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Confusion_Matrix.png?raw=true)

### Reverse Primer Design

#### Flowchart:

![Reverse Primer Design](https://github.com/awc789/Primer_C-VAE/blob/main/pic/Flowchart_Reverse.jpg?raw=true)

#### Running the Code

**There are two ways to run the code:**

1. **Recommended: Using Amazon Web Services**
   - Upload data to AWS S3
   - Create a SageMaker Notebook
   - Clone this project:
     ```
     !git clone https://github.com/awc789/Primer_C-VAE.git
     ```
   - Install the required packages:
     ```
     !pip install -r requirements.txt
     ```
   - Run main.ipynb

2. **Running on local devices**
   - Download the data from GISAID/NCBI and this project
   - Modify the function 'S3_to_sageMaker' used for reading data
   - Run main.ipynb

> **IMPORTANT NOTES:**
> 
> - For primer synthesis information, see: [Integrated DNA Technologies (IDT)](https://www.idtdna.com/)
> - It is **NOT** recommended to use the **`other_code/online_validation.py`** file for In-Silico PCR at [UCSC In-Silico PCR](https://genome.ucsc.edu/cgi-bin/hgPcr).
> - For In-Silico PCR to check primer availability, use [FastPCR](https://primerdigital.com/fastpcr.html) or [Unipro UGENE](http://ugene.net/).

## Primer Distribution

![SARS-CoV-2 Primer Distribution](https://github.com/awc789/Primer_C-VAE/blob/main/pic/SARS-CoV-2%20Primer%20Distribution.png?raw=true)

![S.flexneri Primer Distribution](https://github.com/awc789/Primer_C-VAE/blob/main/pic/S.flexneri%20Primer%20Distribution.png?raw=true)

## Acknowledgement

We gratefully acknowledge the following Authors from the Originating laboratories responsible for obtaining the specimens and the Submitting laboratories where genetic sequence data were generated and shared via the GISAID Initiative, on which this research is based.

```
EPI_SET ID: EPI_SET_20220628va
DOI: https://doi.org/10.55876/gis8.220628va
```

## Reference

This project is updated in March 2025, based on the previous work of **Hanyu Wang**, **Emmanuel K. Tsinda**, **Anthony J. Dunn**, **Francis Chikweto**, **Nusreen Ahmed**, **Emanuela Pelosi** and **Alain B. Zemkoho**: [Deep learning forward and reverse primer design to detect SARS-CoV-2 emerging variants](https://arxiv.org/abs/2209.13591)