# Fine-Grained Lepidoptera Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Status](https://img.shields.io/badge/Status-In_Development-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning pipeline designed to distinguish between visually similar taxa of moths and butterflies (Lepidoptera). This project addresses the challenge of "noisy labels" in biological datasets by implementing a specialized pre-processing stage to filter distinct life stages (larvae vs. adults) before fine-grained classification.

The ultimate goal is to produce a deployment-ready model specialized for UK-native species, served via a scalable cloud architecture.

## Key Features
* **Massive Scale:** Leveraging a self-curated dataset of ~460,000 images scraped from iNaturalist.
* **Two-Stage Pipeline:** Implements a specialized "Noise Filtering" model (ResNet18) to automatically remove larval (caterpillar) images before training the final classifier.
* **UK Specialization:** Targeting high-precision identification of UK-native species by refining global feature representations.

## System Architecture

The pipeline consists of three main modules:

1.  **Data Ingestion & Cleaning (Complete)**
    * Custom scraping module utilizing Dask to process large iNaturalist metadata files.
    * Downloads and organizes images for the top 100 most frequently observed Lepidoptera taxa.
2.  **Noise Reduction (Complete)**
    * **Problem:** Raw biological data lumps all life stages (eggs, caterpillars, adults) into one category.
    * **Solution:** A binary classifier trained on a manually seeded dataset to separate "Adults" from "Larvae."
    * **Performance:** Achieving ~88% accuracy on validation splits, automating the cleaning of the raw dataset.
3.  **Species Classification (Current Phase)**
    * **Architecture:** EfficientNet-B3 backbone selected for fine-grained feature extraction.
    * **Status:** Currently tuning hyperparameters on the cleaned 'Adult' dataset to maximize validation accuracy across the 100 classes.

## Repository Structure

```text
.
├── data_processing/
│   ├── process_metadata.ipynb    # Downloads S3 metadata and filters for top 100 species
│   └── Download_photos.py        # Downloads actual images based on filtered metadata
├── adult_larva_classifier/
│   └── adult_larva_classifier.py # Trains the binary ResNet18 model (Larva vs Adult)
├── requirements.txt
└── README.md
