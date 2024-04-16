### This is a sample model ###
# DNA Sequence Classifier

This project utilizes deep learning models to classify regions of a DNA sequence into 'Gene' or 'Non-Gene'. It leverages a pre-trained model from Hugging Face's transformers library specifically designed for DNA sequence analysis.

## Prerequisites

Before running this script, ensure you have the following:
- Python 3.6 or higher
- PyTorch
- Transformers library by Hugging Face
- BioPython
Download the files:
Fasta file: 
GTF File: 
## Installation

To set up your environment to run the script, follow these steps:

1. Install PyTorch:
   ```bash
   pip install torch

Usage

Place your DNA sequence in a FASTA file at the specified path in the script.
Define the coordinates for gene regions within your sequence in the gene_coordinates list.
Run the script:
python dna_sequence_classifier.py

Script Overview

Tokenization: The script begins by loading a pre-trained tokenizer tailored for DNA sequences.
Model Configuration: Loads a configuration that sets up the model to classify sequences into two categories.
Sequence Classification Model: Initializes a model from the loaded configuration.
Gene Region Labeling: Defines a function to label parts of the sequence as 'Gene' or 'Non-Gene' based on specified coordinates.
Sequence Processing: Loads and processes a DNA sequence from a FASTA file.
Model Prediction: Feeds the tokenized sequence into the model and prints out prediction probabilities.
