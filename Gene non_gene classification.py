import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from Bio import SeqIO

# Load a pre-trained tokenizer for DNA sequences from the Hugging Face model repository.
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Load a configuration for the sequence classification model specifying it should classify into two categories.
config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", num_labels=2)

# Initialize a model for sequence classification from the configuration loaded above.
model = AutoModelForSequenceClassification.from_config(config)

# Define a function to label parts of a DNA sequence as 'Gene' or 'Non-Gene' based on given coordinates.
def filter_for_gene_regions(sequence, gene_coordinates):
    labeled_sequence = []
    for i in range(len(sequence)):  # Iterate through each base in the sequence.
        label = 'Non-Gene'  # Default label for each base.
        for start, end in gene_coordinates:  # Check each base against the provided gene coordinates.
            if start <= i < end:  # If the base index falls within a gene region, label it as 'Gene'.
                label = 'Gene'
                break
        labeled_sequence.append((sequence[i], label))  # Append the base and its label to the list.
    return labeled_sequence

# Specify the path to the FASTA file containing the DNA sequence.
fasta_file = '/path/to/your/sequence/file.fa'
# Define coordinates marking gene regions in the sequence.
gene_coordinates = [(62446549, 62446939), (62447028, 62447186), (62447846, 62447998)]

# Load and convert the FASTA file into a dictionary of sequences using BioPython.
fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
# Select the first sequence from the dictionary to use as an example.
sequence_id = next(iter(fasta_sequences))
sequence = str(fasta_sequences[sequence_id].seq).upper()  # Convert the sequence to uppercase.

# Apply the labeling function to the sequence to mark each base as 'Gene' or 'Non-Gene'.
labeled_sequence = filter_for_gene_regions(sequence, gene_coordinates)

# Tokenize the labeled sequence, converting it into input IDs for the model.
inputs = tokenizer([base for base, _ in labeled_sequence], return_tensors='pt', truncation=True, max_length=512)["input_ids"]

# Disable gradient calculations to save memory and compute, as they are not needed for inference.
with torch.no_grad():
    outputs = model(inputs)  # Run the model to get output logits.
    # Apply softmax to the logits to convert them into probabilities for each class ('Gene' or 'Non-Gene').
    predictions = torch.softmax(outputs.logits, dim=-1)
    print(predictions)  # Print the prediction probabilities.
