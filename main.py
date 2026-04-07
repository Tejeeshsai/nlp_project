# Make sure you have these installed:
# pip install datasets transformers torch sentencepiece
from datasets import load_dataset
from transformers import BartTokenizer

# 1. Load the CSVs you already created
try:
    datasets = load_dataset('csv', data_files={'train': 'train_dataset.csv', 'validation': 'validation_dataset.csv'})
except FileNotFoundError:
    print("Error: 'train_dataset.csv' or 'validation_dataset.csv' not found.")
    print("Please make sure these files are in the same folder as your script.")
    # Stop execution if files aren't found
    exit()

# 2. Load the BART tokenizer
# We'll use 'facebook/bart-base' as our starting model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# 3. Define the preprocessing function (NO PREFIX NEEDED for BART)
def preprocess_function(examples):
    # BART just takes the raw text
    inputs = [str(doc) for doc in examples['gen_z_cleaned']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    # Setup the "labels" (the correct answer)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer([str(doc) for doc in examples['normal_cleaned']], max_length=128, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 4. Apply this function to all our data
tokenized_datasets = datasets.map(preprocess_function, batched=True)

print("--- SUCCESS! Data is tokenized for BART. ---")
print(tokenized_datasets)