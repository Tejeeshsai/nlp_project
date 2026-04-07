import sys
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate
import torch

# --- 1. Load Model and Tokenizer ---
print("--- 1. Loading Model and Tokenizer ---")
model_dir = "./my-genz-translator-bart"
try:
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)
except EnvironmentError:
    print(f"Error: Model not found at {model_dir}")
    print("Please run your 'fine_tune.py' script first to train and save the model.")
    sys.exit()

# --- 2. Load ROUGE Metric ---
print("--- 2. Loading ROUGE Metric ---")
rouge = evaluate.load("rouge")

# --- 3. Load Datasets ---
print("--- 3. Loading Datasets ---")
try:
    validation_dataset = load_dataset('csv', data_files={'validation': 'validation_dataset.csv'})['validation']
    train_dataset = load_dataset('csv', data_files={'train': 'train_dataset.csv'})['train']
except FileNotFoundError:
    print("Error: 'validation_dataset.csv' or 'train_dataset.csv' not found.")
    sys.exit()

print(f"Loaded {len(validation_dataset)} validation samples.")
print(f"Loaded {len(train_dataset)} training samples.")

# --- 4. Define Generation Function ---
def generate_translation(batch):
    inputs = tokenizer(
        batch["gen_z_cleaned"],
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    batch["prediction"] = decoded_preds
    batch["reference"] = batch["normal_cleaned"]
    return batch

# --- 5. Helper Function to Run Evaluation ---
def evaluate_dataset(dataset, description=""):
    print(f"--- Generating predictions for {description} set... ---")
    
    # Apply the generation function to the dataset
    result_dataset = dataset.map(generate_translation, batched=True, batch_size=8)
    
    print(f"--- Calculating ROUGE scores for {description} set... ---")
    
    # Extract predictions and references
    predictions = result_dataset["prediction"]
    references = result_dataset["reference"]

    # Clean up for ROUGE
    cleaned_preds = ["\n".join(pred.strip().split()) for pred in predictions]
    cleaned_labels = ["\n".join(label.strip().split()) for label in references]

    # Compute the final scores
    final_metrics = rouge.compute(
        predictions=cleaned_preds,
        references=cleaned_labels,
        use_stemmer=True
    )
    
    print(f"\n--- Final Metrics for {description} Set ---")
    for key, value in final_metrics.items():
        print(f"   {key}: {value * 100:.4f}") # Multiply by 100

    # Print a few examples
    print("\n--- Example Translations ---")
    for i in range(min(3, len(result_dataset))): # Show 3 examples
        print(f"   Example {i+1}")
        print(f"      GenZ: {result_dataset[i]['gen_z_cleaned']}")
        print(f"   Standard: {result_dataset[i]['reference']}")
        print(f"      Model: {result_dataset[i]['prediction']}")
    print("-" * 30)

# --- 6. Run Evaluation on Both Datasets ---
evaluate_dataset(validation_dataset, description="Validation (Test)")
evaluate_dataset(train_dataset, description="Training")