import sys
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments


print("--- 1. Loading and Tokenizing Data ---")

try:
    # 1. Load the CSVs
    datasets = load_dataset('csv', data_files={'train': 'train_dataset.csv', 'validation': 'validation_dataset.csv'})
except FileNotFoundError:
    print("Error: 'train_dataset.csv' or 'validation_dataset.csv' not found.")
    print("Please make sure these files are in the same folder as your script.")
    sys.exit() 

# 2. Load the BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# 3. Define the preprocessing function
def preprocess_function(examples):
    inputs = [str(doc) for doc in examples['gen_z_cleaned']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer([str(doc) for doc in examples['normal_cleaned']], max_length=128, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 4. Apply the function to create 'tokenized_datasets'
tokenized_datasets = datasets.map(preprocess_function, batched=True)

print(f"Data tokenized successfully: {tokenized_datasets}")


# --- STEP 2: TRAIN THE MODEL ---
print("--- 2. Setting up Training ---")

# 1. Load the pre-trained BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# 2. Define training arguments (using our last fix for your old version)
# 804 training samples / 8 batch size = 101 steps per epoch
steps_per_epoch = 101

training_args = TrainingArguments(
    output_dir="./bart-translator-results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./bart-logs",
    logging_steps=10,
    load_best_model_at_end=False, 
    do_eval=True,
    eval_steps=steps_per_epoch,
    save_steps=steps_per_epoch,
)

# 3. Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 4. Start training!
print("--- 3. Starting Training ---")
trainer.train()
print("--- 4. Training Complete ---")

print("Saving final model...")
model.save_pretrained("./my-genz-translator-bart")
tokenizer.save_pretrained("./my-genz-translator-bart")
print("Model saved to ./my-genz-translator-bart")