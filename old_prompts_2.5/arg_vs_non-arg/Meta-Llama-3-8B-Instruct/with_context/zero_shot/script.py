import csv
import time
import os
import glob
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. Load LLaMA3 model
# -----------------------------
cache_dir = "/DATA5/suyamoon/argmining/huggingface_cache"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading LLaMA3 model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded successfully.")


# -----------------------------
# 2. Helper to fetch case file
# -----------------------------
def get_case_file_content(filename):
    case_filename = filename.replace('.csv', '.txt')
    case_file_path = os.path.join('original_txt_files', case_filename)
    if os.path.exists(case_file_path):
        with open(case_file_path, 'r', encoding='latin-1') as f:
            return f.read()
    else:
        return "Complete case file not found."


# -----------------------------
# 3. Prediction using LLaMA3
# -----------------------------
def get_argumentative_prediction(text, filename):
    case_file_content = get_case_file_content(filename)

    prompt = f"""You are a Harvard-trained legal scholar with expertise in legal argumentation analysis. Your task is to classify legal text with the precision and analytical rigor expected in top-tier legal academia.

            CLASSIFICATION TASK:
            Determine whether the given legal text is "argumentative" or "non-argumentative" based on its rhetorical function within legal discourse.

            DEFINITIONS:
            • ARGUMENTATIVE: Text that advances reasoning, draws conclusions, makes inferences, or presents premises that support or refute a legal position. Contains logical connectors, causal relationships, or evaluative judgments.
            • NON-ARGUMENTATIVE: Text that merely recites facts, quotes statutes/regulations, describes procedures, or provides background information without advancing a position.

            ANALYTICAL FRAMEWORK:
            Look for these argumentative indicators:
            - Causal language ("therefore," "because," "consequently," "thus")
            - Evaluative terms ("must be rejected," "unfounded," "inadequately explained")
            - Logical reasoning chains that connect premises to conclusions
            - Comparative analysis or distinguishing of cases
            - Judicial reasoning that applies law to facts

            Look for these non-argumentative indicators:
            - Statutory citations without interpretation
            - Procedural descriptions
            - Factual recitations
            - Direct quotations of legal text
            - Administrative or clerical information

            CASE FILE CONTEXT:
            {case_file_content}

            INSTRUCTIONS:
            Apply your legal training to analyze the rhetorical function of this text. Consider whether it advances a legal argument or merely states information. 

            OUTPUT FORMAT:
            Respond with exactly one word: "argumentative" or "non-argumentative"

            TEXT TO ANALYZE:
            "{text}"
            
            Your classification:"""

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    response = outputs[0][input_ids.shape[-1]:]
    prediction = tokenizer.decode(response, skip_special_tokens=True).strip().lower()

    # Normalize output
    if 'argumentative' in prediction and 'non-argumentative' not in prediction:
        return 'argumentative'
    elif 'non-argumentative' in prediction:
        return 'non-argumentative'
    else:
        return '--'


# -----------------------------
# 4. Process CSV files
# -----------------------------
def process_csv_files():
    os.makedirs('predictions', exist_ok=True)
    csv_files = glob.glob('test/*.csv')

    if not csv_files:
        print("No CSV files found in 'test' folder!")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    for csv_file in csv_files:
        print(f"\n🔄 Processing file: {csv_file}")
        df = pd.read_csv(csv_file)

        if 'text' not in df.columns or 'class' not in df.columns:
            print(f"❌ Skipping {csv_file}: Missing 'text' or 'class' columns")
            continue

        output_filename = f"predictions/{os.path.basename(csv_file).replace('.csv', '_predictions.csv')}"

        start_idx = 0
        if os.path.exists(output_filename):
            existing_df = pd.read_csv(output_filename)
            start_idx = len(existing_df)
            print(f"📂 Found existing predictions file. Resuming from row {start_idx + 1}")
        else:
            with open(output_filename, 'w', newline='', encoding='latin-1') as f:
                writer = csv.writer(f)
                writer.writerow(['text', 'actual_class', 'actual_label', 'predicted_label'])

        total_rows = len(df)
        for idx in range(start_idx, total_rows):
            row = df.iloc[idx]
            text = row['text']
            actual_class = row['class']
            actual_label = 'argumentative' if actual_class == 1 else 'non-argumentative'

            prediction = get_argumentative_prediction(text, os.path.basename(csv_file))

            print(f"\nRow {idx+1}/{total_rows}")
            print(f"Text: {text[:100]}...")
            print(f"Actual: {actual_label} | Predicted: {prediction}")

            with open(output_filename, 'a', newline='', encoding='latin-1') as f:
                writer = csv.writer(f)
                writer.writerow([text, actual_class, actual_label, prediction])

            print(f"💾 Row {idx + 1} saved to {output_filename}")
            time.sleep(0.5)

        print(f"✅ Completed processing: {output_filename}")


# -----------------------------
# 5. Run script
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Legal Text Classification with LLaMA3")
    process_csv_files()
