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
# 3. Prediction using LLaMA3
# -----------------------------
def get_argumentative_prediction(text):

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

            EXEMPLARS FOR CALIBRATION:

            ARGUMENTATIVE (Premises/Conclusions):
            1. "In those circumstances, the second part of this ground of appeal must be rejected as, in part, unfounded and, in part, inadmissible."
            [Analysis: Contains evaluative conclusion "must be rejected" with reasoning "unfounded and inadmissible"]

            2. "The documents of the proceedings at first instance show that the argument put forward by the Hellenic Republic before the Court to the effect that the settled case-law of the Court relating to the concept of State aid, referred to in paragraph 45 of this judgment, is inapplicable to the present case because of the exceptional economic conditions experienced by the Hellenic Republic in 2009, was not put forward before the General Court."
            [Analysis: Presents evidence-based reasoning with causal connection "because of"]

            3. "At first instance, the Hellenic Republic complained that the Commission did not adequately explain, in the decision at issue, in what respect the compensation payments had conferred on the farmers concerned a competitive advantage affecting trade between Member States, and could, therefore, be classified as State aid, notwithstanding the serious crisis affecting the Greek economy at that time."
            [Analysis: Contains evaluative criticism "did not adequately explain" and logical inference "therefore"]

            NON-ARGUMENTATIVE (Factual/Procedural):
            1. "Under Article 3a of Law 1790/1988, in the version applicable to the dispute, the ELGA insurance scheme is compulsory and covers natural risks."
            [Analysis: Pure statutory description without interpretation or evaluation]

            2. "Point 1 of that communication states: '... The possibility under point 4.2 [of the TCF] to grant a compatible limited amount of aid does not apply to undertakings active in the primary production of agricultural products."
            [Analysis: Direct quotation of regulatory text without analysis]

            3. "By letter lodged at the Court Registry on 2 March 2015, the Greek Government requested, pursuant to the third subparagraph of Article 16 of the Statute of the Court of Justice of the European Union, that the Court sit as a Grand Chamber."
            [Analysis: Procedural/administrative description of court filing]

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

    return prediction
    # # Normalize output
    # if 'argumentative' in prediction and 'non-argumentative' not in prediction:
    #     return 'argumentative'
    # elif 'non-argumentative' in prediction:
    #     return 'non-argumentative'
    # else:
    #     return '--'


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
            with open(output_filename, 'w', newline='', encoding='latin-1' , errors='replace') as f:
                writer = csv.writer(f)
                writer.writerow(['text', 'actual_class', 'actual_label', 'predicted_label'])

        total_rows = len(df)
        for idx in range(start_idx, total_rows):
            row = df.iloc[idx]
            text = row['text']
            actual_class = row['class']
            actual_label = 'argumentative' if actual_class == 1 else 'non-argumentative'

            prediction = get_argumentative_prediction(text)

            print(f"\nRow {idx+1}/{total_rows}")
            print(f"Text: {text[:100]}...")
            print(f"Actual: {actual_label} | Predicted: {prediction}")

            with open(output_filename, 'a', newline='', encoding='latin-1', errors='replace') as f:
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
