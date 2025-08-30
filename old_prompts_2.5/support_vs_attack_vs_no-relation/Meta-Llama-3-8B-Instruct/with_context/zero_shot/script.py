import csv
import time
import os
import glob
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. Load LLaMA3 model once
# -----------------------------
cache_dir = "/DATA5/suyamoon/argmining/huggingface_cache"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded successfully.")

# -----------------------------
# 2. Fixed system prompt
# -----------------------------


# -----------------------------
# 3. LLaMA3 inference function
# -----------------------------
def get_relation_prediction(source_text, target_text, file_name):
    system_prompt = """You are a Harvard-trained legal scholar with expertise in legal argumentation analysis. Your task is to analyze the relationship between two legal arguments with the precision and analytical rigor expected in top-tier legal academia.

CLASSIFICATION TASK:
Determine the relationship between the source argument and target argument. The relationship can be "support", "attack", or "no-relation".

DEFINITIONS:
• SUPPORT: The source argument provides evidence, reasoning, or justification that strengthens, reinforces, or validates the target argument. The source helps establish the credibility or validity of the target.
• ATTACK: The source argument contradicts, undermines, refutes, or weakens the target argument. The source challenges the validity or credibility of the target.
• NO-RELATION: The source and target arguments are independent, unrelated, or address different issues without any logical connection that would constitute support or attack.

ANALYTICAL FRAMEWORK:
Look for SUPPORT indicators:
- Source provides evidence for target's claims
- Source establishes legal precedent that validates target
- Source offers reasoning that strengthens target's position
- Logical flow where source builds toward target's conclusion

Look for ATTACK indicators:
- Source contradicts target's claims or reasoning
- Source provides counter-evidence to target
- Source establishes precedent that undermines target
- Logical inconsistency between source and target positions

Look for NO-RELATION indicators:
- Arguments address completely different legal issues
- No logical connection between the reasoning chains
- Independent factual statements without argumentative relationship

"""
    
    # Step 1: Read the corresponding .txt file
    txt_file_path = os.path.join('original_txt_files', f"{file_name}.txt")
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except FileNotFoundError:
        file_content = f"File '{file_name}.txt' not found in the 'original_txt_files' folder."

    system_prompt += f"\n\n## CASE FILE CONTENTS: {file_content}\n\n"
    system_prompt += """### LIMITATION:
- Do NOT just look for the sample words. Think about the meaning and the relationship.
- Only respond with one lowercase word: "support", "attack", "no-relation"
"""
    
    # Step 2: Update the user_prompt to include file content
    user_prompt = f"""
    CASE FILE CONTEXT:
{file_content}

SOURCE ARGUMENT:
"{source_text}"

TARGET ARGUMENT:
"{target_text}"

INSTRUCTIONS:
Apply your legal training to analyze how the source argument relates to the target argument. Consider whether the source strengthens, weakens, or has no bearing on the target's position.

OUTPUT FORMAT:
Respond with exactly one word: "support" or "attack" or "no-relation"

Your classification:"""

    messages = [
        {"role": "user", "content": system_prompt+ user_prompt},
    ]
    
    # Apply chat template
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
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    
    response = outputs[0][input_ids.shape[-1]:]
    prediction = tokenizer.decode(response, skip_special_tokens=True).strip().lower()
    
    return prediction


# -----------------------------
# 4. Processing CSV files (same as Gemini flow)
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
        
        if 'source_text' not in df.columns or 'target_text' not in df.columns or 'relation' not in df.columns:
            print(f"❌ Skipping {csv_file}: Missing 'source_text', 'target_text', or 'relation' columns")
            continue
        
        output_filename = f"predictions/{os.path.basename(csv_file).replace('.csv', '_predictions.csv')}"
        
        start_idx = 0
        if os.path.exists(output_filename):
            existing_df = pd.read_csv(output_filename)
            start_idx = len(existing_df)
            print(f"📂 Found existing predictions file. Resuming from row {start_idx + 1}")
        else:
            with open(output_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['source_text', 'target_text', 'actual_relation', 'predicted_relation'])
        
        total_rows = len(df)
        for idx in range(start_idx, total_rows):
            row = df.iloc[idx]
            source_text = row['source_text']
            target_text = row['target_text']
            actual_relation = row['relation']
            file_name = row['file_name']
            
            print(f"\nRow {idx+1}/{total_rows}")
            print(f"Source: {source_text[:50]}...")
            print(f"Target: {target_text[:50]}...")
            
            prediction = get_relation_prediction(source_text, target_text, file_name)
            print(f"Actual: {actual_relation} | Predicted: {prediction}")
            
            with open(output_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([source_text, target_text, actual_relation, prediction])
            
            print(f"💾 Saved to {output_filename}")
            time.sleep(0.5)  # slight delay
        
        print(f"✅ Completed processing: {output_filename}")

# -----------------------------
# 5. Run script
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Legal Text Classification with LLaMA3")
    process_csv_files()
