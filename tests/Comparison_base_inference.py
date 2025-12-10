import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_DIR = "./new_qwen_intent" 
BASE_MODEL = "Qwen/Qwen2.5-3B"

STRATEGY_PROMPT = """
INSTRUCTIONS:
You are a skilled negotiator. Your goal is to maximize the selling price.
1. Do NOT lower your price immediately.
2. If the buyer offers a low price, persist on your target price.
3. You must reject low offers at least 1-2 times before making a significant drop.
"""

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ------------------------------------------------------------------
# NEW CLASS: STOPPING CRITERIA
# ------------------------------------------------------------------
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token matches any of our stop tokens
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def load_models():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (4-bit)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    
    return model, tokenizer

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def safe(x):
    return "" if x is None else str(x)

def format_scenario_meta_inference(scenario: dict) -> str:
    cat = safe(scenario.get("category"))
    item_name = scenario.get("item_name")
    item_desc = scenario.get("item_description")
    list_price = scenario.get("list_price", "N/A") 
    
    item_lines = []
    item_lines.append(f"- Name: {safe(item_name)} | List price: ${safe(list_price)}")
    if item_desc:
        item_lines.append(f"  Description: {safe(item_desc)}")

    buyer_target = safe(scenario.get("buyer_target_price"))
    seller_target = safe(scenario.get("seller_target_price"))
    buyer_bottom = safe(scenario.get("buyer_bottomline"))
    seller_bottom = safe(scenario.get("seller_bottomline"))

    meta = [
        f"Category: {cat}",
        "Items:",
        *item_lines,
        f"Buyer target price: {buyer_target}",
        f"Seller target price: {seller_target}",
        f"Buyer bottomline: {buyer_bottom}",
        f"Seller bottomline: {seller_bottom}",
    ]
    return "\n".join(meta)

def build_inference_prompt(scenario, history, new_buyer_msg):
    scenario_block = format_scenario_meta_inference(scenario)
    hist_lines = []
    for role, text in history:
        clean_text = text.replace("\n", " ").strip()
        hist_lines.append(f"{role}: {clean_text}")
    hist_lines.append(f"Buyer: {new_buyer_msg.replace('\n', ' ').strip()}")
    history_block = "\n".join(hist_lines) + "\nSeller:"
    
    prompt = (
        f"{STRATEGY_PROMPT}\n\n"
        "SCENARIO:\n"
        f"{scenario_block}\n\n"
        "NEGOTIATION HISTORY:\n"
        f"{history_block}"
    )
    return prompt

def parse_output(decoded_text):
    # CLEANUP: If the model generated "Buyer:" despite our best efforts, cut it off.
    if "\nBuyer:" in decoded_text:
        decoded_text = decoded_text.split("\nBuyer:")[0]

    if "Intent:" not in decoded_text:
        return decoded_text.strip(), "unknown (format missing)"

    try:
        parts = decoded_text.split("Intent:")
        intent = parts[-1].strip()
        pre_intent = parts[-2] 
        if "Seller:" in pre_intent:
            message = pre_intent.split("Seller:")[-1].strip()
        else:
            message = pre_intent.strip()
        return message, intent
    except:
        return decoded_text, "parsing_error"

# ------------------------------------------------------------------
# GENERATION FUNCTION
# ------------------------------------------------------------------
def generate_comparison(model, tokenizer, scenario, history, buyer_msg):
    prompt = build_inference_prompt(scenario, history, buyer_msg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # DEFINE STOP WORDS
    # We want to stop if the model generates a new line "\n" (which usually precedes "Buyer:")
    # or the token for "Buyer".
    stop_words = ["\nBuyer", "\n", "Buyer:"]
    stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
    stop_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

    # 1. BASELINE (RAW QWEN)
    with model.disable_adapter():
        with torch.no_grad():
            out_base = model.generate(
                **inputs,
                max_new_tokens=80, # Keep short for base model to reduce hallucinations
                temperature=0.7,
                do_sample=True,
                stopping_criteria=stop_criteria, # <--- FORCE STOP
                pad_token_id=tokenizer.eos_token_id
            )
    
    # 2. FINETUNED
    with torch.no_grad():
        out_ft = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
            # Finetuned usually respects EOS, so we might not need strict stopping, 
            # but it doesn't hurt to add it if you see issues.
        )

    # DECODING AND PARSING
    prompt_len = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    
    decoded_base = tokenizer.decode(out_base[0], skip_special_tokens=True)
    # Basic cleanup: remove the prompt part
    base_raw = decoded_base[prompt_len:].strip()
    base_msg, base_int = parse_output(base_raw)

    decoded_ft = tokenizer.decode(out_ft[0], skip_special_tokens=True)
    ft_raw = decoded_ft[prompt_len:].strip()
    ft_msg, ft_int = parse_output(ft_raw)

    return (base_msg, base_int), (ft_msg, ft_int)

# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------
def main():
    model, tokenizer = load_models()

    print("\nPaste scenario JSON OR press Enter for default.")
    raw = input("Scenario JSON: ").strip()

    if raw:
        scenario = json.loads(raw)
    else:
        scenario = {
            "category": "electronics",
            "item_name": "iPhone 12 (128GB)",
            "item_description": "Used, good battery",
            "list_price": 500,
            "seller_target_price": 420,
            "seller_bottomline": 380,
            "buyer_target_price": 350,
            "buyer_bottomline": 400
        }

    history = [] 

    print(f"\nNegotiation started.")
    print("---------------------------------------------------------------")
    print("COMPARISON MODE: [Baseline Qwen] vs [Fine-Tuned Intent Adapter]")
    print("---------------------------------------------------------------")
    print("Type 'exit' to quit.\n")

    while True:
        buyer_msg = input("Buyer: ").strip()
        if buyer_msg.lower() == "exit":
            break

        (base_msg, base_int), (ft_msg, ft_int) = generate_comparison(
            model, tokenizer, scenario, history, buyer_msg
        )

        print("\n" + "="*60)
        print(f" BASELINE MODEL (Qwen 3B Raw)")
        print("-" * 60)
        print(f"Seller: {base_msg}")
        print(f"Intent: {base_int}")
        
        print("\n" + "-"*60)
        print(f" FINETUNED MODEL (Your Adapter)")
        print("-" * 60)
        print(f"Seller: {ft_msg}")
        print(f"Intent: {ft_int}")
        print("="*60 + "\n")

        history.append(("Buyer", buyer_msg))
        history.append(("Seller", ft_msg))

if __name__ == "__main__":
    main()