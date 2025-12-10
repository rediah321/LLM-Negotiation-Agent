import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_DIR = "./new_qwen_intent"
BASE_MODEL = "Qwen/Qwen2.5-3B"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ------------------------------------------------------------------
# PROMPT TEMPLATES (The "Systematic Variations")
# ------------------------------------------------------------------

# 1. CONTROL PROMPT (Standard Seller, Explicit Constraints)
PROMPT_CONTROL = """
INSTRUCTIONS:
You are a skilled Seller. Your goal is to maximize the selling price.
1. Do NOT lower your price immediately.
2. You have specific target and bottomline prices in the scenario. Stick to them.
3. Reject low offers at least once.
"""

# 2. ROLE VARIANT (Different Role Formulation: "Agent")
PROMPT_ROLE_AGENT = """
INSTRUCTIONS:
You are an autonomous Negotiation Agent acting on behalf of a client.
1. Maintain a professional, detached tone.
2. Your primary directive is Asset Value Maximization.
3. Do not yield to pressure. execute counter-offers logically.
"""

# 3. IMPLICIT CUES (Formatting Variation + No Explicit Limits)
PROMPT_IMPLICIT = """
INSTRUCTIONS:
You are selling a personal item.
1. You do not have a hard spreadsheet of numbers. Use your common sense.
2. Look at the Item Description and List Price to decide what is fair.
3. If the buyer seems nice, you can be flexible. If they are rude, hold firm.
"""

def load_model():
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
# HELPER: Scenario Formatting (Dynamic based on Variant)
# ------------------------------------------------------------------
def safe(x):
    return "" if x is None else str(x)

def format_scenario(scenario: dict, mode="explicit") -> str:
    """
    mode='explicit': Includes Target and Bottomline (Standard).
    mode='implicit': Hides Target/Bottomline to test implicit reasoning.
    """
    cat = safe(scenario.get("category"))
    item_name = scenario.get("item_name")
    item_desc = scenario.get("item_description")
    list_price = scenario.get("list_price", "N/A") 
    
    item_lines = [f"- Name: {safe(item_name)} | List price: ${safe(list_price)}"]
    if item_desc:
        item_lines.append(f"  Description: {safe(item_desc)}")

    meta = [f"Category: {cat}", "Items:", *item_lines]

    if mode == "explicit":
        # EXPLICIT CUES
        meta.append(f"Seller target price: {safe(scenario.get('seller_target_price'))}")
        meta.append(f"Seller bottomline: {safe(scenario.get('seller_bottomline'))}")
    else:
        # IMPLICIT CUES ONLY
        meta.append("Seller target price: (Decide yourself based on list price)")
        meta.append("Seller bottomline: (Decide yourself based on item value)")

    return "\n".join(meta)

# ------------------------------------------------------------------
# BUILD PROMPT (The "Prompt Engineering" Layer)
# ------------------------------------------------------------------
def build_variant_prompt(scenario, history, new_buyer_msg, variant_type):
    
    # 1. Choose System Instruction & Scenario Mode
    if variant_type == "control":
        sys_prompt = PROMPT_CONTROL
        scen_text = format_scenario(scenario, mode="explicit")
        role_label = "Seller"
    elif variant_type == "agent_role":
        sys_prompt = PROMPT_ROLE_AGENT
        scen_text = format_scenario(scenario, mode="explicit")
        role_label = "Agent" # Testing if changing role label breaks the model
    elif variant_type == "implicit":
        sys_prompt = PROMPT_IMPLICIT
        scen_text = format_scenario(scenario, mode="implicit")
        role_label = "Seller"

    # 2. Build History (Testing Dialogue Formatting Robustness)
    hist_lines = []
    for r, text in history:
        clean_text = text.replace("\n", " ").strip()
        # If we are testing "Agent" role, we might rename historical labels too
        if variant_type == "agent_role" and r == "Seller":
            hist_lines.append(f"Agent: {clean_text}")
        else:
            hist_lines.append(f"{r}: {clean_text}")

    hist_lines.append(f"Buyer: {new_buyer_msg.replace('\n', ' ').strip()}")
    
    # 3. Assemble
    history_block = "\n".join(hist_lines) + f"\n{role_label}:"

    full_prompt = (
        f"{sys_prompt}\n\n"
        "SCENARIO:\n"
        f"{scen_text}\n\n"
        "NEGOTIATION HISTORY:\n"
        f"{history_block}"
    )
    return full_prompt, role_label

def parse_output(decoded_text, role_label="Seller"):
    if "Intent:" not in decoded_text:
        return decoded_text.strip(), "unknown"

    try:
        parts = decoded_text.split("Intent:")
        intent = parts[-1].strip()
        pre_intent = parts[-2]
        
        # Dynamic splitting based on role label (Seller vs Agent)
        if f"{role_label}:" in pre_intent:
            message = pre_intent.split(f"{role_label}:")[-1].strip()
        else:
            message = pre_intent.strip()
        return message, intent
    except:
        return decoded_text, "parsing_error"

# ------------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------------
def generate_variant(model, tokenizer, scenario, history, buyer_msg, variant_type):
    prompt, role_label = build_variant_prompt(scenario, history, buyer_msg, variant_type)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id, 
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Be robust in parsing: we strip the input prompt from output if needed, 
    # but parse_output usually handles the tail well.
    msg, intent = parse_output(decoded, role_label)
    return msg, intent

# ------------------------------------------------------------------
# MAIN COMPARATIVE LOOP
# ------------------------------------------------------------------
def main():
    model, tokenizer = load_model()

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

    print(f"\nComparative Analysis Started.")
    print("----------------------------------------------------------------")
    print("Testing robustness across: [Control] vs [Agent Role] vs [Implicit]")
    print("----------------------------------------------------------------")

    while True:
        buyer_msg = input("\nBuyer: ").strip()
        if buyer_msg.lower() == "exit":
            break

        print("\n" + "="*80)
        
        # 1. CONTROL (Standard)
        msg_c, int_c = generate_variant(model, tokenizer, scenario, history, buyer_msg, "control")
        print(f" [A] CONTROL (Explicit 'Seller')")
        print(f" Msg:    {msg_c}")
        print(f" Intent: {int_c}")
        print("-" * 80)

        # 2. ROLE VARIANT
        msg_a, int_a = generate_variant(model, tokenizer, scenario, history, buyer_msg, "agent_role")
        print(f" [B] ROLE VARIANT ('Professional Agent')")
        print(f" Msg:    {msg_a}")
        print(f" Intent: {int_a}")
        print("-" * 80)

        # 3. IMPLICIT VARIANT
        msg_i, int_i = generate_variant(model, tokenizer, scenario, history, buyer_msg, "implicit")
        print(f" [C] IMPLICIT SCENARIO (Hidden Limits)")
        print(f" Msg:    {msg_i}")
        print(f" Intent: {int_i}")
        
        print("="*80)

        # FOR CONTINUITY: We must choose ONE history track to continue the chat.
        # We default to Control so the chat doesn't get too weird, 
        # but you can swap this to test how 'Agent' sustains a convo.
        history.append(("Buyer", buyer_msg))
        history.append(("Seller", msg_c)) 

if __name__ == "__main__":
    main()