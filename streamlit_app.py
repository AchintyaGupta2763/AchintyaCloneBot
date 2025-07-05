import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    peft_model_path = "server/outputModels/checkpoint-200"
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.eval()
    return model, tokenizer

st.title("🤖 Chat with Achintya Clone")
st.markdown("Talk to your AI trained on WhatsApp messages!")

model, tokenizer = load_model()

# Store conversation
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.history.append(f"Friend: {user_input}")
    
    # Build prompt with context
    context = "\n".join(st.session_state.history[-6:])  # Last 3 exchanges (user+bot)
    prompt = f"""### Instruction:
You are ACHINTYA GUPTA. Respond like how you talk on WhatsApp.

### Input:
{context}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip()

    st.session_state.history.append(f"ACHINTYA GUPTA: {response}")
    st.markdown(f"**Achintya:** {response}")
