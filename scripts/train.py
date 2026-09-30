from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

dataset = load_dataset("json", data_files="data/train_data.jsonl", split="train")

# Kept identical to the Modelfile's SYSTEM prompt on purpose — training on a different,
# shorter prompt than the one used at inference means the refusal instruction below is
# never actually reinforced by the fine-tune, just hoped for. The LoRA needs to see the
# "say you don't know" instruction paired with real refusal examples during training for
# it to survive alongside the confident factual answers the rest of the dataset teaches.
SYSTEM_PROMPT = (
    "You are Luna, a voice-controlled robotics assistant developed by Michael Harold Sony, "
    "Sreenandana, and Devadathan. You run fully offline on a Raspberry Pi 5. You only know "
    "about the specific people, facts, and departments you were trained on. If someone asks "
    "about a person, fact, or department you don't have information on, say clearly that you "
    "don't have that information — never invent a title, role, or detail for someone you "
    "don't recognize. If a question seems garbled, unclear, or doesn't quite make sense, say "
    "you didn't catch that and ask them to repeat it, rather than guessing at an answer."
)

def format(ex):
    # Train on the model's real chat template, not a made-up format, so what it
    # learns actually matches how Ollama/llama.cpp will prompt it at inference time.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["instruction"]},
        {"role": "assistant", "content": ex["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}
dataset = dataset.map(format)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=8, learning_rate=2e-4, fp16=False, bf16=True,
        output_dir="models/luna-brain-lora", logging_steps=5,
    ),
)
trainer.train()
model.save_pretrained("models/luna-brain-lora")
tokenizer.save_pretrained("models/luna-brain-lora")