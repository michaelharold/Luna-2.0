from unsloth import FastLanguageModel

# Reloads the base model + your trained LoRA adapter together, then writes out
# full merged weights so convert_hf_to_gguf.py has something complete to read.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/luna-brain-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)

model.save_pretrained_merged("models/luna-brain-merged", tokenizer, save_method="merged_16bit")
print("Merged model saved to models/luna-brain-merged")
