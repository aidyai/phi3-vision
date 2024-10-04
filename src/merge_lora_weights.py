from utils import get_model_name_from_path, load_pretrained_model
from huggingface_hub import HfApi, HfFolder

def merge_lora(model_path, model_base, save_model_path, safe_serialization):
    model_name = get_model_name_from_path(model_path)
    processor, model = load_pretrained_model(
        model_path=model_path, 
        model_base=model_base,
        model_name=model_name, 
        device_map='cpu',
    )

    if safe_serialization:
        state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if "wte" not in k}
        model.save_pretrained(save_model_path, state_dict=state_dict, safe_serialization=True)
        processor.save_pretrained(save_model_path)
    else:
        model.save_pretrained(save_model_path, safe_serialization=True)
        processor.save_pretrained(save_model_path)

# Set your parameters directly here
model_path = "/workspace/auto-insurance/checkpoint-348/"
#"/auto-insurance/checkpoint-348/adapter_model.safetensors"
model_base = "microsoft/Phi-3-vision-128k-instruct"
save_model_path = "./auto"
safe_serialization = True

# Run the merge process
merge_lora(model_path, model_base, save_model_path, safe_serialization)
