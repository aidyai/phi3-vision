from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

# Model ID
model_id = "aidystark/insurance"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    # _attn_implementation='flash_attention_2'  # Change this to 'eager' if needed
)

# Load processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=4  # You can adjust this based on performance
)

# Load a single image
url = "./dm.jpeg"
image = Image.open(url)

# Set up the placeholder for the single image
placeholder = "<|image_1|>\n"

# Prepare the message with the placeholder and request
messages = [
    {"role": "user", "content": placeholder + "Identify the damaged components in the vehicle's image"}
]

# Prepare the prompt
prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Process the image and input prompt for the model
inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

# Define generation arguments
generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.0,
    "do_sample": False
}

# Generate the output
generate_ids = model.generate(
    **inputs,
    eos_token_id=processor.tokenizer.eos_token_id,
    **generation_args
)

# Remove input tokens from the generated output
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

# Decode and print the response
response = processor.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(response)
