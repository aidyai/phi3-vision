from huggingface_hub import HfApi, HfFolder

# # Log in to Hugging Face using your token
# HfFolder.save_token("your_huggingface_token")

# Define your repository name (it should be unique to your account)
repo_name = "aidystark/insurance"

# Upload the model to the Hugging Face Hub
api = HfApi()
api.upload_folder(
    folder_path=save_model_path,  # Path to your saved model folder
    repo_id=repo_name,            # Your model repo name on Hugging Face Hub
    repo_type="model"
)