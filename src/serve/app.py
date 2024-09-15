import argparse
from threading import Thread
import gradio as gr
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings

warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<|image_1|>"

def bot_streaming(message, history, generation_args):
    if message["files"]:
        # message["files"][-1] is a Dict or just a string
        if type(message["files"][-1]) == dict:
            image = message["files"][-1]["path"]
        else:
            image = message["files"][-1]
    else:
        # if there's no image uploaded for this turn, look for images in the past turns
        # kept inside tuples, take the last one
        for hist in history:
            if type(hist[0]) == tuple:
                image = hist[0][0]
    try:
        if image is None:
            # Handle the case where image is None
            raise gr.Error("You need to upload an image for Phi3-Vision to work. Close the error and try again with an Image.")
    except NameError:
        # Handle the case where 'image' is not defined at all
        raise gr.Error("You need to upload an image for Phi3-Vision to work. Close the error and try again with an Image.")

    conversation = []
    flag=False
    for user, assistant in history:
        if assistant is None:
            #pass
            flag=True
            conversation.extend([{"role": "user", "content":""}])
            continue
        if flag==True:
            conversation[0]['content'] = f"<|image_1|>\n{user}"   
            conversation.extend([{"role": "assistant", "content": assistant}])
            flag=False
            continue
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    if len(history) == 0:
        conversation.append({"role": "user", "content": DEFAULT_IMAGE_TOKEN + '\n' + message['text']})
    else:
        conversation.append({"role": "user", "content": message['text']})

    prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image = Image.open(image)
    inputs = processor(prompt, image, return_tensors="pt").to(device) 

    streamer = TextIteratorStreamer(processor.tokenizer, **{"skip_special_tokens": True, "skip_prompt": True, 'clean_up_tokenization_spaces':False,}) 
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

def main(args):

    global processor, model, device

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(model_base = args.model_base, model_path = args.model_path, 
                                                device_map=args.device, model_name=model_name, 
                                                load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                                device=args.device, use_flash_attn=use_flash_attn
    )

    chatbot = gr.Chatbot(scale=2)
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...",
                                  show_label=False)
    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args)

    with gr.Blocks(fill_height=True) as demo:
        gr.ChatInterface(
            fn=bot_streaming_with_args,
            title="Phi-3 Vision 128k Instruct",
            stop_btn="Stop Generation",
            multimodal=True,
            textbox=chat_input,
            chatbot=chatbot,
        )


    demo.queue(api_open=False)
    demo.launch(show_api=False, share=False, server_name='0.0.0.0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)