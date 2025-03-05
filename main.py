import torch
import base64
import json
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def ocr_pdf_page(file_path):
    # Render page 1 to an image
    with open(file_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read())

    prompt = 'Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate.'
    # Build the full prompt
    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}


    # Generate the output
    output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=8192,
                num_return_sequences=1,
                do_sample=True,
            )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )

    print(text_output)
    return text_output[0]







FILE_PATH = "./paper.png"
print('poexal')
page_json = ocr_pdf_page(FILE_PATH)
data = json.loads(page_json)
with open(FILE_PATH, 'w') as text_file:
    text_file.write(data["natural_text"])