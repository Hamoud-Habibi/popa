import torch
import base64
import urllib.request
import json
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import matplotlib.pyplot as plt
from olmocr.olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.olmocr.prompts import build_finetuning_prompt
from olmocr.olmocr.prompts.anchor import get_anchor_text
from PyPDF2 import PdfReader

model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
print('end')
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
print('start')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def ocr_pdf_page(pdf_path, page_number):
    # Render page 1 to an image
    image_base64 = render_pdf_to_base64png("./paper.pdf", page_number, target_longest_image_dim=1024)

    # Build the prompt, using document metadata
    anchor_text = get_anchor_text("./paper.pdf", 1, pdf_engine="pdfreport", target_length=4000)

    prompt = build_finetuning_prompt(anchor_text)

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




def get_pdf_page_count(pdf_path):
    """Gets the number of pages in a PDF file using PyPDF2.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        The number of pages in the PDF, or None if an error occurs.
    """
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        print(f"Error using PyPDF2: {e}")
        return None


# Example usage (assuming you have a file named 'paper.pdf'):
page_count = get_pdf_page_count("./paper.pdf")

if page_count:
  print(f"The PDF has {page_count} pages.")






FILE_PATH = "./paper.pdf"

NUM_PAGES = get_pdf_page_count(FILE_PATH)

pages = []

for i in range(NUM_PAGES):
    print(f"Processing page {i+1} of {NUM_PAGES}")
    page_json = ocr_pdf_page(FILE_PATH, i+1)
    pages.append(page_json)


for page_json in pages:
    try:
        data = json.loads(page_json)
        print(data["natural_text"])
    except:
        print(page_json)