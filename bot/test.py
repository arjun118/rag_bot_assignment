# Load images
image_name = "./images/1482003986/dummy.jpg"
model_name = "./models/smolvlm_256_instruct"


# Load model directly
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(model_name)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_name},
            {"type": "text", "text": "describe and generate a caption for the image"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

# # Use a pipeline as a high-level helper

# from transformers import pipeline

# pipe = pipeline("image-text-to-text", model="./models/smolvlm_256_instruct")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "url": "./images/1482003986/dummy.jpg",
#             },
#             {"type": "text", "text": "generate a caption and 3 tags for this image"},
#         ],
#     },
# ]
# print(just before exec pipe)
# res = pipe(text=messages)
# print(res)
