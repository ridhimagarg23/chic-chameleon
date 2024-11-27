from huggingface_hub import InferenceClient
from grrffffrrr import extracted_outfits
from PIL import Image
import random

def outfits_image_generator(extracted_outfits):
    tokens = [
        "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",  # First token
        "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",
        "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",
        "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",
        "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",
        "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",
        "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"
    ]
    # client = InferenceClient("black-forest-labs/FLUX.1-schnell", token="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")
    for color_wise_outfits in extracted_outfits:
        if isinstance(color_wise_outfits, list):
            prompt = color_wise_outfits[-1]# Check if the element is a list
            token = random.choice(tokens)  # Randomly choose a token for each request
            client = InferenceClient("black-forest-labs/FLUX.1-schnell", token=token)


    # Generate the image
        print(prompt)
        image = client.text_to_image(prompt)
        image.show()
import time

start_time = time.perf_counter()

# Your code block here
for i in range(1000000):
    pass



outfits_image_generator(extracted_outfits)
end_time = time.perf_counter()

total_time = end_time - start_time
print(f"Total time taken: {total_time:.6f} seconds")
# TOKENS
# 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
# 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
# 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
# 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
# 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
# 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
# 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