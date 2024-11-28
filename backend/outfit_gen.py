from concurrent.futures.thread import ThreadPoolExecutor
from typing import List
from g4f.client import Client
from bodyshape_detector import body_type
from skin_color_detection import skin_color_code
from gender_detection import gender
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient

from sympy.physics.units import temperature


tokens = [
        "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",  # First token
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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"
    ]
# Initialize the client
client = Client()


# Generate 5 Different Outfits
def generate_outfits(event_type: str, body_shape: str, gender: str) -> List[str]:
    prompt = f"""
        You are a fashion stylist. Recommend 5 different outfits for:
        Event: {event_type}, Body Shape: {body_shape}, Gender: {gender}.

        For each outfit, describe:
        Outfit Style Only
        Provide ONLY LIST WITHOUT numbering and NO EXTRA TEXT and bullets .
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,# Balance between creativity and coherence
        top_p=0.9,  # Nucleus sampling for diversity
        frequency_penalty=0.2,  # Slight penalty for repeated ideas
        presence_penalty=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.split("\n")  # Split lines for processing


# Generate 4 Color Options for Each Outfit
def generate_color_options(outfit: str, skin_tone: str, event_type: str) -> List[str]:
    prompt = f"""
        Based on this outfit {outfit} style
        Suggest 2 color options which suits best for  skin color code  {skin_tone}.in english 
        Provide only colora without numbering and NO EXTRA TEXT and bullets in format like this (comma seperated values)- Color1,Color2,Color3
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,  # Balance between creativity and coherence
        top_p=0.9,  # Nucleus sampling for diversity
        frequency_penalty=0.2,  # Slight penalty for repeated ideas
        presence_penalty=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.split("\n")  # Split lines for processing
def generate_color_outfits(outfit: str,colors: str, skin_tone: str, event_type: str) -> List[str]:
    prompt = f"""
        Based on this outfit: {outfit}
        Suggest outfit style color option {colors} for skin color: {skin_tone}. Include:
        - Outfit Style
        - Accessories
        - Makeup
        - Hair
        - Image generation prompt for this color: "A full-body view of a {outfit} in {colors}, suitable for {event_type}. The model should have Body shape - {body_shape} and skin color - {skin_tone}, The model should be {gender}."
        Provide a clean, list with no extra text.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,  # Balance between creativity and coherence
        top_p=0.9,  # Nucleus sampling for diversity
        frequency_penalty=0.2,  # Slight penalty for repeated ideas
        presence_penalty=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.split("\n")  # Split lines for processing


# Full Process to Generate All Outfits with Colors
def generate_all_colors(event_type: str, body_shape: str, gender: str, skin_tone: str):
    outfits = generate_outfits(event_type, body_shape, gender)
    genersated_outfits = []
    # Ensure outfits are generated
    while not outfits or outfits == ['']:
        print("X")
        outfits = generate_outfits(event_type, body_shape, gender)
    while not len(outfits) == 5:
        outfits = generate_outfits(event_type, body_shape, gender)

    current_token_index = 0  # Start with the first token
    request_count = 0  # Keep track of requests made with the current token
    max_requests_per_token = 2  # Limit of 3 requests per token

    colors_listss = []
    print("Generated Outfits:")
    for outfit in outfits:
        outfit = outfit.strip("1.")  # Clean whitespace
        if not outfit:
            continue

        print(f"- {outfit}")

        # Generate color options for the outfit
        colors = generate_color_options(outfit, skin_tone, event_type)
        while not colors or colors == [''] or colors == ['-']:
            colors = generate_color_options(outfit, skin_tone, event_type)

        print("   Color Options:")
        for color in colors:
            color = color.strip()
            if not color:
                continue

            print(f"   - {color}")
        colors_listss.append((colors))

            # Generate color-specific outfits
    colors_lists = []
    for sublist in colors_listss:
        for string in sublist:
            split_list = string.split(', ')
            colors_lists.append(split_list)
    print("Outfits", outfits)
    print("Colors", colors_lists)
    for outfit, color in zip(outfits, colors_lists):
            for colors in color:
                print("Color --", colors)
                color_outfits = generate_color_outfits(outfit, colors, skin_tone, event_type)
                while not color_outfits or color_outfits == ['']:
                    color_outfits = generate_color_outfits(outfit, colors, skin_tone, event_type)
                out=[]
                print("      Color Outfits:")
                prompt = color_outfits[-1]  # Use the outfit description as the prompt

                # Check if we need to switch tokens
                if request_count >= max_requests_per_token:
                    current_token_index = (current_token_index + 1) % len(tokens)  # Rotate to next token
                    request_count = 0  # Reset request count

                token = tokens[current_token_index]  # Use the current token
                client_image = InferenceClient("black-forest-labs/FLUX.1-schnell", token=token)
                image = client_image.text_to_image(prompt)
                image.show()  # Display the image

                request_count += 1
                for color_outfit in color_outfits:
                    color_outfit = color_outfit.strip()
                    out.append(color_outfit)
                    if color_outfit:
                        print(f"      - {color_outfit}")
    genersated_outfits.append(out)
    print(out)

    # Extracted data for image generation
    print(genersated_outfits)
    return genersated_outfits


# Example Usage
event_type = "indian wear for ethnic day in college"
body_shape = body_type
gender = "male"

skin_color = skin_color_code

extracted_outfits= generate_all_colors(event_type, body_shape, gender, skin_color)
