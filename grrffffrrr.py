from typing import List
from g4f.client import Client
from huggingface_hub import InferenceClient

tokens = [
         # First token
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
        - Outfit Style Only
        Provide a clean, list with no extra text.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.split("\n")  # Split lines for processing


# Generate 4 Color Options for Each Outfit
def generate_color_options(outfit: str, skin_tone: str, event_type: str) -> List[str]:
    prompt = f"""
        Based on this outfit: {outfit}
        Suggest 4 color options for skin color: {skin_tone}. Include:
        - Color
        Provide a clean, list with no extra text.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
        - Image generation prompt for this color: "A full body view of a [description of outfit] in [Color X], suitable for {event_type}. The model should have Body shape - {body_shape}  and skin color - {skin_tone}, and the outfit should include [accessories]. The model should be {gender}."
        Provide a clean, list with no extra text.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.split("\n")  # Split lines for processing


# Full Process to Generate All Outfits with Colors
def generate_all_colors(event_type: str, body_shape: str, gender: str, skin_tone: str):
    outfits = generate_outfits(event_type, body_shape, gender)
    genersated_outfits = []
    # Ensure outfits are generated
    while not outfits or outfits == ['']:
        outfits = generate_outfits(event_type, body_shape, gender)

    print("Generated Outfits:")
    for outfit in outfits:
        outfit = outfit.strip()  # Clean whitespace
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

            # Generate color-specific outfits
            color_outfits = generate_color_outfits(outfit, color, skin_tone, event_type)
            while not color_outfits or color_outfits == ['']:
                color_outfits = generate_color_outfits(outfit, color, skin_tone, event_type)
            out=[]
            print("      Color Outfits:")
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
event_type = "Wedding Reception"
body_shape = "rectangle"
gender = "male"
skin_color = "#e0ac69"

extracted_outfits= generate_all_colors(event_type, body_shape, gender, skin_color)
