from typing import List, Dict
from g4f.client import Client

# Initialize the client for GPT-4
client = Client()


# First Prompt: Generate 5 Different Outfits
def generate_outfits(event_type: str, body_shape: str, gender: str) -> str:
    outfit_prompt_1 = f"""
        You are a fashion stylist. I want you to recommend four different outfits for the following details:
        Event type: {event_type}
        Body shape: {body_shape}
        Gender: {gender}

        For each outfit, include:
        - Outfit Style: [Description of the outfit]
        - Accessories: [Suggested accessories]
        - Makeup: [Suitable makeup style]
        - Hair: [Recommended hairstyle]

        Provide the response in a structured and organized format.
    """
    outfit_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": outfit_prompt_1}]
    )

    # Return the response from GPT (assuming the response contains the outfits data)
    return outfit_response.choices[0].message.content


# Second Prompt: Generate 4 Color Options for Each Outfit
def generate_color_options(outfit_details: Dict[str, str], skin_tone: str, event_type: str, body_shape: str,
                           gender: str) -> str:
    outfit_prompt_2 = f"""
        Based on the following outfit:
        - Outfit Style: {outfit_details['Outfit Style']}
        - Accessories: {outfit_details['Accessories']}
        - Makeup: {outfit_details['Makeup']}
        - Hair: {outfit_details['Hair']}

        Provide 4 color options for the outfit based on the user's skin tone: {skin_tone}. 
        For each color, specify:
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt for this color: "A full body view of a [description of outfit] in [Color X], suitable for {event_type}. The model should have {body_shape} and {skin_tone}, and the outfit should include [accessories]. The model should be {gender}."

        Provide the response in a structured and detailed format.
    """
    color_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": outfit_prompt_2}]
    )

    # Return the response from GPT (assuming the response contains the color options)
    return color_response.choices[0].message.content


# Example Usage
event_type = "Wedding Reception"
body_shape = "Hourglass"
gender = "Female"
skin_tone = "Fair"

# Step 1: Generate 5 Different Outfits
outfit_prompt = generate_outfits(event_type, body_shape, gender)
print("Outfits generated:\n", outfit_prompt)

# Simulated Response for Step 1 (Example data for one outfit)
outfit_details = {
    "Outfit Style": "An elegant evening gown with a mermaid cut and lace detailing.",
    "Accessories": "Diamond earrings, a silver clutch, and a bracelet.",
    "Makeup": "Smokey eyes with nude lips.",
    "Hair": "Loose curls with a side parting."
}

# Step 2: Generate 4 Color Options for the Outfit
color_options_prompt = generate_color_options(outfit_details, skin_tone, event_type, body_shape, gender)
print("\nColor options generated:\n", color_options_prompt)
