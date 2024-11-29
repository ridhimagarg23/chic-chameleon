from concurrent.futures.thread import ThreadPoolExecutor
from typing import List
from g4f.client import Client
# from bodyshape_detector import body_type
from skin_color_detection import skin_color_code
from gender_detection import gender
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient

from sympy.physics.units import temperature
33

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
def generate_outfits(event_type: str,event: str, body_shape: str, gender: str) -> List[str]:
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
def generate_color_outfits(outfit: str,colors: str, skin_tone: str, event_type: str,event: str) -> List[str]:
    prompt = f"""
        Based on this outfit: {outfit}
        Suggest outfit style color option {colors} for skin color: {skin_tone}. Include:
        - Outfit Style
        - Accessories
        - Makeup
        - Hair
        - Image generation prompt for this color: "A full-body view of a {outfit} in {colors}, suitable for {event}. The model should have Body shape - {body_shape} and skin color - {skin_tone}, The model should be {gender}."
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
def generate_all_colors(event_type: str,event: str, body_shape: str, gender: str, skin_tone: str):
    outfits = generate_outfits(event_type,event, body_shape, gender)
    genersated_outfits = []
    # Ensure outfits are generated
    while not outfits or outfits == ['']:
        print("X")
        outfits = generate_outfits(event_type,event, body_shape, gender)
    while not len(outfits) == 5:
        outfits = generate_outfits(event_type,event, body_shape, gender)

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
                color_outfits = generate_color_outfits(outfit, colors, skin_tone, event_type,event)
                while not color_outfits or color_outfits == ['']:
                    color_outfits = generate_color_outfits(outfit, colors, skin_tone, event_type,event)
                out=[]
                print("      Color Outfits:")
                print(color_outfits[-1])
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
event_types = {
    "Daily Wear": "Comfortable and practical outfits for everyday use.",
    "Formal": "Office meetings, presentations, or business dinners.",
    "Informal": "Outings, casual gatherings, or weekend trips.",
    "College Wear": "Daily outfits for regular college days, emphasizing comfort and style.",
    "College Style": "Trendy looks for college fests, cultural events, or stylish group activities.",
    "Festive Wear": "Traditional or semi-formal outfits for celebrations like Diwali, Eid, or Christmas.",
    "Party Look": "Glamorous attire for night-outs, weddings, or clubbing.",
    "Special Occasions": "Birthdays, anniversaries, or personal milestone events.",
    "Marriage": "Traditional, ethnic, or fusion styles for weddings and related ceremonies as a guest."
}
print("Please choose an event type:")
for index, key in enumerate(event_types.keys(), 1):
    print(f"{index}. {key}")

choice = int(input("Enter the number corresponding to your choice: "))

# Get the selected event type
selected_key = list(event_types.keys())[choice - 1]
selected_value = event_types[selected_key]

# Store the key and value as a single string
event_type = f"{selected_key}: {selected_value}"
event = f"{selected_key}"
body_shape = "pear"
gender = "female"

skin_color = skin_color_code

extracted_outfits= generate_all_colors(event_type, event, body_shape, gender, skin_color)
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List
from g4f.client import Client
from skin_color_detection import skin_color_code
from gender_detection import gender
from huggingface_hub import InferenceClient

# Tokens for API authentication
tokens = [
    "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",
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

# Function to generate 5 different outfits
def generate_outfits(event_type: str, event: str, body_shape: str, gender: str) -> List[str]:
    """
    Generate 5 outfit styles based on event type, body shape, and gender.
    
    Parameters:
        event_type (str): The type of event (e.g., Formal, Casual).
        event (str): Specific event details.
        body_shape (str): Body shape category.
        gender (str): Gender of the individual.

    Returns:
        List[str]: List of 5 outfit styles.
    """
    prompt = f"""
        You are a fashion stylist. Recommend 5 different outfits for:
        Event: {event_type}, Body Shape: {body_shape}, Gender: {gender}.

        For each outfit, describe:
        Outfit Style Only
        Provide ONLY LIST WITHOUT numbering and NO EXTRA TEXT and bullets.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.split("\n")  # Split lines for processing

# Function to generate color options for an outfit
def generate_color_options(outfit: str, skin_tone: str, event_type: str) -> List[str]:
    """
    Suggest color options for an outfit based on skin tone and event type.

    Parameters:
        outfit (str): Outfit description.
        skin_tone (str): Skin tone code.
        event_type (str): Type of event.

    Returns:
        List[str]: List of color options.
    """
    prompt = f"""
        Based on this outfit {outfit} style
        Suggest 2 color options which suit best for skin color code {skin_tone}. Provide output in format - Color1,Color2.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.split(",")

# Function to generate detailed color-specific outfit options
def generate_color_outfits(outfit: str, colors: str, skin_tone: str, event_type: str, event: str, body_shape: str, gender: str) -> List[str]:
    """
    Provide detailed suggestions for an outfit in specific colors, including accessories, makeup, and hair.

    Parameters:
        outfit (str): Base outfit description.
        colors (str): Selected color.
        skin_tone (str): Skin tone code.
        event_type (str): Type of event.
        event (str): Specific event details.
        body_shape (str): Body shape.
        gender (str): Gender.

    Returns:
        List[str]: Detailed outfit suggestions.
    """
    prompt = f"""
        Based on this outfit: {outfit}
        Suggest details for {colors} color option for skin color: {skin_tone}.
        Include:
        - Outfit Style
        - Accessories
        - Makeup
        - Hair
        - Image generation prompt: "A full-body view of a {outfit} in {colors}, suitable for {event}. The model should have Body shape - {body_shape}, skin color - {skin_tone}, and be {gender}."
        Provide a clean list with no extra text.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.split("\n")

# Main function to generate all outfits with colors
def generate_all_colors(event_type: str, body_shape: str, gender: str, skin_tone: str):
    """
    Generate outfits and detailed suggestions for specific colors.

    Parameters:
        event_type (str): Type of event.
        event (str): Specific event details.
        body_shape (str): Body shape category.
        gender (str): Gender of the individual.
        skin_tone (str): Skin tone code.

    Returns:
        List[List[str]]: Detailed outfit suggestions.
    """
    event_types = {
    "Daily Wear": "Comfortable and practical outfits for everyday use.",
    "Formal": "Office meetings, presentations, or business dinners.",
    "Informal": "Outings, casual gatherings, or weekend trips.",
    "College Wear": "Daily outfits for regular college days, emphasizing comfort and style.",
    "College Style": "Trendy looks for college fests, cultural events, or stylish group activities.",
    "Festive Wear": "Traditional or semi-formal outfits for celebrations like Diwali, Eid, or Christmas.",
    "Party Look": "Glamorous attire for night-outs, weddings, or clubbing.",
    "Special Occasions": "Birthdays, anniversaries, or personal milestone events.",
    "Marriage": "Traditional, ethnic, or fusion styles for weddings and related ceremonies as a guest."
    } 
    if event_type in event_types.keys():
        event = event_type
        event_type = f"{event_type}: {event_types[event_type]}"

    outfits = generate_outfits(event_type, event, body_shape, gender)
    while not outfits or len(outfits) != 5:
        outfits = generate_outfits(event_type, event, body_shape, gender)

    generated_outfits = []
    current_token_index = 0
    request_count = 0
    max_requests_per_token = 2

    for outfit in outfits:
        colors = generate_color_options(outfit, skin_tone, event_type)
        for color in colors:
            detailed_outfits = generate_color_outfits(outfit, color, skin_tone, event_type, event, body_shape, gender)

            # Image generation
            prompt = detailed_outfits[-1]  # Extract image prompt
            if request_count >= max_requests_per_token:
                current_token_index = (current_token_index + 1) % len(tokens)
                request_count = 0

            token = tokens[current_token_index]
            client_image = InferenceClient("black-forest-labs/FLUX.1-schnell", token=token)
            image = client_image.text_to_image(prompt)
            image.show()
            request_count += 1

            generated_outfits.append(detailed_outfits)

    return generated_outfits

# Example Usage
if __name__ == "__main__":
    event_types = ["Daily Wear", "Formal", "Informal","Daily Wear", "College Style","Festive Wear","Party Look","Special Ocassions","Marriage"]
    print("Please choose an event type:")
    for event in event_types:
        print(f"- {event}")
    event_choice = int(input("Enter the number corresponding to your choice: "))
    event_type = f"{event_type[event_choice-1]}"
    body_shape = "Hourglass"
    gender = "Female"
    skin_tone = "Medium"

    all_outfits = generate_all_colors(event_type, body_shape, gender, skin_tone)
    print("Generated Outfits with Details:", all_outfits)
