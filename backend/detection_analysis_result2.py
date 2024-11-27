from transformers import pipeline

# Load pre-trained model from Hugging Face
generator = pipeline('text-generation', model='gpt-3.5-turbo')  # or the correct model name you are using

def generate_outfits(event_type, skin_tone, body_shape, gender):
    # Define the prompt template
    prompt = f"""
    You are a fashion stylist. I want you to recommend five different outfits based on the following event and the user's skin tone, body shape, and gender. For each outfit, provide four color variations that suit the user's skin tone.

    Event type: {event_type}
    - **Daily Wear:** Comfortable and practical outfits for everyday use.
    - **Formal:** Office meetings, presentations, or business dinners.
    - **Informal:** Outings, casual gatherings, or weekend trips.
    - **College Wear:** Daily outfits for regular college days, emphasizing comfort and style.
    - **College Style:** Trendy looks for college fests, cultural events, or stylish group activities.
    - **Festive Wear:** Traditional or semi-formal outfits for celebrations like Diwali, Eid, or Christmas.
    - **Party Look:** Glamorous attire for night-outs, weddings, or clubbing.
    - **Special Occasions:** Birthdays, anniversaries, or personal milestone events.
    - **Marriage:** Traditional, ethnic, or fusion styles for weddings and related ceremonies.

    Skin tone color: {skin_tone}
    Body shape: {body_shape}
    Gender: {gender}

    For each outfit, generate four color variations and for each color provide the following:

    1. Color 1 (suitable for the skin tone): 
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt for this color: "A full body view of a [description of outfit] in [Color 1], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."

    2. Color 2 (suitable for the skin tone): 
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt for this color: "A full body view of a [description of outfit] in [Color 2], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."

    3. Color 3 (suitable for the skin tone): 
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt for this color: "A full body view of a [description of outfit] in [Color 3], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."

    4. Color 4 (suitable for the skin tone): 
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt for this color: "A full body view of a [description of outfit] in [Color 4], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."

    Repeat the above for **5 different outfits**.

    Output:
    1. Outfit 1 - Color Options (4 colors):
        - Color 1: [Recommended color for the skin tone]
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt: "A full body view of a [description of outfit] in [Color 1], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."

        - Color 2: [Recommended color for the skin tone]
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt: "A full body view of a [description of outfit] in [Color 2], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."

        - Color 3: [Recommended color for the skin tone]
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt: "A full body view of a [description of outfit] in [Color 3], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."

        - Color 4: [Recommended color for the skin tone]
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt: "A full body view of a [description of outfit] in [Color 4], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."

    2. Outfit 2 - Color Options (4 colors):
        - Color 1: [Recommended color for the skin tone]
        - Outfit Style: [Outfit Style for this color]
        - Accessories: [Accessories for this color]
        - Makeup: [Makeup for this color]
        - Hair: [Hair styling for this color]
        - Image generation prompt: "A full body view of a [description of outfit] in [Color 1], suitable for [event]. The model should have [body shape] and [skin tone] and the outfit should include [accessories]. The model should be [gender]."
        ...

    Repeat for all 5 outfits.
    """

    # Generate the response using the model
    generated_text = generator(prompt, max_length=500)[0]['generated_text']

    return generated_text

# Example usage
event_type = "Party Look"  # Example event type
skin_tone = "#F5D0A9"  # Example skin tone color
body_shape = "Hourglass"  # Example body shape
gender = "Female"  # Example gender

outfit_suggestions = generate_outfits(event_type, skin_tone, body_shape, gender)
print(outfit_suggestions)
