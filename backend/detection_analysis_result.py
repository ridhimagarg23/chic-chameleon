from g4f.client import Client
import time

# Function to communicate with ChatGPT and get suggestions based on the inputs
def get_chatgpt_suggestions(gender, body_type, skin_tone):
    prompt = f"""
    Based on the following user details:
    - Gender: {gender}
    - Body Type: {body_type}
    - Skin Tone: {skin_tone}
    
    Please provide:
    1. A list or dictionary of **colors that suit** the user's skin tone along with their corresponding color codes (e.g., '#FF5733').
    2. A list or dictionary of **dress types** that are recommended based on the user's body type WITH PROMPTS GENERGTOR FOR IMAGE AI GENERATOR
    .
    3. A list of **at least 10 do's** related to the body type and skin tone.
    4. A list of **at least 10 don'ts** related to the body type and skin tone.
    
    Respond in a structured format as JSON:
    {{
        "colors_suited": [{{"color": "color_name", "code": "color_code"}}],
        "dress_recommendations": ["dress_type1", "dress_type2", ...],
        "dos": ["do1", "do2", ...],
        "donts": ["dont1", "dont2", ...]
    }}
    """
    
    retries = 5  # Number of retries before giving up
    for attempt in range(retries):
        try:
            # Initialize the g4f Client and get the response
            client = Client()
            response = client.chat.completions.create(
                model="gpt-4",  # You can change to any other available model
                messages=[{"role": "user", "content": prompt}]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(10 * (attempt + 1))  # Exponential backoff (wait longer between attempts)

    print("Max retries reached. Please try again later.")
    return None

# Example Input (User's details)
gender = "FEMALE"
body_type = "PEAR"
skin_tone = "FAIR"

# Get ChatGPT response
response = get_chatgpt_suggestions(gender, body_type, skin_tone)

if response:
    print("Response from ChatGPT:")
    print(response)
else:
    print("Failed to get a response.")
