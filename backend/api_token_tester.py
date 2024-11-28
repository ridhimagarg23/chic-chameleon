from huggingface_hub import InferenceClient
import random

# Define extracted_outfits as input (example data)
extracted_outfits = [["A man in a black suit standing on a stage."]]


def outfits_image_generator(extracted_outfits):
    tokens = [

        "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",
        "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",
        "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",

    ]

    # Results storage
    results = {}

    for token in tokens:
        try:
            # Initialize client
            client = InferenceClient("black-forest-labs/FLUX.1-schnell", token=token)
            print(f"Testing token: {token}")

            # Generate image
            if isinstance(extracted_outfits, list) and extracted_outfits:
                prompt = extracted_outfits[0][-1]
                client.text_to_image(prompt)

                # Mark token as valid if no exception occurs
                results[token] = "Valid"
                print(f"Token {token} is valid.")
            else:
                results[token] = "No valid prompt"
                print(f"Invalid prompt format for token {token}.")
        except Exception as e:
            # Capture errors for this token
            results[token] = f"Invalid - {str(e)}"
            print(f"Error with token {token}: {e}")

    # Print final results
    print("\nFinal Token Results:")
    for token, status in results.items():
        print(f"Token: {token} - Status: {status}")


# Call the function
outfits_image_generator(extracted_outfits)
