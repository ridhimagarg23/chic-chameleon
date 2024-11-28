import os
import requests

def remove_background(input_image_path, output_image_path):
    # Allowed image extensions
    allowed_extensions = ['.png', '.jpg', '.jpeg']
    
    # Check if the file has a valid extension
    _, file_extension = os.path.splitext(input_image_path)
    if file_extension.lower() not in allowed_extensions:
        print(f"Error: The file {input_image_path} has an unsupported file extension.")
        return
    
    # API key for remove.bg (replace with your actual key)
    api_key = 'FkcFUZ5J9PwdTf8bhLM4WQip'  # Replace with your remove.bg API key
    url = 'https://api.remove.bg/v1.0/removebg'

    # Check if input image exists
    if os.path.exists(input_image_path):
        with open(input_image_path, 'rb') as image_file:
            response = requests.post(
                url,
                files={'image_file': image_file},
                data={'size': 'auto'},  # You can specify the image size here
                headers={'X-Api-Key': api_key}
            )

        # Check if the API request was successful
        if response.status_code == requests.codes.ok:
            # Save the result image to the specified output path (overwrite if exists)
            with open(output_image_path, 'wb') as out_file:
                out_file.write(response.content)
        else:
            # If the request failed, print the error
            print(f"Error: {response.status_code}, {response.text}")
    else:
        print(f"Error: The file {input_image_path} does not exist.")


def find_image_in_frontend(extensions=['.png', '.jpg', 'image.jpeg']):
    # Define the folder path where images are stored
    folder_path = './assets'
    
    # Loop through all files in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file extension is in the allowed list
        if any(file_name.lower().endswith(ext) for ext in extensions):
            return os.path.join(folder_path, file_name)
    
    # If no image is found, return None
    return None


if __name__ == "__main__":
    # Find the first valid image file in the frontend/assets/ directory
    input_path = find_image_in_frontend()

    if input_path:        
        # Define output image path in the backend/static folder
        output_path = './static/image.png'  # Processed image will be saved here

        # Check if output file exists, and remove it if necessary
        if os.path.exists(output_path):
            os.remove(output_path)

        # Call the function to remove background
        remove_background(input_path, output_path)
    else:
        print("No valid image file found in frontend/assets/.")
