import os
from rembg import remove
from PIL import Image, ImageOps

def preprocess_images(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the target size
    target_size = (896, 1024)
    target_width, target_height = target_size

    # Iterate over each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jpg")

            # Open the image and remove the background
            with open(input_path, 'rb') as img_file:
                img = Image.open(img_file)
                img = remove(img)

            # Convert to RGBA if necessary
            img = img.convert("RGBA")

            # Calculate the aspect ratio and resize to fit within the target size
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                # If the image is wider than the target aspect ratio
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
            else:
                # If the image is taller or square
                new_height = target_height
                new_width = int(new_height * aspect_ratio)

            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Create a white background image
            background = Image.new("RGB", target_size, (255, 255, 255))

            # Calculate position to paste the resized image centered
            offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
            background.paste(img, offset, img)  # Use img as a mask to keep transparency

            # Ensure the image fits well with minimum padding
            if new_width < target_width or new_height < target_height:
                background = ImageOps.fit(background, target_size, method=Image.LANCZOS, centering=(0.5, 0.5))

            # Save the final image as JPG
            background.save(output_path, "JPEG")

input_directory = 'anya_image'
output_directory = 'anya_image_out'
preprocess_images(input_directory, output_directory)

# def resize_and_fill_background(input_path, output_path, target_size=(896, 1024)):
#     # Open the image
#     img = Image.open(input_path)
    
#     # Convert to RGBA if not already (to handle transparency)
#     img = img.convert("RGBA")
    
#     # Calculate the aspect ratio and resize to fit within the target size
#     img_width, img_height = img.size
#     target_width, target_height = target_size
#     aspect_ratio = img_width / img_height

#     if aspect_ratio > (target_width / target_height):
#         # Fit to the target width
#         new_width = target_width
#         new_height = int(target_width / aspect_ratio)
#     else:
#         # Fit to the target height
#         new_height = target_height
#         new_width = int(target_height * aspect_ratio)

#     img = img.resize((new_width, new_height), Image.LANCZOS)

#     # Create a white background image
#     background = Image.new("RGB", target_size, (255, 255, 255))

#     # Calculate position to paste the resized image centered
#     offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
#     background.paste(img, offset, img)  # Use img as a mask to keep transparency

#     # Save the final image as JPG
#     background.save(output_path, "JPEG")

# # Path to the input and output files
# input_path = './anya_image/Anya_Forger_Manga_Full_Body_Uniform.webp'
# output_path = './anya_image_o/Anya_Forger_Manga_Full_Body_Uniform_resized.jpg'

# # Perform the resize and background fill
# resize_and_fill_background(input_path, output_path)