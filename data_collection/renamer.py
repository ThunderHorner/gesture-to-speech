import os
import cv2
import string

if __name__ == '__main__':
    training_images_path = '/home/thunderhorn/PycharmProjects/gesture-to-speech/training_images/'  # Update with your image path
    output_images_path = '/home/thunderhorn/PycharmProjects/gesture-to-speech/labeled_images/'  # Update with output path
    extracted_data = {}
    sequence = []  # Initialize a sequence to store 5 frames for each gesture
    letter_index = 0  # Start with 'a'
    label_list = list(string.ascii_lowercase)  # Label list for letters

    # Make sure output path exists
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    # Sort the image files to maintain sequence order
    image_files = sorted(os.listdir(training_images_path))

    for i in image_files:
        # Load the image
        image_path = os.path.join(training_images_path, i)
        image = cv2.imread(image_path)

        # Show the image
        cv2.imshow('Image', image)
        cv2.waitKey(1)  # Display the image

        # Get user input for the label of the image
        label = input(f"Enter label for image {i}: ")

        # Save the image with the new label as the filename
        new_image_name = f"{label}.jpg"
        new_image_path = os.path.join(output_images_path, new_image_name)
        cv2.imwrite(new_image_path, image)

        # Print confirmation
        print(f"Saved {i} as {new_image_name}")

    # Close all OpenCV windows
    cv2.destroyAllWindows()
