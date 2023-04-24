import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import math


def generate_occupancy_map(width,
                           height,
                           num_objects,
                           outer_wall_width=10) -> np.ndarray:
    UNKNOWN_COLOR = 127
    FREE_SPACE_COLOR = 255
    WALL_COLOR = 0

    # Create an empty grayscale image
    img = Image.new("L", (width, height), color=UNKNOWN_COLOR)

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Generate walls at least 10 pixels from the edge of the images
    draw.rectangle([(outer_wall_width, outer_wall_width),
                    (width - outer_wall_width, height - outer_wall_width)],
                   fill=FREE_SPACE_COLOR,
                   width=2)

    # Generate random rectangles to simulate objects in the room
    for i in range(num_objects):
        # Define the rectangle coordinates
        cx, cy = random.randint(50,
                                width - 100), random.randint(50, height - 100)
        w, h = random.randint(20, 100), random.randint(20, 100)
        angle = math.radians(random.randint(0, 359))
        dx = math.cos(angle) * w / 2
        dy = math.sin(angle) * w / 2
        x1, y1 = int(cx - dx), int(cy - dy)
        x2, y2 = int(cx + dx), int(cy + dy)
        dx = math.sin(angle) * h / 2
        dy = math.cos(angle) * h / 2
        x3, y3 = int(cx + dx), int(cy - dy)
        x4, y4 = int(cx - dx), int(cy + dy)

        # Draw the rectangle
        draw.polygon([(x1, y1), (x4, y4), (x2, y2), (x3, y3)],
                     fill=UNKNOWN_COLOR)

    # Convolve the image with a 3x3 blur kernel
    img = img.filter(ImageFilter.Kernel((3, 3), np.ones(9) / 9))

    # Convert img to numpy array
    img = np.array(img)
    img[(img != FREE_SPACE_COLOR) & (img != UNKNOWN_COLOR)] = WALL_COLOR

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simulated occupancy map for a robot.")
    parser.add_argument("name", type=str, help="the name of the map")
    parser.add_argument("--save_folder",
                        type=Path,
                        default="./maps/",
                        help="the save folder")
    parser.add_argument("width", type=int, help="the width of the map")
    parser.add_argument("height", type=int, help="the height of the map")
    parser.add_argument("--seed", type=int, default=0, help="the random seed")
    # num objects
    parser.add_argument("--num_objects",
                        type=int,
                        default=2,
                        help="the number of objects")
    args = parser.parse_args()

    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generate the occupancy map
    img = generate_occupancy_map(args.width, args.height, args.num_objects)

    # Create the save folder if it does not exist
    args.save_folder.mkdir(parents=True, exist_ok=True)

    # Save the image using the specified name in the save folder
    save_path = args.save_folder / (args.name + ".png")
    # Save the map as an image
    Image.fromarray(img).save(save_path)
