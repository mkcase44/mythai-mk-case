import os
import json
import shutil
import imageio
import numpy as np
import matplotlib.pyplot as plt


def read_ndjson_file(file_path: str) -> dict:
    """
    Reads a dataset from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the dataset.

    Returns:
        dict: The dataset loaded from the JSON file.
    """
    dataset = []
    with open(file_path, 'r') as ndjson_file:
        for data_line in ndjson_file:
            data = json.loads(data_line)
            dataset.append(data)    
    return dataset

def get_subset(dataset: list, indices_json_path: str, subset_name: str) -> list:
    """
    Extracts a subset of a dataset based on indices specified in a JSON file.
    Args:
        dataset (list): The complete dataset from which to extract the subset.
        indices_json_path (str): Path to the JSON file containing indices for subsets.
        subset_name (str): The key in the JSON file specifying which subset to extract.
    Returns:
        list: The subset of the dataset corresponding to the provided subset name.
    """
    with open(indices_json_path, 'r') as indices_file:
        indices = json.load(indices_file)

    subset_dataset = [dataset[i]["drawing"] for i in indices[subset_name]]
    return subset_dataset

def convert_drawing_to_5d_format(raw_drawing: list) -> np.ndarray:
    """
    Converts a list of raw drawings into a 5D format suitable for further processing.
    Each drawing is represented as a list of strokes, where each stroke contains two lists:
    one for x-coordinates and one for y-coordinates. The function processes each point in
    every stroke and computes the following for each point:
        - delta_x: Change in x-coordinate from the previous point.
        - delta_y: Change in y-coordinate from the previous point.
        - state: A one-hot encoded list indicating the point's status:
            [1, 0, 0]: Regular point within a stroke.
            [0, 1, 0]: End of a stroke.
            [0, 0, 1]: End of the drawing.
    Args:
        raw_drawing (list): A list of drawings, where each drawing is a list of strokes,
            and each stroke is a list containing two lists: x-coordinates and y-coordinates.
    Returns:
        list of np.ndarray: A list where each element is a NumPy array of shape (N, 5),
            representing the converted drawing in 5D format (delta_x, delta_y, state).
    """

    converted_raw_drawing = []
    for drawing_idx in range(len(raw_drawing)):
        drawing = raw_drawing[drawing_idx]

        last_x, last_y = 0, 0
        converted_drawing = []
        num_strokes = len(drawing)
        for stroke_idx, stroke in enumerate(drawing):

            x_coords = stroke[0]
            y_coords = stroke[1]
            num_points_in_stroke = len(stroke[0])
            for point_idx in range(len(x_coords)):
                x, y = x_coords[point_idx], y_coords[point_idx]

                delta_x = x - last_x
                delta_y = y - last_y
                is_end_of_stroke = (point_idx == num_points_in_stroke - 1)
                is_end_of_drawing = is_end_of_stroke and (stroke_idx == num_strokes - 1)

                state = [0, 0, 0]
                if is_end_of_drawing:
                    state = [0, 0, 1]
                elif is_end_of_stroke:
                    state = [0, 1, 0]
                else:
                    state = [1, 0, 0]

                converted_drawing.append([delta_x, delta_y, *state])
                last_x, last_y = x, y

        converted_raw_drawing.append(np.array(converted_drawing, dtype=np.float32))
    return converted_raw_drawing

def split_drawing_into_strokes(drawing_5d: np.ndarray) -> list:
    end_indices = np.where((drawing_5d[:, 3] == 1) | (drawing_5d[:, 4] == 1))[0]
    
    if len(end_indices) == 0:
        return [drawing_5d]

    strokes = []
    start_idx = 0
    for end_idx in end_indices:
        strokes.append(drawing_5d[start_idx : end_idx + 1])
        start_idx = end_idx + 1
        
    return strokes

def calculate_max_single_stroke(all_drawings_5d: list) -> int:
    max_len = 0
    
    for drawing_5d in all_drawings_5d:
        strokes = split_drawing_into_strokes(drawing_5d)
        
        for stroke in strokes:
            if len(stroke) > max_len:
                max_len = len(stroke)
                
    return max_len

def get_random_history_target_pair(drawing_5d: np.ndarray, ) -> tuple:
    strokes = split_drawing_into_strokes(drawing_5d)

    if len(strokes) == 0:
        return None, drawing_5d
    
    target_stroke_index = np.random.randint(0, len(strokes))
    target_stroke = strokes[target_stroke_index]
    
    if target_stroke_index == 0:
        history = None
    else:
        previous_strokes = strokes[:target_stroke_index]
        history = np.concatenate(previous_strokes, axis=0)
        
    return history, target_stroke

def convert_5d_to_raw_format(processed_drawing: np.ndarray) -> list:
    raw_drawing = []
    current_stroke_x = []
    current_stroke_y = []
    current_x, current_y = 0, 0
    for row in processed_drawing:
        delta_x, delta_y = row[0], row[1]
        
        current_x += delta_x
        current_y += delta_y
        current_stroke_x.append(int(current_x))
        current_stroke_y.append(int(current_y))

        is_end_of_stroke = (int(row[3]) == 1)
        is_end_of_drawing = (int(row[4]) == 1)

        if is_end_of_stroke or is_end_of_drawing:
            raw_drawing.append([current_stroke_x, current_stroke_y])
            
            current_stroke_x = []
            current_stroke_y = []            
    return raw_drawing
            
def calculate_normalization_params(raw_drawing: list) -> tuple:
    """
    Calculates the mean and standard deviation of all delta values in the provided raw drawing data.
    Args:
        raw_drawing (list): A list of drawings, where each drawing is a list of strokes,
            and each stroke is expected to be a sequence containing delta values at index 0 and 1.
    Returns:
        tuple: A tuple containing the mean and standard deviation (mean, std) of all delta values.
    """

    all_deltas = []
    for drawing in raw_drawing:
        deltas = drawing[:, :2].flatten()
        all_deltas.append(deltas)           

    all_deltas = np.concatenate(all_deltas)
    mean = np.mean(all_deltas)
    std = np.std(all_deltas)
    return mean, std

def normalize(drawing: np.ndarray, std: float = 1.0) -> np.ndarray:
    """
    Normalizes the first two columns of a drawing array by dividing them by the given standard deviation.
    Args:
        drawing (np.ndarray): The input array representing the drawing, where the first two columns are coordinates.
        std (float, optional): The standard deviation value to normalize by. Defaults to 1.0.
    Returns:
        np.ndarray: The normalized drawing array.
    """
    
    drawing[:, :2] /= std
    return drawing

def denormalize(drawing: np.ndarray, std: float = 1.0) -> np.ndarray:
    """
    Denormalizes the coordinates of a drawing by multiplying the first two columns by the given standard deviation.
    Args:
        drawing (np.ndarray): The input array representing the drawing, where the first two columns are coordinates.
        std (float, optional): The standard deviation value to multiply the coordinates by. Defaults to 1.0.
    Returns:
        np.ndarray: The denormalized drawing array.
    """
    
    drawing[:, :2] *= std
    return drawing

def generate_gif(drawing, output_path: str, fps: int = 25, output_name: str = "drawing.gif") -> None:
    """
    Generates a GIF from a drawing represented as a list of strokes.
    Args:
        drawing (list): A list of strokes, where each stroke is a list of two lists:
            [x_coords, y_coords].
        output_path (str): The path to the output directory where the GIF will be saved.
        fps (int, optional): Frames per second for the GIF. Defaults to 25.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.invert_yaxis()
    ax.axis('off')

    tmp_folder_path = os.path.join(output_path, "tmp")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tmp_folder_path, exist_ok=True)

    frame_counter = 0
    for stroke in drawing:
        x_coord = stroke[0]
        y_coord = stroke[1]

        total_frames = len(x_coord)
        for frame_idx in range(total_frames):
            ax.plot(x_coord[:frame_idx+1], y_coord[:frame_idx+1], color='black', linewidth=2)

            tmp_frame_path = os.path.join(tmp_folder_path, f"frame_{str(frame_counter).zfill(5)}.png")
            plt.savefig(tmp_frame_path, dpi=96)
            frame_counter += 1

    plt.close(fig)
    frame_list = [imageio.imread(os.path.join(tmp_folder_path, f)) for f in sorted(os.listdir(tmp_folder_path)) if f.endswith(".png")]
    imageio.mimsave(os.path.join(output_path, output_name), frame_list, fps=fps)
    shutil.rmtree(tmp_folder_path, ignore_errors=True)
    return os.path.join(output_path, output_name)