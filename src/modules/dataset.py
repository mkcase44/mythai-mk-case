import numpy as np
from torch.utils.data import Dataset

from src.utils.dataset_utils import normalize, get_random_history_target_pair, calculate_max_single_stroke


class QuickDrawDataset(Dataset):
    def __init__(self,
                 drawing_5d: list,
                 max_seq_length: int,
                 std: float = 1.0,
                 random_scale_factor: float = 0.0,
                 augment_stroke_prob: float = 0.0,
                 limit: int = 1000):
        super().__init__()

        self.std = std
        self.limit = limit
        self.max_seq_length = max_seq_length + 1
        self.random_scale_factor = random_scale_factor
        self.augment_stroke_prob = augment_stroke_prob

        self.sos_token = np.array([0, 0, 1, 0, 0], dtype=np.float32)
        self.__preprocess_drawing(drawing_5d)

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        strokes = self.strokes[idx]

        augmented_strokes = self.random_scale(strokes)
        augmented_strokes_copy = np.copy(augmented_strokes)

        if self.augment_stroke_prob > 0:
            augmented_strokes_copy = self.augment_strokes_5d(augmented_strokes_copy)

        padded_stroke = np.zeros((self.max_seq_length, 5), dtype=augmented_strokes_copy.dtype)

        padded_stroke[1:augmented_strokes_copy.shape[0]+1] = augmented_strokes_copy
        padded_stroke[0] = self.sos_token

        return {
            "stroke": padded_stroke,
        }

    def __preprocess_drawing(self, drawing_5d):
        """
        Preprocesses a batch of 5D drawing data by normalizing and filtering sequences.
        Args:
            drawing_5d (list or np.ndarray): A list or array of drawing sequences, 
                where each sequence is expected to be a numpy array.
        Returns:
            None: The processed and sorted drawing sequences are stored in self.strokes.
        Notes:
            - Only drawings with a sequence length less than self.max_seq_length are processed.
            - Each drawing is clipped to the range [-self.limit, self.limit] and normalized using self.std.
            - The processed drawings are sorted by sequence length in ascending order and stored in self.strokes.
        """

        seq_len = []
        raw_data = []
        count_data = 0

        for drawing_idx in range(len(drawing_5d)):
            drawing = drawing_5d[drawing_idx]
            if drawing.shape[0] < self.max_seq_length:
                count_data += 1

                _drawing = np.minimum(drawing, self.limit)
                _drawing = np.maximum(_drawing, -self.limit)
                _drawing = normalize(_drawing, self.std)

                raw_data.append(_drawing)
                seq_len.append(_drawing.shape[0])

        seq_len = np.array(seq_len)
        idx = np.argsort(seq_len)

        self.strokes = []
        for i in range(len(seq_len)):
            self.strokes.append(raw_data[idx[i]])

    def random_scale(self, data):
        """
        Applies a random scaling transformation to the input 
        
        Args:
            data (np.ndarray): Input array of shape (N, 2), where N is the number of data points.
                               The first column represents x-coordinates, and the second column represents y-coordinates.
        Returns:
            np.ndarray: Scaled data array of the same shape as the input.
        """

        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result
        
    def augment_strokes_5d(self, strokes):
        """
        Augments a sequence of 5D stroke data by probabilistically merging consecutive points within a stroke.
        
        Args:
            strokes (np.ndarray or list): Sequence of stroke points, where each point is a 5D vector.
        Returns:
            np.ndarray: Augmented sequence of stroke points as a NumPy array of type float32.
        """
        
        if self.augment_stroke_prob <= 0:
            return np.array(strokes, dtype=np.float32)
        
        count = 0
        result = []
        for i in range(len(strokes)):
            current_point = strokes[i]
            is_stroke_ending = (current_point[3] == 1 or current_point[4] == 1)

            if is_stroke_ending:
                count = 0
            else:
                count += 1

            if (len(result) > 0 and current_point[2] == 1 and \
                    result[-1][2] == 1 and count > 2 and np.random.rand() < self.augment_stroke_prob):
                result[-1][0] += current_point[0]
                result[-1][1] += current_point[1]

            else:
                result.append(list(current_point))          
        return np.array(result, dtype=np.float32)
    
class StrokeHistoryQuickDrawDataset(Dataset):
    def __init__(self,
                 drawing_5d: list,
                 max_seq_length: int,
                 std: float = 1.0,
                 random_scale_factor: float = 0.0,
                 augment_stroke_prob: float = 0.0,
                 limit=1000):
        super().__init__()

        self.std = std
        self.limit = limit
    
        self.max_seq_length = max_seq_length + 1
        self.random_scale_factor = random_scale_factor
        self.augment_stroke_prob = augment_stroke_prob

        self.sos_token = np.array([0, 0, 1, 0, 0], dtype=np.float32)
        self.__preprocess_drawing(drawing_5d)

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        stroke = self.strokes[idx]

        augmented_stroke = self.random_scale(stroke)
        augmented_stroke_copy = np.copy(augmented_stroke)

        if self.augment_stroke_prob > 0:
            augmented_stroke_copy = self.augment_strokes_5d(augmented_stroke_copy)

        stroke_history, stroke = get_random_history_target_pair(augmented_stroke_copy)
        if stroke_history is None:
            stroke_history = self.sos_token.reshape(1, 5)
        else:
            stroke_history = np.vstack((self.sos_token, stroke_history))

        padded_stroke = np.zeros((self.max_stroke_length, 5), dtype=stroke.dtype)
        padded_stroke[:stroke.shape[0]] = stroke

        padded_stroke_history = np.zeros((self.max_seq_length, 5), dtype=stroke_history.dtype)
        padded_stroke_history[:stroke_history.shape[0]] = stroke_history
        stroke_history_mask = np.zeros((self.max_seq_length,), dtype=np.float32)
        stroke_history_mask[:stroke_history.shape[0]] = 1.
    
        return {
            "stroke": padded_stroke,
            "stroke_history": padded_stroke_history,
            "stroke_history_mask": stroke_history_mask
        }

    def __preprocess_drawing(self, drawing_5d):
        """
        Preprocesses a batch of 5D drawing data by normalizing and filtering sequences.
        Args:
            drawing_5d (list or np.ndarray): A list or array of drawing sequences, 
                where each sequence is expected to be a numpy array.
        Returns:
            None: The processed and sorted drawing sequences are stored in self.strokes.
        Notes:
            - Only drawings with a sequence length less than self.max_seq_length are processed.
            - Each drawing is clipped to the range [-self.limit, self.limit] and normalized using self.std.
            - The processed drawings are sorted by sequence length in ascending order and stored in self.strokes.
        """
        seq_len = []
        raw_data = []
        count_data = 0

        for drawing_idx in range(len(drawing_5d)):
            drawing = drawing_5d[drawing_idx]
            if drawing.shape[0] < self.max_seq_length:
                count_data += 1

                _drawing = np.minimum(drawing, self.limit)
                _drawing = np.maximum(_drawing, -self.limit)
                _drawing = normalize(_drawing, self.std)

                raw_data.append(_drawing)
                seq_len.append(_drawing.shape[0])

        seq_len = np.array(seq_len)
        idx = np.argsort(seq_len)

        self.strokes = []
        for i in range(len(seq_len)):
            self.strokes.append(raw_data[idx[i]])

        self.max_stroke_length = calculate_max_single_stroke(self.strokes)

    def random_scale(self, data):
        """
        Applies a random scaling transformation to the input 
        
        Args:
            data (np.ndarray): Input array of shape (N, 2), where N is the number of data points.
                               The first column represents x-coordinates, and the second column represents y-coordinates.
        Returns:
            np.ndarray: Scaled data array of the same shape as the input.
        """
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result
        
    def augment_strokes_5d(self, strokes):
        """
        Augments a sequence of 5D stroke data by probabilistically merging consecutive points within a stroke.
        
        Args:
            strokes (np.ndarray or list): Sequence of stroke points, where each point is a 5D vector.
        Returns:
            np.ndarray: Augmented sequence of stroke points as a NumPy array of type float32.
        """

        if self.augment_stroke_prob <= 0:
            return np.array(strokes, dtype=np.float32)
        
        count = 0
        result = []
        for i in range(len(strokes)):
            current_point = strokes[i]
            is_stroke_ending = (current_point[3] == 1 or current_point[4] == 1)

            if is_stroke_ending:
                count = 0
            else:
                count += 1

            if (len(result) > 0 and current_point[2] == 1 and \
                    result[-1][2] == 1 and count > 2 and np.random.rand() < self.augment_stroke_prob):
                result[-1][0] += current_point[0]
                result[-1][1] += current_point[1]

            else:
                result.append(list(current_point))          
        return np.array(result, dtype=np.float32)
