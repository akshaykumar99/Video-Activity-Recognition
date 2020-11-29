from PIL import Image
import numpy as np
from tqdm import tqdm
from skvideo.utils import rgb2gray
from skvideo.io import FFmpegReader, ffprobe
from keras.preprocessing import image

class Videos_Processing(object):

    def __init__(self, to_gray = True, target_size = None, max_frames = None, extract_frames = 'middle',
                        required_fps = None, normalize_pixels = None):
        """
            to_gray (boolean):  If True, then each frame will be converted to gray scale. Otherwise, not.
            target_size (tuple): (New_Width, New_Height)
            max_frames (int): The maximum number of frames to return for each video.
            extract_frames (str): {'first', 'middle', 'last'}
                'first': Extract the first 'N' frames
                'last': Extract the last 'N' frames
                'middle': Extract 'N' frames from the middle
            required_fps (int): Capture 'N' frame(s) per second from the video.
                Only the first 'N' frame(s) for each second in the video are captured.
            normalize_pixels (tuple/str): If 'None', the pixels will not be normalized.
                If a tuple - (New_min, New_max) is passed, Min-max Normalization will be used.
                If the value is 'z-score', then Z-score Normalization will be used.
                For each pixel p, z_score = (p - mean) / std
        """
        self.fps = None
        self.to_gray = to_gray
        self.target_size = target_size
        self.max_frames = max_frames
        self.extract_frames = extract_frames
        self.required_fps = required_fps
        self.normalize_pixels = normalize_pixels

    def process_video(self, video):
        # Returns: Numpy.ndarray, tensor (processed video) with shape (<max_frames>, <height>, <width>, <channels>)
        total_frames = video.shape[0]
        if self.max_frames <= total_frames:
            if self.extract_frames == 'first':
                video = video[:self.max_frames]
            elif self.extract_frames == 'last':
                video = video[(total_frames - self.max_frames):]
            elif self.extract_frames == 'middle':
                front = ((total_frames - self.max_frames) // 2) + 1
                video = video[front:(front + self.max_frames)]
        else:
            raise IndexError('max_frames is greater than the total number of frames in the video')
        return video

    def read_video(self, path):
        # Return: Numpy.ndarray 5-d tensor with shape (1, <No. of frames>, <height>, <width>, <channels>)
        capt = FFmpegReader(filename = path)
        self.fps = int(capt.inputfps)
        list_of_frames = []

        for index, frame in enumerate(capt.nextFrame()):
            # frame -> (<height>, <width>, 3)
            capture_frame = True
            if self.required_fps != None:
                is_valid = range(self.required_fps)
                capture_frame = (index % self.fps) in is_valid

            if capture_frame:
                if self.target_size is not None:
                    temp_image = image.array_to_img(frame)
                    frame = image.img_to_array(temp_image.resize(self.target_size,Image.ANTIALIAS)).astype('uint8')
                list_of_frames.append(frame)
        temp_video = np.stack(list_of_frames)
        capt.close()
        if self.to_gray:
            temp_video = rgb2gray(temp_video)
        if self.max_frames is not None:
            temp_video = self.process_video(video = temp_video)
        return np.expand_dims(temp_video, axis = 0)
        
    def read_videos(self, paths):
        # Return: Numpy.ndarray A 5-d tensor with shape (<No. of Videos>, <No. of frames>, <height>, <width>, <channels>)
        videos_list = [self.read_video(path) for path in tqdm(paths)]
        tensor = np.vstack(videos_list)
        if self.normalize_pixels != None:
            if (type(self.normalize_pixels) == tuple):
                new_min = self.normalize_pixels[0]
                diff = self.normalize_pixels[1] - new_min
                min_ = np.min(tensor, axis = (1, 2, 3), keepdims = True)
                max_ = np.max(tensor, axis = (1, 2, 3), keepdims = True)
                return ((tensor.astype('float32') - min_) / (max_ - min_)) * diff + new_min
            elif self.normalize_pixels == 'z-score':
                mean = np.mean(tensor, axis=(1, 2, 3), keepdims=True)
                std = np.std(tensor, axis=(1, 2, 3), keepdims=True)
                return (tensor.astype('float32') - mean) / std
        return tensor