import argparse
from pathlib import Path

import dlib
import scipy
import numpy as np
from PIL import Image
from tqdm import tqdm


DLIB_68_LANDMARK_PREDICTOR = './pretrained/shape_predictor_68_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(DLIB_68_LANDMARK_PREDICTOR)
face_detector = dlib.get_frontal_face_detector()


def get_landmarks(im, face_detector, shape_predictor):
    rects = face_detector(im, 1)
    shape = shape_predictor(im, rects[0])
    # parse result to numpy array
    coords = np.zeros((68, 2), dtype='int')
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def align_face_image(
    input_path, output_path, output_size=512,
    enable_padding=True, rotate_level=True, random_shift=0.0, retry_crops=False,
):
    '''Align and crop image like FFHQ dataset

    References:
      https://github.com/NVlabs/ffhq-dataset/blob/4826aa6ea77aa7f1a7802b938ed7c40afb985cda/download_ffhq.py#L259
      https://github.com/fengdu78/machine_learning_beginner/blob/master/deep-learning-with-tensorflow-keras-pytorch/deep-learning-with-keras-notebooks-master/7.5-face-landmarks-detection.ipynb
    '''
    np.random.seed(12345)
    img = Image.open(input_path)
    np_img = np.array(img)
    lm = get_landmarks(np_img, face_detector, shape_predictor)

    # Parse landmarks.
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    if rotate_level:
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c0 = eye_avg + eye_to_mouth * 0.1
    else:
        x = np.array([1, 0], dtype=np.float64)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c0 = eye_avg + eye_to_mouth * 0.1

    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2

    # Keep drawing new random crop offsets until we find one that is contained in the image
    # and does not require padding
    if random_shift != 0:
        for _ in range(1000):
            # Offset the crop rectange center by a random shift proportional to image dimension
            # and the requested standard deviation
            c = (c0 + np.hypot(*x)*2 * random_shift * np.random.normal(0, 1, c0.shape))
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            if not retry_crops or not (crop[0] < 0 or crop[1] < 0 or crop[2] >= img.width or crop[3] >= img.height):
                # We're happy with this crop (either it fits within the image, or retries are disabled)
                break
        else:
            # rejected N times, give up and move to next image
            # (does not happen in practice with the FFHQ data)
            print('rejected image')
            return

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    transform_size = 4 * output_size
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.LANCZOS)

    img.save(output_path)


def dir_align_face_image(input_dir, output_dir, extentions=['png', 'jpg', 'jpeg'], **kwargs):
    input_path_lst = []
    for ext in extentions:
        input_path_lst += Path(input_dir).glob(f'*.{ext}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in tqdm(input_path_lst):
        output_path = output_dir / input_path.name
        align_face_image(input_path, output_path, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input image path or directory')
    parser.add_argument('-o', '--output', help='output image path or directory')
    parser.add_argument('--output_size', type=int, default=512, help='output image size')
    args = parser.parse_args()

    if Path(args.input).is_file():
        align_face_image(args.input, args.output, output_size=args.output_size)
    elif Path(args.input).is_dir():
        dir_align_face_image(args.input, args.output, output_size=args.output_size)
    else:
        raise FileNotFoundError(f'Input {args.input} not found.')
