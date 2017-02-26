# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from line_detector import *

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    result = lane_detector(image)
    return result

if __name__ == "__main__":
    challenge_output = 'extra.mp4'
    clip2 = VideoFileClip('challenge.mp4')
    challenge_clip = clip2.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)