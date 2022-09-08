import os
import moviepy.video.io.ImageSequenceClip

import glob
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def nkeys(text):
    return [atoi(c) for c in re.split(r"(\d+)",text)]

inDir="./siteDependent3/coef0.8/0s12impurityStrength0Gmax6tTot500k00.8a10.01coefPi0.8/"
files=glob.glob(inDir+"out/*.png")
files.sort(key=nkeys)

fps=100
clip=moviepy.video.io.ImageSequenceClip.ImageSequenceClip(files,fps=fps)
clip.write_videofile(inDir+"coef"+str(0.8)+"a1"+str(0.01)+".mp4")

