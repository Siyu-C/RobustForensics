import os
import subprocess
import sys

filedir = "DFDC-Kaggle"
outdir = "DFDC-Kaggle_image"


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass

    
def main(video_file, video_id):
    videopath = os.path.join(filedir, video_file)
    mkdir(os.path.join(outdir, video_id))
    outpath = os.path.join(outdir, video_id, 'frame%d.png')
    ffmpeg_command = ['ffmpeg',  '-i',  videopath,  outpath, '-loglevel', 'panic']
    subprocess.call(ffmpeg_command)
    print('Frames at %s extracted to %s'%(videopath, outpath))
    sys.stdout.flush()


if __name__ == '__main__':

    files = []
    ids = []
    clses = os.listdir(filedir)
    for cls in clses:
        vids = os.listdir(os.path.join(filedir, cls))
        for vid in vids:
            if not vid.endswith('.mp4'):
                continue
            files.append(os.path.join(cls, vid))
            ids.append(os.path.join(cls, vid[:-4]))

    print(len(ids))
            
    for i in range(len(ids)):
        main(files[i], ids[i])
