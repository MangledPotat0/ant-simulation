#!/bin/bash

wd=/app/antproject 
#/mnt/c/Users/user/Desktop/Coding/ant
dd=$(date '+%Y%m%d')
fname=$1

rawdir=${wd}/data/rawvideos
vidd=${wd}/data/videos
pydir=${wd}/codebase/tracking
trajdir=${wd}/data/trajectories
montdir=${wd}/data/montages
sleapdir=${wd}/data/sleap
cd $wd

# convert raw h264 video into mp4
ffmpeg -i ${rawdir}/${fname}.h264 ${vidd}/${fname}.mp4 -y

# Crop video to preset dimensions
# python ${pydir}/cropper.py -v ${vidd}/${fname} -c 100 100 100 100

# Modify fname to indicate it's beed cropped
fname=${fname}cropped

# Run SLEAP detection and initial tracking
sleap-track ${vidd}/${fname}.mp4 \
	--video.input_format channels_last \
	-frames 0-599
	-m ${sleapdir}/current/centered/training_config.json \
	-m ${sleapdir}/current/centroid/training_config.json \
	--tracking.tracker simple \
	--verbosity json \
	--no-empty-frames \
	-o ${sleapdir}/${fname}/${fname}.slp

# Reformat SLEAP output

#python ${pydir}/sleapconverter.py -f ${fname}.slp

# Fix tracks
# python tracking/sleap_trajectory_extraction.py -f [file]
echo 'Write the thing in python'

# Create montage

(cd $pydir && python montagecopy.py -f $fname)

# Time formatter to measure runtime
format_time() {
	((h=${1}/3600))
	((m=(${1}%3600)/60))
	((s=${1}%60))
	printf "%02d:%02d:%02d\n" $h $m $s
}

echo "Analysis pipeline total runtime: $(format_time $SECONDS)"
