#!/bin/bash

wd=/app/antproject 
#/mnt/c/Users/user/Desktop/Coding/ant
dd=$(date '+%Y%m%d')
fname=$1

vidd=${wd}/data/videos
pydir=${wd}/codebase/python
trajdir=${wd}/data/trajectories
montdir=${wd}/data/montages
sleapdir=${wd}/data/sleap
cd $wd

# Run SLEAP detection and initial tracking
sleap-track $vidd${fname}.mp4 \
	--video.input_format channels_last \
	-m ${sleapdir}/current/centered/training_config.json \
	-m ${sleapdir}/current/centroid/training_config.json \
	--tracking.tracker simple \
	--verbosity json \
	--no-empty-frames \
	-o ${fname}.slp

# Reformat SLEAP output

#python sleapconverter.py -f $(fname)_ready.slp

# Fix tracks

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
