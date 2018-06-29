import time
import random
import math

import OSC

c = OSC.OSCClient()
c.connect(('127.0.0.1', 9001))   # localhost, port 57120


import sys; from PIL import Image; import numpy as np; import curses; import time;
from moviepy.editor import VideoFileClip;

display = True
# chars = np.asarray(list(' .,:;irsXA253hMHGS#9B&@'))
chars = np.asarray(list('  .,-=+*&@'))

def renderImageASCII(stdscr, img):
		if display:
			height,width = stdscr.getmaxyx()
			stdscr.clear()

		img = img.resize((int(img.size[0]*(2.)), img.size[1]))
		if display:
			img.thumbnail((width, height))

		img = np.asarray( img )
	
		#img = img.reshape([img.shape[0], img.shape[1]])

		img = np.sum(img, axis=2)
		img -= img.min()
		img = np.round( (img/float(img.max()))*(chars.size-1) )

		try:
			lines = ["".join(r) for r in chars[img.astype(int)]]
		except:
			return

		if display:
			try:
				for i in range(len(lines)):
					stdscr.addstr((height - img.shape[0])/2 + i, (width - img.shape[1])/2, lines[i])
			except:
				pass

			stdscr.refresh()

def displayImage(f, duration=0, start=0):	
	if display:
		stdscr = curses.initscr()
		curses.cbreak()
		curses.noecho()
		curses.curs_set(0)
		stdscr.keypad(1)
		pass

	try:
		if f.endswith('gif') or f.endswith('mp4'):
			clip = VideoFileClip(f)

			duration = duration or clip.duration
			clip = clip.subclip(start, min(start + duration, clip.duration - start))

			for frame in clip.iter_frames():
				img = Image.fromarray(frame)
				renderImageASCII(stdscr, img)
				
				time.sleep(1.0 / clip.fps)
		elif f.endswith('png') or f.endswith('jpg') or f.endswith('jpeg'):
			img = Image.open(f)

			renderImageASCII(stdscr, img)
			time.sleep(duration or 3)

	finally:
		if display:
			curses.nocbreak()
			stdscr.keypad(0)
			curses.echo()
			curses.endwin()

timeDelays = [0.169, 0.265, 0.256, 0.257, 0.272, 0.255, 0.263, 0.251, 0.599, 0.169, 0.262, 0.255, 0.261, 0.265, 0.26, 0.254, 0.252, 0.269, 0.583, 0.156, 0.247, 0.26, 0.261, 0.262, 0.251, 0.262, 0.261, 0.262, 0.577, 0.155, 0.263, 0.252, 0.262, 0.254, 0.251, 0.257, 0.263, 0.258, 0.568, 0.159, 0.266, 0.265, 0.264, 0.268, 0.263, 0.256, 0.252, 0.256, 0.571, 0.15, 0.25, 0.272, 0.266, 0.254, 0.267, 0.258, 0.255, 0.255, 0.563, 0.151, 0.264, 0.261, 0.265, 0.267, 0.263, 0.261, 0.259, 0.266, 0.562, 0.166, 0.271, 0.251, 0.259, 0.259, 0.251, 0.277, 0.17, 0.239, 0.254, 0.259, 0.249, 0.247, 0.251, 0.261, 0.551, 0.16, 0.246, 0.26, 0.253, 0.261, 0.261, 0.254, 0.29, 0.253, 0.578, 0.166, 0.253, 0.256, 0.257, 0.264, 0.266, 0.253, 0.248, 0.26, 0.588, 0.165, 0.258, 0.26, 0.261, 0.249, 0.254, 0.247, 0.254, 0.25, 0.564, 0.156, 0.257, 0.247, 0.265, 0.249, 0.249, 0.254, 0.263, 0.256, 0.555, 0.167, 0.261, 0.255, 0.249, 0.256, 0.252, 0.259, 0.266, 0.265, 0.545, 0.157, 0.252, 0.245, 0.258, 0.247, 0.253, 0.257, 0.272, 0.248, 0.566, 0.158, 0.262, 0.25, 0.262, 0.255, 0.247, 0.258, 0.157, 0.249, 0.268, 0.249, 0.249, 0.254, 0.252, 0.264, 0.561, 0.155, 0.246, 0.249, 0.252, 0.265, 0.263, 0.249, 0.269, 0.258, 0.544, 0.157, 0.256, 0.269, 0.266, 0.257, 0.25, 0.263, 0.249, 0.253, 0.546, 0.162, 0.262, 0.262, 0.267, 0.265, 0.255, 0.267, 0.254, 0.264, 0.553, 0.164, 0.262, 0.249, 0.263, 0.248, 0.263, 0.261, 0.267, 0.247, 0.557, 0.16, 0.25, 0.275, 0.253, 0.269, 0.248, 0.262, 0.252, 0.249, 0.546, 0.165, 0.252, 0.263, 0.271, 0.263, 0.265, 0.274, 0.26, 0.252, 0.554, 0.161, 0.247, 0.253, 0.269, 0.265, 0.261]

time.sleep(0.3)


datasetSize = random.randint(1000, 2000)

nThreads		= 8
batchSize		= 64
niter			= random.randint(20, 99)

epoch			= 1
totalBatches	= int(math.ceil(datasetSize / float(batchSize)))
currentBatch	= 1

print("""
{
	ntrain : inf
	nThreads : %d
	niter : %d
	batchSize : %d
	ndf : 32
	fineSize : 128
	nz : 100
	loadSize : 129
	gpu : 1
	ngf : 64
	dataset : "folder"
	lr : 0.0002
	noise : "normal"
	name : "dcgan-experiment"
	beta1 : 0.5
	display_id : 10
	display : 1
}
""" % (nThreads, niter, batchSize))

time.sleep(0.1)

randomSeed = random.randint(1000, 9800)

print( "Random Seed: {:d}".format(randomSeed) )

for i in range(nThreads):
	time.sleep(random.uniform(0.01, 0.05))

	print( "Starting donkey with id: {:d} seed: {:d}".format(i+1, randomSeed+i+1) )
	print( "table: {:s}".format(hex(int(random.uniform(0x40000000, 0x42000000)))) )

for i in range(nThreads):
	time.sleep(random.uniform(0.01, 0.02))

	print( "Loading train metadata from cache" )

time.sleep(0.4)

print( "Dataset: folder  Size:  {:d}".format(datasetSize) )

time.sleep(0.3)

print( "Found Environment variable CUDNN_PATH = /home/machine/cuda/lib64/libcudnn.so.5" )

time.sleep(1.7)

totalEpochTime = 0.0

i = 0

while True:
	i += 1
	i = i % len(timeDelays)

	# timeDelay	= random.uniform(0.240, 0.310)
	timeDelay	= timeDelays[i]
	dataTime 	= random.uniform(0.010, 0.020)
	err_G		= random.uniform(0.4, 1.6)
	err_D		= random.uniform(0.4, 1.6)

	try:
		oscmsg = OSC.OSCMessage()
		oscmsg.setAddress("/beat")
		c.send(oscmsg)
	except:
		pass


	totalEpochTime += timeDelay

	time.sleep(timeDelay)

	print( "Epoch: [{:d}][{:8d} /{:8d} ] Time: {:.3f}  DataTime: {:.3f}    Err_G: {:.4f}  Err_D: {:.4f}".format(epoch, currentBatch, totalBatches , timeDelay, dataTime, err_G, err_D) )

	currentBatch += 1

	if currentBatch > totalBatches:
		print( "End of epoch {:d} / {:d}   Time Taken: {:.4f}".format(epoch, niter, totalEpochTime) )

		currentBatch 	 = 1
		epoch			+= 1
		totalEpochTime	 = 0.0

		# displayImage("path/to/image_video_or_gif", duration , start)

		if epoch >= niter:
			break