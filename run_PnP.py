import json
import glob
import numpy as np
import imageio

from renderer import Renderer
from est_Pw import est_Pw
from PnP import PnP

def main():

	# intrinsic (Given. Do not change)
	K = np.array([[823.8, 0.0, 304.8], 
	              [0.0, 822.8, 236.3], 
	              [0.0, 0.0, 1.0]])
	focal  = [K[0,0], K[1,1]]
	center = [K[0,2], K[1,2]]
	

	# extrinsic
	print('Loading data ...')
	tag_size = 0.14 # meter
	Pw = est_Pw(tag_size)
	corners = np.load('corners.npy')

	# load and process mesh
	with open('mesh.json', 'r') as in_file:
		mesh = json.load(in_file)
	vertices = np.array(mesh['vertices'])
	faces = np.array(mesh['faces'])
	renderer = Renderer(focal, center, img_w=640, img_h=480, 
						faces=faces, color=(0.3, 0.8, 0.3, 1.0))

	# Process each frame
	print('Processing each frame ...')
	frames = glob.glob('frames/*.jpg')
	frames.sort()

	final = np.zeros([len(frames), 480, 640, 3], dtype=np.uint8)
	for i, f in enumerate(frames):
		img = np.array(imageio.imread(f))
		# PnP
		Pc = corners[i]
		R, t = PnP(Pc, Pw, K)

		# Render
		t = -R.T.dot(t)
		R = R.T
		points = vertices.dot(R.T) + t.squeeze()
		output, _ = renderer(points, np.eye(3), [0,0,0], img)
		final[i] = output

	# Save frames as GIF
	print('Saving GIF ...')
	with imageio.get_writer('bird_collineation.gif', mode='I') as writer:
	    for i in range(len(final)):
	        img = final[i]
	        writer.append_data(img)

	print('Complete.')
	
	return


if __name__ == '__main__':

	main()

