import pickle
import json
import os

for name in os.listdir('new_data'):
	print(name)
	path = os.path.join('new_data', name)
	data = pickle.load(open(path, 'rb'))
	for pageid in data['pages']:
		for key in data['pages'][pageid]['information']:
			data['pages'][pageid]['information'][key] = \
				[data['pages'][pageid]['information'][key]]
	target_path = os.path.join('new_new_data', name)
	pickle.dump(data, open(target_path, 'wb'))