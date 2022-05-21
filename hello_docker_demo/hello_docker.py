from joblib import dump, load
import numpy as np

def hello_docker():
	print('Hello Docker! I created my first image!')

	with open('/usr/src/hello_docker/hello_file.txt', 'w') as f:
		f.write('Hello Docker, now in a file!')
		print('hello_file.txt successfully created')

	example_array = np.zeros([10, 10])
	dump(example_array, '/usr/src/hello_docker/example_array')
	print('example_array file successfully created')

if __name__ == '__main__':
	hello_docker()