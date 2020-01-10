import tensorflow as tf
import pandas as pd
import numpy as np




def get_test_file(filename):
	image_array = []
	with open(filename, 'r') as f:
		lines = f.readlines()
	for line in lines:
		line_arr = line.split( );
		image_array.append(line_arr[0])
	return image_array



def prepare_input(test_image_file):
	content = tf.io.read_file(test_image_file)
	test_pic = tf.image.decode_image(content,channels=3, dtype=tf.float32)
	test_pic = tf.expand_dims(test_pic, 0)
	test_pic = np.array(test_pic)
	return test_pic



def representative_dataset_gen():
	num_calibration_steps = 999
	all_lines = get_test_file("val_1000.txt")
	#print(all_lines)
	counter = 0
	for i in range(num_calibration_steps):
		# Get sample input data as a numpy array in a method of your choosing.
		line = all_lines[i]
		print(line)
		pic_data = prepare_input(line)
		if pic_data is not None:
			counter += 1
			print(counter)
			yield [pic_data]
		else:
			continue


print (tf.__version__)
tf.enable_eager_execution()
graph_def_file = "try1_opt.pb"
output_node_names = ['y3/Reshape']

input_node_name = ['Reshape']
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_node_name, output_node_names, input_shapes={"Reshape":[1, 416,416,3]})

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

open("model_mv2_q_rep_dataset.tflite", "wb").write(tflite_quant_model)


