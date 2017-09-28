import os
import numpy as np

num_points = 1000
vectors_set = []
for i in xrange(num_points):
         x1= np.random.normal(0.0, 0.55)
         y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

import matplotlib.pyplot as plt

#Graphic display
plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.show()

import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(8):
     sess.run(train)
     print(step, sess.run(W), sess.run(b))
     print(step, sess.run(loss))

     #Graphic display
     plt.plot(x_data, y_data, 'ro')
     plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
     plt.xlabel('x')
     plt.xlim(-2,2)
     plt.ylim(0.1,0.6)
     plt.ylabel('y')
     plt.legend()
     plt.show()
     
# Save
out_dir = "./tf_model"
if os.path.isdir(out_dir) is False:
     os.makedirs(out_dir)
saver = tf.train.Saver()
saver.save(sess, out_dir+"/tensorflow_model")

# Retrieve the protobuf graph definition
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

all_nodes = [] 
print("Name of all the nodes/operations in the loaded tensorflow model:\n")
for n in input_graph_def.node:
    all_nodes.append(n.name)
    print(n.name)
for n in all_nodes:
    if any(i == "gradients" for i in n.split("/")):
        break
    output_node_names = n
print("Extracted output node name:\n", output_node_names)    
print("Collect   output node name:\n",y)
