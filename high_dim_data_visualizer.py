## VISUALIZE HIGH DIMENSIONAL VECTORS WITHOUT LABELS

# import os
# import tensorflow as tf 
# import numpy as np

# LOG_DIR = "logs/second"

# high_dim_vectors = np.random.rand(50, 10)

# vectors = tf.Variable(high_dim_vectors, name="high_dim_vectors")

# with tf.Session() as sess:
#     saver = tf.train.Saver([vectors])
#     sess.run(tf.global_variables_initializer())
#     saver.save(sess, os.path.join(LOG_DIR, "images.ckpt"))




## VISUALIZE HIGH DIMENSIONAL VECTORS WITH LABELS

import os
import numpy as np
import tensorflow as tf 
from tensorflow.contrib.tensorboard.plugins import projector 

LOG_DIR = "logs/third"
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

high_dimensional_vectors = np.random.rand(50, 10)
tensors_to_visualize = tf.Variable(high_dimensional_vectors, name="high_dim_vectors")

with open(metadata, "w") as metadata_file:
    classes = 10
    total_data_num = high_dimensional_vectors.shape[0]
    # write header 
    metadata_file.write("id\tlabel\n")
    for i in range(total_data_num):
        # assign random class to the data point [1 to 10]
        metadata_file.write("%i\t%d\n" % (i,(int((np.random.rand())*10) + 1)))


with tf.Session() as sess:
    saver = tf.train.Saver([tensors_to_visualize])

    # initialize the Variable in the session 
    sess.run(tensors_to_visualize.initializer)
    saver.save(sess, os.path.join(LOG_DIR, "hi_dim_vectors.ckpt"))

    # configurations to provide metadata/labels
    config = projector.ProjectorConfig()
    # one can add multiple embeddings
    embedding = config.embeddings.add()
    embedding.tensor_name = tensors_to_visualize.name 
    # link this tensor to its metadata file (e.g labels)
    embedding.metadata_path = metadata
    # save a config file that tensorboard will read during startup
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)


## Quick Note:
## after running this code 
## adjust metadata path in projector_config.pbtxt file