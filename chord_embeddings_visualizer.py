## VISUALIZE THE EMBEDDINGS OF ONLY THOSE CHORDS WHOSE NAME WE KNOW

import os
import numpy as np
import tensorflow as tf 
import keras
from tensorflow.contrib.tensorboard.plugins import projector 

from data_processing import get_chord_dict
from configuration import chord_triad_tuple_to_chord_name

LOG_DIR = "logs/chord_embeddings/unshifted_epoch_30_nottingham_dataset"
metadata = os.path.join(LOG_DIR, 'metadata.tsv')


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR) 



# extract embedding from chord model
# chord_model = "models/chords/1564233418-Shifted_False_Lr_1e-05_EmDim_10_opt_Adam_bi_False_lstmsize_512_trainsize_4_testsize_1_samples_per_bar8/model_Epoch100_4.pickle"
chord_model = "models/chords/1564241436-Shifted_False_Lr_1e-05_EmDim_10_opt_Adam_bi_False_lstmsize_512_trainsize_4_testsize_1_samples_per_bar8_dataset_data/Nottingham_unshifted_dataset/original/model_Epoch30_4.pickle"
model = keras.models.load_model(chord_model)
embeddings = model.layers[0].get_weights()
embeddings = np.array(embeddings)
# embeddings shape: (1, 50, 10)
embeddings = embeddings.reshape(-1,embeddings.shape[2])

# find the chord embeddings with the name
chords_with_name = {}

index_to_chords = get_chord_dict()[1]

total_data_num = embeddings.shape[0]
for i in range(total_data_num):
    triad_tuple = index_to_chords[i]
    if triad_tuple in chord_triad_tuple_to_chord_name:
        chords_with_name[i] = chord_triad_tuple_to_chord_name[triad_tuple]


new_embeddings = []
for i in chords_with_name.keys():
    new_embeddings.append(embeddings[i])


# high_dimensional_vectors = np.random.rand(50, 10)
# tensors_to_visualize = tf.Variable(high_dimensional_vectors, name="high_dim_vectors")

tensors_to_visualize = tf.Variable(np.array(new_embeddings), name="chord_with_name_embeddings")


with open(metadata, "w") as metadata_file:
    for i in chords_with_name.keys():
        metadata_file.write("{}\n".format(chords_with_name[i]))



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

