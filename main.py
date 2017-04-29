import tensorflow as tf
from model import Model
from data_prep import extract_features, split_data

flags = tf.app.flags

# Directories
flags.DEFINE_boolean('test', False, 'true if testing, false if training [Default: False]')
flags.DEFINE_string('data_dir', 'disney_touring_data/AK07.csv', 'Directory for data sets [Default: disney_touring_data/')
flags.DEFINE_string('save_dir', 'save', 'Save path directory[Default: save')

# Training parameters
flags.DEFINE_boolean('load', False, 'Start training from saved model? [False]')
flags.DEFINE_integer('save_period', 80, 'Save period [80]')
flags.DEFINE_integer('batch_size', 128, 'Batch size during training and testing[128]')
flags.DEFINE_integer('num_epochs', 128, 'Number of epochs for training[128]')
flags.DEFINE_float('learning rate', 0.0001, 'Learning rate [0.0001]')
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay- L2 regularization [0.001]')
flags.DEFINE_float('d_rate', 1.0, 'Dropout rate-1.0 to have it off [1.0]')
flags.DEFINE_string("feature_maps", "[50,100,150,200,200,200,200]", "Feature maps in CNN [50,100,150,200,200,200,200]")
flags.DEFINE_string("kernels", "[1,2,3,4,5,6,7]", "CNN kernels [1,2,3,4,5,6,7]")

FLAGS = flags.FLAGS


def main(_):
    train, test = split_data(FLAGS.data_dir)
    train = extract_features(FLAGS.batch_size, train)
    with tf.Session() as session:

        model = Model(flags)
        session.run(tf.initialize_all_variables())

        if FLAGS.test:
            model.load(session)
            model.eval(session, test)
        else:
            if FLAGS.load:
                model.load(session)
            model.train(session, train)

if __name__ == '__main__':
    tf.app.run()
