import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from functools import reduce


def make_dataset():
    X,y = make_classification(n_samples=60000, n_features = 6, n_informative=4, n_redundant=2, n_repeated=0,
                                         n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
                                         hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42)

    return X_train, X_test, y_train, y_test




def parse_custom_weights( graph , hidden_shape , input_dim,  n_class ):
    with graph.as_default():
        shape_list = list(hidden_shape)
        shape_list.insert(0,input_dim)
        shape_list.append(n_class)
        weight_variable_list = [tf.Variable(tf.truncated_normal([shape_list[i], shape_list[i+1]], stddev = math.sqrt(2.0 / (input_dim)))) for i in range(len(shape_list) - 1)]
        return weight_variable_list

def parse_custom_biases( graph , hidden_shape , input_dim,  n_class ):
    with graph.as_default():
        shape_list = list(hidden_shape)
        shape_list.append(n_class)
        bias_variable_list = [tf.Variable(tf.zeros([shape])) for  shape in  shape_list]
        return bias_variable_list

def parse_mlp_perceptron(x , weights , biases , n_layer, graph ):
    with graph.as_default():
        previous_layer = x
        for i in range(n_layer):
            next_layer = tf.add(tf.matmul(previous_layer, weights[i]), biases[i])
            next_layer = tf.nn.tanh(next_layer)
            previous_layer = next_layer

        out_layer = tf.matmul(next_layer, weights[-1]) + biases[-1]
        return out_layer

def add_metrics(graph,unnormalized_scores, label_piece):
    with graph.as_default():
        binary_predictions = tf.argmax(unnormalized_scores, 1)
        correct = tf.equal(binary_predictions, tf.argmax(label_piece, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        graph.add_to_collection('acc', accuracy)
        graph.add_to_collection('preds', binary_predictions)


def get_tensors(graph=tf.get_default_graph()):
    print([t for op in graph.get_operations() for t in op.values()])
    return [t for op in graph.get_operations() for t in op.values()]

def main_graph(trainingdata, testdata , traininglabels, testlabels, n_class , batch_size = 32, test_batch_size = 32, hidden_shape = (20,20,20) , betanum = 0.00001):

    input_dim = trainingdata.shape[1]
    n_layer = len(hidden_shape)
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():

        one_hot_batch_train_label = tf.one_hot(traininglabels, n_class)
        one_hot_batch_test_label = tf.one_hot(testlabels, n_class)


        weights = parse_custom_weights(graph, hidden_shape, input_dim, n_class)
        biases =  parse_custom_biases(graph, hidden_shape, input_dim, n_class)

        training_dataset = tf.data.Dataset.from_tensor_slices((trainingdata ,one_hot_batch_train_label))
        test_dataset = tf.data.Dataset.from_tensor_slices((testdata, one_hot_batch_test_label))


        train_batches = training_dataset.batch(batch_size)
        test_batches = test_dataset.batch(test_batch_size)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_batches.output_types, train_batches.output_shapes)

        training_iterator = train_batches.make_initializable_iterator()
        test_iterator = test_batches.make_initializable_iterator()

        graph.add_to_collection('train_iter' ,training_iterator )
        graph.add_to_collection('test_iter', test_iterator)
        graph.add_to_collection('handle', handle)

        data_piece, label_piece = iterator.get_next()

        data_piece = tf.cast(data_piece, tf.float32)

        output = parse_mlp_perceptron(data_piece, weights, biases, n_layer, graph)
        regularizer = sum([ tf.nn.l2_loss(weight ) for weight in weights] )

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label_piece) , name = 'loss')
        graph.add_to_collection('loss', loss)

        cost = tf.reduce_mean(loss + betanum * regularizer)
        optimizer = tf.train.AdamOptimizer().minimize(cost, name = 'optim_op')

        add_metrics(graph, output, label_piece)

        return graph


def train(graph, epochnum = 100, display_step = 1):
    query_list = [graph.get_collection_ref(str)[0] for str in ['train_op','loss','acc']]

    with graph.as_default():
        init = tf.global_variables_initializer()
        training_epochs = epochnum

        with tf.Session() as session:
            session.run(init)
            training_handle = session.run(graph.get_collection_ref('train_iter')[0].string_handle())

            # Training cycle
            for epoch in range(training_epochs):
                epoch_loss = 0.
                acc = 0
                session.run(graph.get_collection_ref('train_iter')[0].initializer)
                step = 0
                while True:
                    try:
                        data_feed = {graph.get_collection_ref('handle')[0]: training_handle}
                        _,batch_loss,batch_acc = session.run(query_list ,data_feed)
                        epoch_loss += batch_loss
                        acc += batch_acc
                        step = step + 1
                    except tf.errors.OutOfRangeError:
                        print('epoch_train_end')
                        epoch_loss = epoch_loss/step
                        acc = acc/step
                        break
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(epoch_loss))
                    print("Epoch:", '%04d' % (epoch + 1), "acc=", "{:.9f}".format(acc))

    return graph

def inference(graph):
    query_list = [graph.get_collection_ref(str)[0] for str in ['acc','loss']]

    with graph.as_default():
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            test_handle = session.run(graph.get_collection_ref('test_iter')[0].string_handle())
            test_loss = 0.
            test_acc = 0.
            session.run(graph.get_collection_ref('test_iter')[0].initializer)
            step = 0
            while True:
                try:
                    data_feed = {graph.get_collection_ref('handle')[0]: test_handle}
                    test_batch_acc, test_batch_loss = session.run(query_list ,data_feed)
                    test_loss += test_batch_loss
                    test_acc += test_batch_acc
                    step = step + 1
                except tf.errors.OutOfRangeError:
                    print('test_end')
                    test_loss = test_loss/step
                    test_acc = test_acc/step
                    break
            print("Test_" +  "loss=", "{:.9f}".format(test_loss))
            print("Test_"  +  "acc=", "{:.9f}".format(test_acc))


def main():
    trainingdata, testdata, traininglabels, testlabels = make_dataset()
    graph = main_graph(trainingdata, testdata, traininglabels, testlabels, n_class = 2, batch_size=64, test_batch_size=64,
                   hidden_shape=(10, 10, 10, 10 , 10), betanum=0.0001)
    graph = train(graph, epochnum=600, display_step=1)
    inference(graph)


if __name__ == "__main__":
    main()