import collections
import pickle
import re
import string

import numpy as np
import tensorflow as tf
from progressbar import ProgressBar

import Config
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import nltk
from nltk.corpus import wordnet


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config, x_train, y_train, x_dev, y_dev, x_test, y_test):

        self.build_graph(graph, embedding_array, Config, x_train, y_train, x_dev, y_dev, x_test, y_test)

    def build_graph(self, graph, embedding_array, Config, x_train, y_train, x_dev, y_dev, x_test, y_test):

        with tf.Graph().as_default():
            sess = tf.Session()
            sequence_length = len(x_train[0])
            num_classes = len(y_train[0])

            with sess.as_default():
                self.input_headlines = tf.placeholder(tf.int32, [None, sequence_length])
                self.input_labels = tf.placeholder(tf.float32, [None, num_classes])
                self.dropout = tf.placeholder(tf.float32)
                self.len = tf.constant(1024, tf.int32)

                self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
                self.embedded_chars = tf.nn.embedding_lookup(self.embeddings, self.input_headlines)
                self.embedded_chars = tf.expand_dims(self.embedded_chars, -1)
                
                pools = []

                for filter_size in Config.filter_sizes:
                    filter_shape = [filter_size, Config.embedding_size, 1, Config.num_filters]
                    weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                    biases = tf.Variable(tf.constant(0.1, shape=[Config.num_filters]))
                    conv = tf.nn.conv2d(
                        self.embedded_chars,
                        weights,
                        strides=[1, 1, 1, 1],
                        padding="VALID")

                    h = tf.nn.relu(tf.add(conv, biases))
                    
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID')
                    pools.append(pooled)

                
                total_filters = Config.num_filters * len(Config.filter_sizes)
                self.concat_pool = tf.concat(pools, 3)
                self.flattenedPool = tf.reshape(self.concat_pool, [-1, total_filters])
                self.h_drop = tf.nn.dropout(self.flattenedPool, self.dropout)
                l2_loss = tf.constant(0.0)
                h_shape = self.h_drop.get_shape().as_list()
                weights_input = tf.Variable(
                    tf.random_normal([h_shape[1], Config.hidden_size], stddev=0.1))
                biases_input = tf.Variable(tf.zeros([Config.hidden_size]))
                weights_output = tf.Variable(
                    tf.random_normal([Config.hidden_size, num_classes], stddev=0.1))
                results = tf.matmul(self.h_drop, weights_input)
                results = tf.add(results, biases_input)
                # Activation
                h = tf.pow(results, tf.fill(tf.shape(results), 3.0))
                self.scores = tf.matmul(h, weights_output, transpose_a=False, transpose_b=False)
                self.predictions = tf.argmax(self.scores, 1)
                l2_loss = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(biases_input)
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_labels)
                self.loss = tf.reduce_mean(losses) + Config.lam * l2_loss

                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
                self.precision = tf.metrics.precision(
                                    tf.argmax(self.input_labels,1),
                                    self.predictions,
                                    weights=None,
                                    metrics_collections=None,
                                    updates_collections=None,
                                    name=None
                                )
                self.recall = tf.metrics.recall(
                                    tf.argmax(self.input_labels,1),
                                    self.predictions,
                                    weights=None,
                                    metrics_collections=None,
                                    updates_collections=None,
                                    name=None
                                )
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads = optimizer.compute_gradients(self.loss)
                self.app = optimizer.apply_gradients(grads, global_step=global_step)
                saver = tf.train.Saver(tf.global_variables())
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                def train(x_batch, y_batch):
                    feed_dict = {
                      self.input_headlines: x_batch,
                      self.input_labels: y_batch,
                      self.dropout: Config.dropout
                    }
                    _, step, loss, accuracy, precision,recall = sess.run(
                        [self.app, global_step, self.loss, self.accuracy, self.precision, self.recall],feed_dict)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % Config.display_step == 0:
                        f1 = round(2*precision[0]*recall[0]/(precision[0]+recall[0]),2)
                        print("Step "+str(step))
                        print( "Avg loss: "+str(round(loss,4)))
                        print("Accuracy: "+str(round(accuracy,4)))
                        print("Precision: "+str(round(precision[0],4)))
                        print("Recall: "+str(round(recall[0],4)))
                        print("F1-Score: "+str(f1))
            
                    
                def dev(x_batch, y_batch, writer=None):
                        
                        feed_dict = {
                          self.input_headlines: x_batch,
                          self.input_labels: y_batch,
                          self.dropout: Config.dropout
                        }
                        step, loss, accuracy, precision,recall = sess.run(
                            [global_step, self.loss, self.accuracy, self.precision, self.recall],
                            feed_dict)
                        print("\nEvaluation at dev set")
                        print("Step "+str(step))
                        print( "Avg loss: "+str(round(loss,4)))
                        print("Accuracy: "+str(round(accuracy,4)))
                        print("Precision: "+str(round(precision[0],4)))
                        print("Recall: "+str(round(recall[0],4)))
                        print("F1-Score: "+str(round(2*precision[0]*recall[0]/(precision[0]+recall[0]),2)))
                        print("\n\n")
                    

                def test(x_test, y_test):
                    feed_dict = {
                          self.input_headlines: x_test,
                          self.input_labels: y_test,
                          self.dropout: Config.dropout
                        }
                    loss, accuracy, precision,recall = sess.run(
                        [self.loss, self.accuracy, self.precision, self.recall],
                        feed_dict)

                    print("\n\n----------------------------------------\nEvaluation at test set")
                    print( "Avg loss: "+str(round(loss,4)))
                    print("Accuracy: "+str(round(accuracy,4)))
                    print("Precision: "+str(round(precision[0],4)))
                    print("Recall: "+str(round(recall[0],4)))
                    print("F1-Score: "+str(round(2*precision[0]*recall[0]/(precision[0]+recall[0]),2)))
                    print("----------------------------------------")
                    print("\n\n")



                def batch(data):
                    data = np.array(data)
                    data_size = len(data)
                    num_batches_per_epoch = int((len(data)-1)/Config.batch_size) + 1
                    for epoch in range(Config.max_iter):
                        shuffle_indices = np.random.permutation(np.arange(data_size))
                        shuffled_data = data[shuffle_indices]
                        for batch_num in range(num_batches_per_epoch):
                            start_index = batch_num * Config.batch_size
                            end_index = min((batch_num + 1) * Config.batch_size, data_size)
                            yield shuffled_data[start_index:end_index]


                batches = batch(list(zip(x_train, y_train))) 
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    current_step = tf.train.global_step(sess, global_step)
                    train(x_batch, y_batch)
                    if current_step % Config.validation_step == 0:
                        dev(x_dev, y_dev)

                test(x_test, y_test)



def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    multipleSpaces = re.compile("\ {2,}")
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " " + punctuation.encode('utf-8') + " ")
    for i in range(10):
        text = text.replace(str(i), " " + str(i).encode('utf-8') + " ")
    text = multipleSpaces.sub(" ", text)
    text2 = "\n".join(line.strip() for line in text.split("\n"))
    printable = set(string.printable)
    textString = filter(lambda x: x in printable, text2)

    tokens = []

    for line in textString.split("\n"):
        temp = []
        if line != '':
            for word in line.split(" "):
                if word != '':
                    temp.append(lemmatizer.lemmatize(word))

            tokens.append(" ".join(temp))
    return "\n".join(tokens)

def replaceUnkownWords(vocabulary, sentence):
    UNK = Config.UNKNOWN
    temp=[]

    for word in sentence.split(" "):
        if word in vocabulary:
            temp.append(word)
        else:
            temp.append(UNK)
    return " ".join(temp)
    
def addWordsToDictionary(dataList,words):
    for sentence in dataList.split("\n"):
        for word in sentence.split(" "):
            words.append(word)

def genDictionaries(nonClickbait, clickbait):
    nonClickbait = preprocess(nonClickbait)
    clickbait = preprocess(clickbait)
    words = []
    
    addWordsToDictionary(clickbait,words)
    addWordsToDictionary(nonClickbait,words)

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(Config.vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0 
          unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return nonClickbait, clickbait, dictionary, reverse_dictionary

def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]

def addPadding(dataList, temp):
    if len(temp)<31:
        req = 31 - len(temp)
        temp =  temp + [0]*req
        dataList.append(temp)

def addWordData(data, dataList):
    for sentence in data:
        temp=[]
        for word in sentence.split(" "):
            temp.append(getWordID(word))
        # Adding padding to the sentence if lenght is less than some threshold (in this case, 31)
        addPadding(dataList,temp)

def getFeatures(nonClickBait, clickbait):
    nonClickbaitList=[]
    clickbaitList = []
    addWordData(nonClickBait,nonClickbaitList)
    addWordData(clickbait,clickbaitList)
    return (nonClickbaitList, clickbaitList)

def genTrainExamples(nonClickBait, clickbait):
    vocabulary = wordDict.keys()
    nonClickbait = []
    clickBait = []
    nonClickBait = nonClickBait.split("\n")
    clickbait = clickbait.split("\n")
    pbar = ProgressBar()
    print("Non_Clickbait_Data", len(nonClickBait))
    for i in pbar(range(len(nonClickBait))):
        nonClickbait.append(replaceUnkownWords(vocabulary, nonClickBait[i]))
    pbar = ProgressBar()
    print("Clickbait_Data", len(clickbait))
    for i in pbar(range(len(clickbait))):
        clickBait.append(replaceUnkownWords(vocabulary, clickbait[i]))
    clickbait = clickBait[:-1]
    nonClickBait = nonClickbait[:-1]
    return getFeatures(nonClickBait, clickbait)

def genTestExamples(nonClickBait, clickbait):
    vocabulary = wordDict.keys()
    nonClickBait = preprocess(nonClickBait)
    clickbait = preprocess(clickbait)
    nonClickBait = nonClickBait.split("\n")
    clickbait = clickbait.split("\n")
    nonClickbait = []
    clickBait = []
    pbar = ProgressBar()
    print("Non_Clickbait_Data", len(nonClickBait))
    for i in pbar(range(len(nonClickBait))):
        nonClickbait.append(replaceUnkownWords(vocabulary, nonClickBait[i]))
    pbar = ProgressBar()
    print("Clickbait_Data", len(clickbait))
    for i in pbar(range(len(clickbait))):
        clickBait.append(replaceUnkownWords(vocabulary, clickbait[i]))
    clickbait = clickBait[:-1]
    nonClickBait = nonClickbait[:-1]
    return getFeatures(nonClickBait, clickbait)

def load_embeddings(filename, wordDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01

    return embedding_array

def get_wordnet_pos(tag):

    treebank_tag = tag[1]
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


if __name__ == '__main__':

    # Read non-clickbait and clickbait data from the file
    nonClickBait = open("data/genuine_train_data").read()
    clickbait = open("data/clickbait_train_data").read()

    #Get some metrics of the loaded data 
    data, count, wordDict, reverse_dictionary = genDictionaries(nonClickBait, clickbait)

    #get the embeddings
    embedding_filename = 'word2vec.model'
    embedding_array = load_embeddings(embedding_filename, wordDict)
    print "Generating Traning Examples"
    nonClickbait, clickbait = genTrainExamples(data, count)
    print "Done."

    #Get train features
    trainFeats = np.concatenate((nonClickbait,clickbait),axis=0)

    #Label the data accordingly and put the labels in one variable
    nonClickbaitLabels = []
    clickbaitLabels = []
    for a in nonClickbait:
        nonClickbaitLabels.append([0,1])
    for a in clickbait:
        clickbaitLabels.append([1,0])
    trainLabels = np.concatenate([nonClickbaitLabels, clickbaitLabels], 0)

    #Shuffle the data and partition it into two steps according to the ratio mentioned in the Config file
    x_train = []
    x_dev = []
    devSample = int(Config.dev_sample * float(len(trainFeats)))
    for a in range(len(trainFeats)):
        if a < len(trainFeats)/2 + devSample/2 and a > len(trainFeats)/2 - devSample/2:
            x_dev.append(trainFeats[a])
        else:
            x_train.append(trainFeats[a])
    y_train=[]
    y_dev=[]
    for a in range(len(trainFeats)):
        if a < len(trainFeats)/2 + devSample/2 and a > len(trainFeats)/2 - devSample/2:
            y_dev.append(trainLabels[a])
        else:
            y_train.append(trainLabels[a]) 


    print("Generating Test Examples")
    nonClickbaitTest = open("data/genuine_test_data").read()
    clickbaitTest = open("data/clickbait_test_data").read()
    nonClickbaitTest, clickbaitTest = genTestExamples(nonClickbaitTest, clickbaitTest)
    print("Done.")
    testFeats = np.concatenate((nonClickbaitTest,clickbaitTest),axis=0)
    nonClickbaitTestLabels = []
    clickbaitTestLabels = []
    for a in nonClickbaitTest:
        nonClickbaitTestLabels.append([0,1])
    for a in clickbaitTest:
        clickbaitTestLabels.append([1,0])
    testLabels = np.concatenate([nonClickbaitTestLabels, clickbaitTestLabels], 0)


    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config, x_train, y_train, x_dev, y_dev, testFeats, testLabels)
