import os
import re
import unicodedata

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from bidi.algorithm import get_display
from sklearn.model_selection import train_test_split



path = 'C:/Users/hp/Desktop/internship2020/NMT att/machine-translation-nmt/model/real.txt'

class LanguageIndex():
    ''' Creates a word -> index mapping

    This class creates a word -> index mapping
    (e.g., "dad" -> 5) and vice-versa
    (e.g., 5 -> "dad") for each language,

    Attributes:
        lang: A langauage to map to its index.
    '''

    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0

        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


class nmt_dt2ar(object):

    def __init__(self):
        # Run prepossessing

        lines = open(path, encoding='UTF-8').read().strip().split('\n')
        sent_pairs = []
        try:
          for line in lines:
              sent_pair = []
              DT, AR = line.split("/")
              DT = self.preprocess_sentence(DT)
              sent_pair.append(DT)
              AR = self.preprocess_sentence(AR)
              sent_pair.append(AR)
              sent_pairs.append(sent_pair)
        except ValueError:
            pass

        # Load the dataset
        self.input_tensor, self.target_tensor, self.inp_lang, self.targ_lang, self.max_length_inp, self.max_length_targ = self.load_dataset(sent_pairs, len(lines))

        # Creating training and validation sets using an 80-20 split
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(self.input_tensor, self.target_tensor, test_size=0.2)

        self.BUFFER_SIZE = len(input_tensor_train)
        self.BATCH_SIZE = 24
        self.N_BATCH = self.BUFFER_SIZE // self.BATCH_SIZE
        self.embedding_dim = 256
        self.units = 300
        self.vocab_inp_size = len(self.inp_lang.word2idx)
        self.vocab_tar_size = len(self.targ_lang.word2idx)


        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(self.BUFFER_SIZE)
        dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        
        example_input_batch, example_target_batch = next(iter(dataset))
        example_input_batch.shape, example_target_batch.shape

        self.encoder = self.Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)

        # sample input
        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_hidden = self.encoder(example_input_batch, sample_hidden)
        

        attention_layer = self.BahdanauAttention(10)
        attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
        




        self.encoder = self.Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        self.decoder = self.Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)

    def unicode_to_ascii(self, s):
        """
        Converts the unicode file to ascii

        :param s: UniCode file
        :return: ASCII file
        """
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w):
        """
        Convert Unicode to ASCII
        Creating a space between a word and the punctuation following it
        eg: "he is a boy." => "he is a boy ."
        Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        Replacing everything with space except (a-z, A-Z, ا-ي ".", "?", "!", ",")
        Adding a start and an end token to the sentence

        :param w: A single word
        :return: Single normalize word
        """

        w = self.unicode_to_ascii(w.lower().strip())

        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)


        w = '<start> %s <end>' % w
        return w


    def max_length(self, tensor):
        """
        :param tensor: Tensor of indexed words
        :return: The maximum size for the longest tensor
        """
        return max(len(t) for t in tensor)

    def load_dataset(self,pairs, num_examples):
            # pairs => already created cleaned input, output pairs

            # index language using the class defined above    
        inp_lang = LanguageIndex(en for en, ma in pairs)
        targ_lang = LanguageIndex(ma for en, ma in pairs)
            
            # Vectorize the input and target languages
            
            # Tunisian sentences
        input_tensor = [[inp_lang.word2idx[s] for s in en.split(' ')] for en, ma in pairs]
            
            # arabic sentences
        target_tensor = [[targ_lang.word2idx[s] for s in ma.split(' ')] for en, ma in pairs]
            
            # Calculate max_length of input and output tensor
            # Here, we'll set those to the longest sentence in the dataset
        max_length_inp, max_length_tar = self.max_length(input_tensor), self.max_length(target_tensor)
            
            # Padding the input and output tensor to the maximum length
        input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                        maxlen=max_length_inp,
                                                                        padding='post')
            
        target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                        maxlen=max_length_tar, 
                                                                        padding='post')
            
        return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar 
    
    def gru(self, units):
        """
        :param units: number of units
        :return: GRU units
        """

        # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')
    class Encoder(tf.keras.Model):
          def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
            super(nmt_dt2ar.Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = nmt_dt2ar.gru(self, self.enc_units)


          def call(self, x, hidden):
            x = self.embedding(x)
            output, state =self.gru(x, initial_state = hidden)
            return output, state

          def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.enc_units))

    class BahdanauAttention(tf.keras.layers.Layer):
          def __init__(self, units):
            super(nmt_dt2ar.BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

          def call(self, query, values):
            # query hidden state shape == (batch_size, hidden size)
            # query_with_time_axis shape == (batch_size, 1, hidden size)
            # values shape == (batch_size, max_len, hidden size)
            # we are doing this to broadcast addition along the time axis to calculate the score
            query_with_time_axis = tf.expand_dims(query, 1)

            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            score = self.V(tf.nn.tanh(
                self.W1(query_with_time_axis) + self.W2(values)))

            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)

            return context_vector, attention_weights   

    class Decoder(tf.keras.Model):
          def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
            super(nmt_dt2ar.Decoder, self).__init__()
            self.batch_sz = batch_sz
            self.dec_units = dec_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru =nmt_dt2ar.gru(self, self.dec_units)

            self.fc = tf.keras.layers.Dense(vocab_size)

            # used for attention
            self.attention = nmt_dt2ar.BahdanauAttention(self.dec_units)

          def call(self, x, hidden, enc_output):
            # enc_output shape == (batch_size, max_length, hidden_size)
            context_vector, attention_weights = self.attention(hidden, enc_output)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            output, state = self.gru(x)

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size, vocab)
            x = self.fc(output)

            return x, state, attention_weights


    def loss_function(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def load_model(self, save_dir):
        encoder_prefix = os.path.join(save_dir, "encoder_weights")
        decoder_prefix = os.path.join(save_dir, "decoder_weights")

        self.encoder.load_weights(encoder_prefix)
        self.decoder.load_weights(decoder_prefix)

 

    def evaluate(self, sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
        attention_plot = np.zeros((max_length_targ, max_length_inp))

        sentence = self.preprocess_sentence(sentence)

        inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

            # storing the attention weigths to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += targ_lang.idx2word[predicted_id] + ' '

            if targ_lang.idx2word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    # function for plotting the attention weights
    def plot_attention(self, attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        heatmap = ax.matshow(attention, cmap='rainbow')

        for y in range(attention.shape[0]):
            for x in range(attention.shape[1]):
                ax.text(x, y, '%.4f' % attention[y, x],
                        horizontalalignment='center',
                        verticalalignment='center', color='black')

        fig.colorbar(heatmap)

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        plt.show()

    def translate(self, sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
        result, sentence, attention_plot = self.evaluate(sentence, encoder, decoder, inp_lang, targ_lang,
                                                         max_length_inp, max_length_targ)

        #print('Input: {}'.format(sentence))
        #print('Predicted translation: {}'.format(result))

        #attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        #self.plot_attention(attention_plot, get_display(arabic_reshaper.reshape(sentence)).split(' '),result.split(' '))

        return result

    def translate_api_response(self, sentence):
        result = self.translate(sentence, self.encoder, self.decoder, self.inp_lang, self.targ_lang, self.max_length_inp,self.max_length_targ)
        result = result.replace('<start>','').replace('<end>','').strip()
        confidence, backend = 0, -1
        return result, confidence, backend
