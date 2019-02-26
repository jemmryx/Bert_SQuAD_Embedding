# coding = utf-8
import tensorflow as tf
import csv
import numpy as np
import modeling, tokenization

data_root = './cased_L-12_H-768_A-12/'
bert_config_file = data_root + 'bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = data_root + 'bert_model.ckpt'
bert_vocab_file = data_root + 'vocab.txt'


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab
        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file + '.from', 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 1:
                    print 'Warning: incorrectly formatted line in vocabulary file: %s\n' % line
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    max_size, self._count)
                    break
        print "Finished constructing vocabulary of %i total words. Last word added: %s" % (
        self._count, self._id_to_word[self._count - 1])

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def write_metadata(self, fpath):
        """Writes metadata file for Tensorboard word embedding visualizer as described here:
          https://www.tensorflow.org/get_started/embedding_viz
        Args:
          fpath: place to write the metadata file
        """
        print "Writing word embedding metadata file to %s..." % (fpath)
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in xrange(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


vocab = Vocab('../../../vocab/vocab20000',20000)
vsize = vocab.size()
print("squad vocab size is:{}".format(vsize))

# graph
input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')

model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment)

encoder_last_layer = model.get_sequence_output()

embedding = np.random.uniform(-0.25, 0.25, (vsize, 768))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    token = tokenization.FullBasicTokenizer(vocab_file=bert_vocab_file)
    for word in vocab._word_to_id:
        query = unicode(word)
        print(query)
        split_tokens = token.tokenize(query)
        #print(split_tokens)
        word_ids = token.convert_tokens_to_ids(split_tokens)
        print(word_ids)
        word_mask = [1] * len(word_ids)
        word_segment_ids = [0] * len(word_ids)

        fd = {input_ids: [word_ids], input_mask: [word_mask], segment_ids: [word_segment_ids]}
        last = sess.run([encoder_last_layer], feed_dict=fd)
        print(last[0][0][0][0])
        embedding[vocab.word2id(query)] = last[0][0][0]

    embedding = np.array(embedding, dtype=np.float32)
    with open(data_root+'bert_embed.txt', 'w') as src_embed_out_file:
        np.savetxt(src_embed_out_file, embedding)

