import io
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
from bert import data
import random
import csv


np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.gpu(0)
bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                        dataset_name='book_corpus_wiki_en_uncased',
                        pretrained=True, ctx=ctx, use_pooler=True,
                        use_decoder=False, use_classifier=False)
bert_classifier = nlp.model.BERTClassifier(bert_base, num_classes=2, dropout=0.1)
bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
bert_classifier.hybridize(static_alloc=True)
filename="./base_2.params"
nlp.utils.load_parameters(bert_classifier,filename,ctx=ctx)



Sentences = input('Please enter your comment here:')


with open('test.tsv','w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerow([Sentences, '0'])  





field_separator = nlp.data.Splitter('\t')
data_inference_row = nlp.data.TSVDataset(filename='test.tsv',
                field_separator=field_separator)

bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
max_len = 128
all_labels = ["0", "1"]
pair = False
transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
                        class_labels=all_labels,
                        has_label=True,
                        pad=True,
                        pair=pair)
data_inference=data_inference_row.transform(transform)
batch_size = 1
data_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_inference],
                      batch_size=batch_size,
                      shuffle=True)


data_loader = mx.gluon.data.DataLoader(data_inference, batch_sampler=data_sampler)
for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(data_loader):
  token_ids = token_ids.as_in_context(ctx)
  valid_length = valid_length.as_in_context(ctx)
  segment_ids = segment_ids.as_in_context(ctx)
  label = label.as_in_context(ctx)
  out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
#  print(data_inference_row[0][0])
#  print('label: ',data_inference_row[0][1])
  if (out.argmax(axis=1))[0]==1:
    # print('inference: 1')
    print('Your comment is Positive!')
  else:
    print('inference: ')
    print('Your comment is Negative!')
