# BERT
bert finetune and inference with gluonnlp  

# Dependencies
* mxnet-cu100==1.6.0
* gluonnlp==0.9.1

# Finetune
### Prepare date
Download data [here](https://pan.baidu.com/s/183N4hpF8fTaHfQiZuUW3XA). 提取码：ojuy
<br>数据集为IMDB的影评数据集，格式已经处理过，方便读取
### Begin to finetune
使用预训练的GluonNLP BERT BASE模型精调，运行`BERT_finetune_base.ipynb`
<br>使用预训练的GluonNLP BERT LARGE模型精调，运行`BERT_finetune_large.ipynb`
<br>由于是在`google colab`上运行的，本地运行时前面的一些步骤可以省略，需要修改相应的数据集路径，最后一次运行的结果（硬件信息，中间过程，log等）已包含在相应文件中以供查看
### Model
根据最后一次的精调结果，选择了在验证集上精度最高的model进行推理。[BERT BASE](https://drive.google.com/file/d/1-75dm6ePDBa-GyCNAl9-75oSqjcoep_g/view?usp=sharing) model, [BERT LARGE](https://drive.google.com/file/d/1-8y0nIfZYnOLyKNqNmooMUBClDUtz_-V/view?usp=sharing) model.
# Inference
使用精调后的BERT BASE模型进行推理，运行`BERT_inference_base.ipynb`
<br>使用精调后的BERT LARGE模型进行推理，运行`BERT_inference_large.ipynb`
<br>同样，由于是在`google colab`上运行的，本地运行时前面的一些步骤可以省略，需要修改相应的模型路径以及`test.tsv`的路径。其中，`test.tsv`用来储存待推理的sentence，如需修改待推理的sentence，则需对该文件进行修改。`test.tsv`只储存一句待推理的sentence，格式为`sentence label`，以`\t`进行分隔，若无标签，则输入0，且忽略输出结果中的label这一行。最后一次inference的结果也已包含在对应文件中以供查看。
