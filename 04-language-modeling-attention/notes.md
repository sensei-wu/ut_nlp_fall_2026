# Notes for week 4:

## RNNs using glove embedding

I played around with an RNN network. RNNs are nowadays out of fashion thanks to Transformers. Its limitations are well known, still it is a good exercise to learn the 
basics of sequential inputs and outputs. I built `04-language-modeling-attention/rnn_glove.py` with the sample dataset `20newsgroups` from `sklearn.datasets`, using two categories: 
`["rec.autos", "sci.space"]`. I achieved a training accuracy of 97% and test accuracy of 88%. The model was tested on some made up sentences and got the following results:

```
'the goalie made a great save in the third period'      -> sci.space (0.84)
'nasa launched the shuttle into orbit'                  -> sci.space (0.98)
'the ferrari was slow'                                  -> rec.autos (0.81)
'there are speed limits'                                -> rec.autos (0.77)
```

The results looked sensible, except for the goalie sentence.

I analyzed the result using Claude and found its following observation interesting:

    The goalie sentence → space (0.84). A 2-class model can't answer "neither". Softmax always splits 100% between the two labels it knows, so a hockey sentence has to land somewhere. The 0.84 shows the bigger problem: high confidence on out-of-distribution input. Confidence measures which known class the input looks more like, not how likely the answer is to be correct. Why space in particular is a guess on my part. "Period" (as in orbital period) and "third" (as in third stage) plausibly show up in space posts.

Other observation:

Test accuracy plateaus around 0.85–0.88 while train accuracy climbs to 0.96. That's a mild overfitting gap.

## Replacing RNN with GRU

A simple change in nn module call to replace RNN with GRU improved test accuracy and also gave a smoother learning. Results are below:

```
train 1139  test 750  labels ['rec.autos', 'sci.space']
vocab size 4875
glove: 2828/4875 words
epoch 1  loss 0.694  train 0.653  test 0.589
epoch 2  loss 0.620  train 0.805  test 0.733
epoch 3  loss 0.408  train 0.860  test 0.812
epoch 4  loss 0.281  train 0.901  test 0.861
epoch 5  loss 0.209  train 0.934  test 0.879
epoch 6  loss 0.141  train 0.971  test 0.887
epoch 7  loss 0.097  train 0.989  test 0.907
epoch 8  loss 0.038  train 0.992  test 0.907
'the goalie made a great save in the third period'      -> sci.space (0.82)
'nasa launched the shuttle into orbit'                  -> sci.space (1.00)
'the ferrari was slow'                                  -> rec.autos (0.84)
'there are speed limits'                                -> rec.autos (0.73)
```

Increasing maximum length of the sentences from 100 to 300 did not help much. More text might be giving diminishing returns here, because the topic is usually clear early in a post.