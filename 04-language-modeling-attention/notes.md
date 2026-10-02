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