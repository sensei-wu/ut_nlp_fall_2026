# Notes for week 2:

## Learning rate

I learnt in Optimization course that optimal learning rate (lr) for GD $\eta$ is inversely related to Lipschitz constant $L$ of the gradient for any L-smooth f. For a NN where the loss function is usually non convex, there is no guarantee to find a global minima. 

But what about lr? $\eta < 2/L$ still guarantees descent for any L-smooth f; convexity isn't needed. NN losses aren't globally L-smooth (no finite L), so the guarantee doesn't
formally apply.But we will have to figure $\eta$ empirically because $L$ is unbounded, varies during training, and is impractical to compute.

### Adam

Why does Adam work so well in practice? It is the combination of momentum and adaptive method that replaces expensive Hessian computations using adaptive learning rates that make it a very useful technique.

## SST Dataset labels

We are using [Stanford Sentiment Treebank or SST](https://huggingface.co/datasets/stanfordnlp/sst) dataset for assignments. We are using a label compressed version of it where there are only two labels - positive and negative. A few things that I have noticed:

    0   While some will object to the idea of a Vietnam picture with such a rah-rah , patriotic tone , Soldiers ultimately achieves its main strategic objective : dramatizing the human cost of the conflict that came to define a generation .

why is the above example negative? To me it looked positive and when I checked with Claude, it got wrong as well (using Opus 5). It has the following to say, when I asked what could be the reasons:

    Two possibilities. Either the fragment came from a review whose overall verdict was negative and the snippet inherited the document label, same as the "sin" case. Or the label is just noisy; SST phrase annotations were crowdsourced and the boundary cases aren't clean.
 

Claude got another one wrong as well, which I would have trouble predicting altogether. That is the "sin" case mentioned:

    0   Lookin ' for sin , American-style ?

For the above one, I cannot imagine any model learning to predict it because of lack of any sentiment bearing information.
Another example where I think a neutral sentiment is falling back to negative.

    0   Although I did n't hate this one , it 's not very good either .

I thought it was neutral-collapse, but the assignment README says neutrals are discarded, so what explains these labels? Perhaps they were mislabeled or there was a disagreement among annotators.