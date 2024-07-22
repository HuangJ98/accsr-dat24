# NeMo's Conformer Transducer

NeMo offers a pre-trained Conformer Transducer which performs best according to a study. Here I look into how it works and investigate how to incorporate the accent classifier.

I believe the way to go is using hooks. As explained [here](https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/), hooks are callable objects with a certain set signature that can be registered to any nn.Module object. When the forward() method is triggered in a model forward pass, the module itself, along with its inputs and outputs are passed to the forward_hook before proceeding to the next module. You can place them on the output of any layer of an instantiated model, which is that you have when initializing a pre-trained model with NeMo.