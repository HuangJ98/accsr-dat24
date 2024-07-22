# NeMo LR Schedulers

Here is a short list of the LR schedulers available in NeMo. The latest list of available schedulers can be found at the end of [this file](https://github.com/NVIDIA/NeMo/blob/main/nemo/core/optim/lr_scheduler.py). When possible, I have copied parts of the classes strings, or links to websites that introduce the schedulers. For some I couldn't find anything, but the definition in the aforementioned file is usually brief.

- WarmupPolicy: Adds warmup kwargs and warmup logic to lr policy
- WarmupHoldPolicy: Variant of WarmupPolicy which maintains high learning rate for a defined number of steps.
- SquareAnnealing
- [CosineAnnealing](https://paperswithcode.com/method/cosine-annealing)
- [NoamAnnealing](https://docs.allennlp.org/main/api/training/learning_rate_schedulers/noam/)
- NoamHoldAnnealing: Unlike NoamAnnealing, the peak learning rate can be explicitly set for this scheduler. The schedule first performs linear warmup, then holds the peak LR, then decays with some schedule for the remainder of the steps. Therefore the min-lr is still dependent on the hyper parameters selected.
- WarmupAnnealing
- [InverseSquareRootAnnealing](https://paperswithcode.com/method/)inverse-square-root-schedule
- [T5InverseSquareRootAnnealing](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.nnet.schedulers.html#speechbrain.nnet.schedulers.)InverseSquareRootScheduler
- SquareRootAnnealing
- [PolynomialDecayAnnealing](https://paperswithcode.com/method/polynomial-rate-decay)
- PolynomialHoldDecayAnnealing
- [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
- [ExponentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html)
- [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
- [CyclicLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html)
