# GoXLM

GoXLM is an independently trained decoder-only Transformer research project
focused on science, mathematics, programming, and engineering. Fresh runs start
from random weights. The baseline has 3,257,856 parameters, a 128-token context,
and a UTF-8 byte tokenizer.

## Source checkout

The public CI workflow uses Python 3.12 and PyTorch 2.12.1 from the official
CPU wheel index. The tests use synthetic fixtures and require neither external
datasets nor model weights.

```sh
GOXLM_TEST_DEVICE=cpu OMP_NUM_THREADS=1 python -B -m unittest training.test_train goxlm_v02.test_v02 goxlm_v02.test_governance goxlm_review.test_pilot
```

- [Model, training, and catalog documentation](goxlm_v02/README.md)
- [Human rights-review workflow](goxlm_review/README.md)
- [Example training configuration](configs/train.example.json)
- [Public release checklist](PUBLIC_RELEASE.md)
- [Local CUDA validation and benchmarking](training/CUDA.md)

Public Git history tracks the source, tests, and documentation. Training inputs,
checkpoints, rights evidence, reviewer records, and other research artifacts
live separately in the private research workspace. Live baseline verification
and governed training require the authorized local artifacts. Dataset
admission requires a human review of each document's exact content and
evidence for commercial AI training. A clean Git tree can still contain
ignored private artifacts.

## Contributing

Use a feature branch and a pull request against `main`. Complete the public
release checklist before pushing and review the CPU checks before merging.
