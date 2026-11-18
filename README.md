# GoXLM

GoXLM is an independently trained decoder-only Transformer research project
focused on science, mathematics, programming, and engineering. Fresh runs start
from random weights. The baseline has 3,257,856 parameters, a 128-token context,
and a UTF-8 byte tokenizer.

## Source checkout

Use Python with a compatible PyTorch installation. The tests run on CPU with
synthetic fixtures and require neither external datasets nor model weights.

```sh
python -B -m unittest goxlm_v02.test_v02 goxlm_v02.test_governance goxlm_review.test_pilot
```

- [Model, training, and catalog documentation](goxlm_v02/README.md)
- [Human rights-review workflow](goxlm_review/README.md)
- [Example training configuration](configs/train.example.json)

The public repository contains code, tests, and documentation. Training inputs,
checkpoint files, research evidence, and human decision records are maintained
in the private research workspace. Live baseline verification and governed
training require their separately authorized local artifacts. Dataset admission
requires a human review of each document's exact content and evidence for
commercial AI training.

## Contributing

Use a feature branch and a pull request against `main`. Run the synthetic suite
and inspect the staged diff and file list before every push. Stage explicitly
selected source files. Review any proposed public research material individually
before adding it to a pull request. Keep private information, credentials,
checkpoint files, and training corpora in the local workspace.
