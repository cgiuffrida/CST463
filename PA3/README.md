# Homework 03: CNN Design on ImageNet-100

## Background

Convolutional neural networks (CNNs) use local filters and shared weights to
recognize visual patterns wherever they occur in an image. Their architecture
still leaves important design choices: channel widths, number of stages,
normalization, pooling, regularization, data augmentation, optimizer, and
learning-rate schedule. Those choices affect both how quickly a model learns
and how well it generalizes.

This assignment uses the fixed 100-class ImageNet subset hosted as
[`clane9/imagenet-100`](https://huggingface.co/datasets/clane9/imagenet-100).
Each example is a natural-color image and has one integer class label. The
course pins an exact revision so every experiment uses the same classes and
split. See `data/README.md` for download, cache, and provenance details.

## Setup

Run `uv sync` from the repository root if you have not already. **You must
download and retain the full ImageNet-100 cache (about 8.4 GB) before working
on this assignment.** The first call to `make_imagenet100_loaders` downloads
the pinned dataset through Hugging Face and stores it in the standard Hugging
Face cache. Start this download early on a stable connection; later runs reuse
the cache.

To download both splits before beginning experiments, run this from the PA3
directory:

```bash
uv run python -c "import student_code as sc; sc.make_imagenet100_loaders()"
```

Set the optional `cache_dir` argument if the default cache location does not
have enough space. The dataset's access terms apply to every copy you use.

## Overview

There are three parts to this homework:

1. Complete `student_code.py` and pass `test_unittests.py`.
2. Use `notebook.ipynb` to run controlled CNN experiments.
3. Write and compile `report.md` to `report.pdf`.

`student_code.py` is due one week after release. `report.pdf` and your final
benchmark source/configuration are due one week later; see Canvas for dates.

## Step 1: Build the CNN experiment tools

Complete `student_code.py` until all tests pass:

```bash
uv run pytest test_unittests.py
```

The file contains these pieces:

- `load_imagenet100` and `make_imagenet100_loaders`: load the pinned Hugging
  Face dataset and create training/validation loaders with image transforms.
- `CNNClassifier`: a configurable CNN with `forward`, `loss`,
  `training_step`, `validation_step`, `configure_optimizers`, and
  `configure_scheduler` methods.
- `evaluate` and `train_model`: the supplied experiment loop.
- `count_trainable_parameters`: report a model's capacity.

The supplied CNN is a deliberately modest baseline: convolution, batch
normalization, ReLU, and pooling repeated for each requested channel width,
then global average pooling and a linear classifier. Start by making that
baseline correct. During the analysis week you may revise its architecture and
training choices as long as its public method contracts remain intact.

`CNNClassifier` packages its loss and optimizer choices with the model. This
is a course convention for this assignment, not a requirement imposed by
PyTorch: an `nn.Module` fundamentally needs registered parameters and a
`forward` method. The packaging lets one fixed training loop run many models.

Read docstrings as contracts. Tests cover important cases, but passing tests
does not replace understanding the training loop. In particular, `forward`
must return **logits**, not softmax probabilities, because cross-entropy
expects logits.

## Step 2: Investigate CNN design choices

Use the notebook to answer:

> **Under a fixed training budget, which CNN architecture and optimization
> choices improve validation accuracy on ImageNet-100, and why?**

Begin with the supplied baseline. Form a hypothesis, then make at least three
controlled comparisons. Change one primary factor at a time when possible: for
example, channel widths/depth, dropout, data augmentation, optimizer,
learning rate, weight decay, or scheduler. Record the dataset source, seed,
batch size, epochs, parameter count, and training cost for each run.

Use development-validation results to choose a final design; do not claim that
a single lucky run proves a conclusion. Your report should include a baseline,
learning curves, 3--5 controlled experiments, and an explanation of the
evidence that selected your final design. A focused comparison is better than
an exhaustive grid search.

You may use the course GPU trial service at most **four** times. Each trial
trains for 15 epochs and returns a complete training/development-validation
log. Treat those trials as experiments to plan, not as a live leaderboard.

### Final benchmark and bonus

Submit `student_code.py` and `benchmark-config.yaml` only; do not submit
trained weights. The instructor will train each valid final submission from
newly initialized weights on the full pinned dataset for 90 epochs under several
instructor-selected seeds. Runs have a two-hour safety limit. A run that
exceeds the limit is not eligible for the benchmark bonus, but this never
reduces the normal homework grade. The configuration records the constructor
arguments for your model and requested training-loader batch size; the
evaluator controls the dataset revision, epoch count, evaluation split, and seeds.

Bonus results use the mean private-evaluation accuracy across the final five
epochs and across the selected seeds. You will receive your own complete
per-epoch log under a hashed run ID after the deadline. No public leaderboard
will be published.

Keep models reasonably sized: models that exceed the two-hour safety limit are
not eligible for the benchmark bonus. Parameter count will be reported but is
not itself a scoring term.

### Allowed work

Your submitted model must be written by you using standard PyTorch tensor
operations, `torch.nn` layers, `torch.nn.functional`, `torch.optim`, and
PyTorch learning-rate schedulers. You may build your own convolutional,
normalization, pooling, activation, dropout, and linear-layer arrangements.

Do **not** use pretrained weights, model zoos or imported architectures
(`torchvision.models`, `timm`, Hugging Face models, or external model code),
external datasets, or downloaded weights. Cite any non-course source that
influenced a design idea, as required by the course collaboration policy.

## Step 3: Write your report

Use `report.md` to make an evidence-based argument. Include your hypothesis,
research question, baseline, experimental protocol, controlled comparisons,
parameter/training-cost measurements, plots or tables, and the reasoning that
selects your final model. State what you expect the final benchmark model to
do before benchmark results are released.

Write for a reader who could reproduce your work. Results without a stated
configuration or a comparison without a control are not persuasive evidence.

Compile the one-file submission from this directory:

```bash
make compile-report
```

Use `make check-report` before compiling to catch missing figure references.
Do not include your name in the report.

## Submitting

- **`student_code.py`**, due one week after release.
- **`report.pdf`**, due one week after release of the code milestone.
- **Final `student_code.py` and `benchmark-config.yaml`**, due with the report.

Follow the Canvas submission instructions for the exact deadlines and the
benchmark-source submission location.
