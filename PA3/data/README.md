# ImageNet-100 data source

PA3 uses the Hugging Face dataset
[`clane9/imagenet-100`](https://huggingface.co/datasets/clane9/imagenet-100),
revision `0519dc2f402a3a18c6e57f7913db059215eee25b`. It contains 126,689
training images and 5,000 validation images from the CMC ImageNet-100 class
list; images have a 160-pixel short side.

`student_code.load_imagenet100` retrieves this exact revision via the
`datasets` package. Pass its `cache_dir` argument to choose the cache location;
otherwise Hugging Face uses its standard local cache. Do not commit downloaded
data or cache files to this repository.

The full cached dataset is approximately 8.4 GB. Students must download and
retain this full cache; the final evaluator uses the same pinned dataset.

Use is for this educational course and remains subject to the ImageNet terms
included in the source dataset card.
