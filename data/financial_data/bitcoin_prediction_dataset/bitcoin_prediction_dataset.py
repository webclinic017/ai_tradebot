"""bitcoin_prediction_dataset dataset."""

import tensorflow_datasets as tfds

# TODO(bitcoin_prediction_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(bitcoin_prediction_dataset): BibTeX citation
_CITATION = """
"""


class BitcoinPredictionDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for bitcoin_prediction_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(bitcoin_prediction_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
      builder=self,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
          'date': tfds.features.Tensor(shape=(), dtype=tf.string),
          'btc': tfds.features.Tensor(shape=(), dtype=tf.int32),
          'eth': tfds.features.Tensor(shape=(), dtype=tf.int32),
          'trends_btc': tfds.features.Tensor(shape=(), dtype=tf.int32),
          'trends_eth': tfds.features.Tensor(shape=(), dtype=tf.int32),
          'trends_total': tfds.features.Tensor(shape=(), dtype=tf.int32),
          'negative': tfds.features.Tensor(shape=(), dtype=tf.int32),
          'neutral': tfds.features.Tensor(shape=(), dtype=tf.int32),
          'other': tfds.features.Tensor(shape=(), dtype=tf.int32),
          'positive': tfds.features.Tensor(shape=(), dtype=tf.int32),
      }),
      # If there's a common (input, target) tuple from the
      # features, specify them here. They'll be used if
      # `as_supervised=True` in `builder.as_dataset`.
      supervised_keys=None,  # e.g. ('image', 'label')
      homepage='https://dataset-homepage/',
      citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(bitcoin_prediction_dataset): Downloads the data and defines the splits
    path = dl_manager.download('https://github.com/johann-su/ai_tradebot/blob/main/data/financial_data/price_prediction_dataset.csv.csv)

    # TODO(bitcoin_prediction_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
      'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(bitcoin_prediction_dataset): Yields (key, example) tuples from the dataset
    with open(path) as f:
      yield 'key', {}
