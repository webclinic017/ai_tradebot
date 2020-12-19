"""financial_sentiment_dataset dataset."""

import tensorflow_datasets as tfds
import csv
import tensorflow as tf
import os

# TODO(financial_sentiment_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
# Sentiment Analysis for Financial News
## Context

This dataset (FinancialPhraseBank) contains the sentiments for financial news headlines from the perspective of a retail investor.

## Content

The dataset contains two columns, "Sentiment" and "News Headline". The sentiment can be negative, neutral or positive.

## Acknowledgements

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4), 782-796.
"""

# TODO(financial_sentiment_dataset): BibTeX citation
_CITATION = """\
@article{YADAV2020589,
  title = "Sentiment analysis of financial news using unsupervised approach",
  journal = "Procedia Computer Science",
  volume = "167",
  pages = "589 - 598",
  year = "2020",
  note = "International Conference on Computational Intelligence and Data Science",
  issn = "1877-0509",
  doi = "https://doi.org/10.1016/j.procs.2020.03.325",
  url = "http://www.sciencedirect.com/science/article/pii/S1877050920307912",
  author = "Anita Yadav and C K Jha and Aditi Sharan and Vikrant Vaish",
  keywords = "Sentiment Analysis, Financial news, Semantic Orientation, Unsupervised techniques",
  abstract = "Sentiment analysis aims to determine the sentiment strength from a textual source for good decision making. This work focuses on application of sentiment analysis in financial news. The semantic orientation of documents is first calculated by tuning the existing technique for financial domain. The existing technique is found to have limitations in identifying representative phrases that effectively capture the sentiment of the text. Two alternative techniques - one using Noun-verb combinations and the other a hybrid one, are evaluated. Noun-verb approach yields best results in the experiment conducted."
}
"""

_DATA_OPTIONS = [
    'one_hot', 'class_label'
]

# TODO: Add import options
# @dataclasses.dataclass
# class DatasetConfig(tfds.core.BuilderConfig):
#   def __init__(self, *, label_format=None, **kwargs):

#     if label_format not in _DATA_OPTIONS:
#       print(f"THE LABEL: {label_format}")
#       raise ValueError("labels must be one of %s" % _DATA_OPTIONS)

#     super(DatasetConfig, self).__init__(**kwargs)
#     self.label_format = label_format


class FinancialSentimentDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for financial_sentiment_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }
  # BUILDER_CONFIGS = [
  #   DatasetConfig(
  #     name=config_name,
  #     description=("A dataset containing news headlines and a correspnding sentiment score."),
  #     version=tfds.core.Version(
  #       "1.0.0",
  #       "First Release"),
  #     labels=config_name,
  #   ) for config_name in _DATA_OPTIONS
  # ]


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(financial_sentiment_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
      builder=self,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
        'text': tfds.features.Text(),
        # 'score': self.builder_config.labels
        'score': tfds.features.Tensor(shape=(4,), dtype=tf.float32)
      }),
      # If there's a common (input, target) tuple from the
      # features, specify them here. They'll be used if
      # `as_supervised=True` in `builder.as_dataset`.
      supervised_keys=('text', 'score'),  # e.g. ('image', 'label')
      homepage='https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news',
      citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(financial_sentiment_dataset): Downloads the data and defines the splits
    path = dl_manager.download_kaggle_data('ankurzing/sentiment-analysis-for-financial-news')

    # TODO(financial_sentiment_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
      'train': self._generate_examples(path=os.path.join(path, 'all-data.csv')),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(financial_sentiment_dataset): Yields (key, example) tuples from the dataset

    with open(path, encoding="ISO-8859-1") as f:
      for i, row in enumerate(csv.DictReader(f, fieldnames=['score', 'text'])):
        score = ''
        text = ''
        for key, value in row.items():
          if key=='score':
            score = value
          elif key == 'text':
            text = value

        categories = tf.constant(['positive', 'neutral', 'negative'])
        indices = tf.range(len(categories), dtype=tf.int64)
        table_init = tf.lookup.KeyValueTensorInitializer(categories, indices)
        num_oov_buckets = 1
        table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

        # one hot encode labels of dataset
        label = table.lookup(tf.constant(score))
        label_enc = tf.one_hot([label], depth=len(categories) + num_oov_buckets)

        # And yield (key, feature_dict)
        yield i, {
          'text': tf.constant(text, name='text'),
          'score': label_enc[0]
          # 'score': label_enc[0] if self.builder_config.name == 'one_hot' else label
        }