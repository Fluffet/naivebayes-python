import math

class NaiveBayes(object):

  def __init__(self):
    self.classes = set()
    self.vocabulary = {}
    self.class_count = {}
    self.probabilities = {}
    self.global_vocabulary = set()

  def predict(self, unpredicted):
    """ Predict something. Returns the class as a string.
    Uses log probability to prevent underflow errors"""

    maximum_probable_class = ""
    max_prob = float("-inf")

    for classname in self.classes:
      current_class_count = self.class_count[classname]
      total_document_count = sum( self.class_count.values() )
      classname_prob = math.log( current_class_count / total_document_count )

      prob = classname_prob

      for obj in unpredicted:
        if obj in self.probabilities[classname]:
          word_prob = self.probabilities[classname][obj]
          prob += word_prob

      if prob > max_prob:
        maximum_probable_class = classname
        max_prob = prob

    return maximum_probable_class

  def test_accuracy(self, test_data):
    correct_classifications = 0

    for case in test_data:
      prediction = classifier.predict(case[0])

      if prediction == case[1]: correct_classifications += 1

    accuracy = correct_classifications / len(test_data)
    return accuracy

  @classmethod
  def train(cls, data, k=1):
    """Train a new classifier on training data using maximum
    likelihood estimation and additive smoothing.

    k = smoothing constant
    data: [something_hashable, "class"]
    example 1 (unigram): [["This", "is", "an", "example", "string"], "class1"]
    example 2  (bigram): [[('This', 'is'), ('is', 'an'), ('an', 'example'), ('example', 'string')], "class2"]
    """
    nb = cls()

    # Scan through all documents to fetch classes
    # first.. lazy, but results in cleaner code
    for d in data: nb.classes.add(d[1])
    for c in nb.classes:
      nb.vocabulary[c]    = {}
      nb.probabilities[c] = {}
      nb.class_count[c]   = 0


    for d in data:
        classname = d[1]

        nb.class_count[classname] += 1

        for obj in d[0]:
          if obj not in nb.global_vocabulary:
            nb.global_vocabulary.add(obj)
            for class_name in nb.classes:
              nb.vocabulary[class_name][obj] = k
              nb.probabilities[class_name][obj] = k

          if obj in nb.vocabulary[classname]:
            nb.vocabulary[classname][obj] += 1

    for classname in nb.classes:
      word_count_sum = sum(nb.vocabulary[classname].values())

      for word_key in nb.vocabulary[classname]:
        word_count = nb.vocabulary[classname][word_key]
        word_probability = math.log( ( word_count ) / ( word_count_sum + (k * len(nb.global_vocabulary)) ) )
        nb.probabilities[classname][word_key] = word_probability

    return nb
