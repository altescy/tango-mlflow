local ref(name) = { type: 'ref', ref: name };
local preprocess(dataset) = {
  type: 'preprocess',
  scale: 'standard',
  dataset: ref(dataset),
};

{
  steps: {
    train: {
      type: 'load_dataset',
      subset: 'train',
    },
    test: {
      type: 'load_dataset',
      subset: 'test',
    },
    preprocessed_train: preprocess('train'),
    preprocessed_test: preprocess('test'),
    model: {
      type: 'train',
      dataset: ref('preprocessed_train'),
      model: {
        type: 'logistic_regression',
        learning_rate: 1e-2,
        penalty: 0.5,
      },
    },
    metrics: {
      type: 'evaluate',
      dataset: ref('preprocessed_test'),
      model: ref('model'),
    },
  },
}
