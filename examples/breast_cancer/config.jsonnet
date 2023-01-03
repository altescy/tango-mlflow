local preprocess_scale = std.extVar('preprocess_scale');
local learning_rate = std.parseJson(std.extVar('learning_rate'));
local penalty = std.parseJson(std.extVar('penalty'));

local ref(name) = { type: 'ref', ref: name };
local preprocess(dataset) = {
  type: 'preprocess',
  scale: preprocess_scale,
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
        learning_rate: learning_rate,
        penalty: penalty,
      },
    },
    metrics: {
      type: 'evaluate',
      dataset: ref('preprocessed_test'),
      model: ref('model'),
    },
  },
}
