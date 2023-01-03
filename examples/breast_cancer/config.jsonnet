local preprocessor = std.extVar('preprocessor');
local learning_rate = std.parseJson(std.extVar('learning_rate'));
local penalty = std.parseJson(std.extVar('penalty'));

local ref(name) = { type: 'ref', ref: name };

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
    preprocessor: {
      type: 'train_preprocessor',
      dataset: ref('train'),
      preprocessor: preprocessor,
    },
    preprocessed_train: {
      type: 'preprocess',
      dataset: ref('train'),
      preprocessor: ref('preprocessor'),
    },
    preprocessed_test: {
      type: 'preprocess',
      dataset: ref('test'),
      preprocessor: ref('preprocessor'),
    },
    model: {
      type: 'train_model',
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
