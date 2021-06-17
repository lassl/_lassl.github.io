---
layout: post
title: lightning-transformers로 살펴본 Hydra 어플리케이션 구성
author: hyungjun.kim
categories: [implementation]
tags: [Hydra, Config, lighting-transformers]
---

이번 편에서는 [간단한 예제로 살펴본 Hydra를 이용한 어플리케이션 구성](./2021-05-31-hydra.md) 내용에 이어 `lightning-transformers`을 사례를 보면서 정리해보겠습니다.

```
conf/
┣ backbone/  # Configs defining the backbone of the model/pre-trained model if any
┣ dataset/ # Configs defining datasets
┣ optimizer/ # Configs for optimizers
┣ scheduler/ # Configs for schedulers
┣ task/ # Configs defining the task, and any task specific parameters
┣ tokenizer/ # Configs defining tokenizers, if any.
┣ trainer/ # Configs defining PyTorch Lightning Trainers, with different configurations
┣ training/ # Configs defining training specific parameters, such as batch size.
┗ config.yaml # The main entrypoint containing all our chosen config components
```

먼저 가장 기본이 되는 `config.yaml`을 정의합니다. 아래의 예시를 보면 `lightning-transformers`에서는 [`conf/config.yaml`](https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/config.yaml)에서 defaults로 task, optimizer, scheduler, training, trainer로 하위 계층 구조를 정의하고 있습니다.

```yaml
defaults: # loads default configs
  - task: default
  - optimizer: adamw
  - scheduler: linear_schedule_with_warmup
  - training: default
  - trainer: default

experiment_name: ${now:%Y-%m-%d}_${now:%H-%M-%S}
log: False
ignore_warnings: True # todo: check warnings before release
```

> 주) `now`는 Hydra에서 미리 등록되어 사용할 수 있습니다. [https://github.com/facebookresearch/hydra/blob/master/hydra/core/utils.py#L185](https://github.com/facebookresearch/hydra/blob/master/hydra/core/utils.py#L185)

다음으로 [`conf/task/default.yaml`](https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/task/default.yaml)를 살펴보면, `package`와 `_group_` 개념이 사용되고 있습니다.

`conf/task/default.yaml`

```yaml
# @package task
defaults:
  - /dataset@_group_: default

# By default we turn off recursive instantiation, allowing the user to instantiate themselves at the appropriate times.
_recursive_: false

_target_: lightning_transformers.core.model.TaskTransformer
optimizer: ${optimizer}
scheduler: ${scheduler}
```

dataset의 config group을 default로 설정하였기 때문에, [`conf/dataset/default.yaml`](https://github.com/PyTorchLightning/lightning-transformers/blob/master/conf/dataset/default.yaml)을 로드합니다. 또한 optimizer와 scheduler는 각각 `conf/config.yaml`에서 앞서 정의된 `optimizer`와 `scheduler`를 변수로 받습니다. `_recursive_` 는 해당 파일에 종속된 다른 config들을 instantiate 할 것인지를 나타냅니다.

----

대표적인 예로 LanguageModeling를 살펴봅니다.

```sh
python train.py task=nlp/language_modeling
```

train.py를 살펴보면 가장 기본이 되는 `conf/config.yaml` config_path와 config_name을 정의하고 있습니다.

```python
"""The shell entry point `$ pl-transformers-train` is also available"""
import hydra
from omegaconf import DictConfig

from lightning_transformers.cli.train import main


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
```

`conf/config.yaml`

```yaml
defaults: # loads default configs
  - task: default
  - optimizer: adamw
  - scheduler: linear_schedule_with_warmup
  - training: default
  - trainer: default

experiment_name: ${now:%Y-%m-%d}_${now:%H-%M-%S}
log: False
ignore_warnings: True # todo: check warnings before release
```

[이전 편](./2021-05-31-hydra.md)에서 배운 overrding을 살펴봅니다. 예를 들어 Q. batch 사이즈를 바꾸고 싶은데 어떻게 해야할까요? batch 사이즈는 `training/default.yaml`에 정의되어 있습니다. 따라서 CLI 명령어에 training.batch_size=8 등을 추가하면 변경이 됩니다.

`conf/training/default.yaml`

```yaml
run_test_after_fit: True
lr: 5e-5
output_dir: '.'

# read in dataset
batch_size: 16
num_workers: 16
```

```sh
python train.py task=nlp/language_modeling training.batch_size=8
```

본격적으로 `task`에 대해 알아봅니다. 먼저 `conf/config.yaml`에 의해 `conf/task/default.yaml` 내용을 입력 받아야 합니다. 그런데, CLI 명령에 의해 `conf/task/nlp/language_modeling.yaml`로 대체됩니다.

`conf/task/nlp/language_modeling.yaml`

```yaml
# @package task
defaults:
  - nlp/default
  - override /dataset@_group_: nlp/language_modeling/default
_target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer
downstream_model_type: transformers.AutoModelForCausalLM
```

그런데, 처음 defaults에 의해 `conf/task/nlp/default.yaml` 필요한 값들을 불러옵니다.

`conf/task/nlp/default.yaml`

```yaml
# @package task
defaults:
  - /dataset@_group_: default

# By default we turn off recursive instantiation, allowing the user to instantiate themselves at the appropriate times.
_recursive_: false

_target_: lightning_transformers.core.model.TaskTransformer
optimizer: ${optimizer}
scheduler: ${scheduler}
```

dataset은 `conf/dataset/default.yaml` 파일을 가져옵니다.

`conf/dataset/default.yaml`

```yaml
# @package dataset
_target_: lightning_transformers.core.data.TransformerDataModule
cfg:
  # torch data-loader specific arguments
  batch_size: ${training.batch_size}
  num_workers: ${training.num_workers}
```

그런데, `language_modeling`은 `TransformerDataModule` 대신 `LanguageModelingTransformer`을 사용합니다. 그리고 모형 아키텍쳐 역시 `transformers.AutoModelForCausalLM` 사용해야합니다. 따라서 CLI 명령어를 통해 `python train.py task=nlp/language_modeling` 실행이 필요합니다.

`conf/nlp/language_modeling.yaml`

```yaml
# @package task
defaults:
  - nlp/default
  - override /dataset@_group_: nlp/language_modeling/default
_target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer
downstream_model_type: transformers.AutoModelForCausalLM
```

또한, 데이터셋 역시 그룹을 `default`(`task/nlp/default.yaml`에서 정의된)에서 `dataset/nlp/language_modeling/default.yaml`로 overrding 합니다.  그리고 `_target_`과 `downstream_model_type`로 각각 정의되어 있습니다.

> override에 관한 자세한 것은 [https://hydra.cc/docs/next/advanced/override_grammar/basic/](https://hydra.cc/docs/next/advanced/override_grammar/basic/) 참고하시기 바랍니다.

따라서, `task=nlp/language_modeling.yaml`를 정리해보면 (1) `conf/task/nlp/default.yaml`에 의해

```yaml
# @package task
defaults:
  - /task/default
  - /backbone@_group_: nlp/default # use AutoModel backbone by default
  - override /dataset@_group_: nlp/default
backbone: ${backbone}
```

`backbone`값을 받고, (2) `conf/nlp/language_modeling.yaml` 에 의해 `_target_`과 `downstream_model_type`을 받습니다. 그리고, (3) 최초 `conf/task/default.yaml`에서 `_recuresive_`와 `optimizer` 그리고 `scheduler`를 가져옵니다. 이때 `_target_`은 `lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer`로 override 되었습니다.

`conf/task/default.yaml`

```yaml
# @package task
defaults:
  - /dataset@_group_: default

# By default we turn off recursive instantiation, allowing the user to instantiate themselves at the appropriate times.
_recursive_: false

_target_: lightning_transformers.core.model.TaskTransformer
optimizer: ${optimizer}
scheduler: ${scheduler}
```

이를 `task` 전체에 대해 정리해보면 다음과 같습니다.


```yaml
task:
  _recursive_: false
  _target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer
  optimizer: ${optimizer}
  scheduler: ${scheduler}
  backbone: ${backbone}
  downstream_model_type: transformers.AutoModelForCausalLM
```

`task`는 앞서 살펴본대로 `defaults`에 의해 `backbone`과 `dataset`에도 영향을 줍니다. 먼저 `backbone`을 살펴보면 `tokenizer`를 설정하고 `pretrained_model_name_or_path`를 결정하고 있습니다.

`conf/backbone/nlp/default.yaml`

```yaml
# @package backbone
defaults:
  - /tokenizer@_group_: autotokenizer # use AutoTokenizer by default
pretrained_model_name_or_path: bert-base-cased
```

`dataset`은 `nlp/language_modeling/default.yaml`에서

```yaml
# @package task
defaults:
  - nlp/default
  - override /dataset@_group_: nlp/language_modeling/default
_target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer
downstream_model_type: transformers.AutoModelForCausalLM
```

`nlp/default`로 부터 cfg 다음을 입력 받고,

```yaml
# @package dataset
defaults:
  - /dataset/default
_target_: lightning_transformers.core.nlp.HFDataModule
cfg:
  dataset_name: null
  dataset_config_name: null
  train_file: null
  validation_file: null
  test_file: null
  train_val_split: null
  max_samples: null
  cache_dir: null
  padding: 'max_length'
  truncation: 'only_first'
  preprocessing_num_workers: 1
  load_from_cache_file: True
  max_length: 128
  limit_train_samples: null
  limit_val_samples: null
  limit_test_samples: null
```

`dataset/default`로 부터

```yaml
# @package dataset
_target_: lightning_transformers.core.data.TransformerDataModule
cfg:
  # torch data-loader specific arguments
  batch_size: ${training.batch_size}
  num_workers: ${training.num_workers}
```

위의 cfg를 입력 받습니다.

따라서, 최종적으로 2개의 cfg가 합쳐지며,

```yaml
dataset:
  _target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingDataModule
  cfg:
    batch_size: ${training.batch_size}
    num_workers: ${training.num_workers}
    dataset_name: null
    dataset_config_name: null
    train_file: null
    validation_file: null
    test_file: null
    train_val_split: null
    max_samples: null
    cache_dir: null
    padding: max_length
    truncation: only_first
    preprocessing_num_workers: 1
    load_from_cache_file: true
    max_length: 128
    limit_train_samples: null
    limit_val_samples: null
    limit_test_samples: null
    block_size: null
```

`_target_`은 `conf/task/nlp/language_modeling.yaml`의 `_target_`인 `lightning_transformers.task.nlp.language_modeling.LanguageModelingDataModule`로 변경 됩니다.

또 다른 예제, [Causal Language Modeling](./2021-05-29-lightning-transformers-clm.md)에서 `config`와 `datasets`, 그리고 `model` 등을 Hydra와 연결하여 잘 설명하고 있습니다.

----
그렇다면 새로운 흥미로운 예제에 응용을 해볼까요?

최근 한국에서 공개된 [KLUE(Korean Language Understanding Evlauation)](!https://klue-benchmark.com/) 데이터가 있습니다. 현재 KLUE 데이터셋과 벤치마크 모형이 오픈소스로 제공되고 있으므로 이를 Hydra와 Ligtning-transformer를 이용하여 학습하는 법에 대해 살펴봅니다.

![KLUE Leaderboard](https://user-images.githubusercontent.com/11376047/122318262-baec0100-cf59-11eb-9d9f-7f7d79ad3be7.png)


다양한 TASK 중에서 비교적 손쉬운 [KLUE-KNLI](https://klue-benchmark.com/tasks/68/overview/description) 분류 문제(classification)를 다뤄봅니다.

데이터셋은 [https://huggingface.co/datasets/viewer/?dataset=klue](https://huggingface.co/datasets/viewer/?dataset=klue) 에서도 확인할 수 있으며, KLUE 벤치마크 모형은 [https://huggingface.co/klue](https://huggingface.co/klue) 에서 확인할 수 있습니다.

> KLUE 데이터셋은 [minho.ryu](https://lassl.github.io/menu/authors.html#minho-ryu)님과 `jungwhank`님께서 기여하였습니다. [https://github.com/huggingface/datasets/blob/master/datasets/klue/klue.py](https://github.com/huggingface/datasets/blob/master/datasets/klue/klue.py)

`ligtning-transformers`를 돌아와서 아래의 레포지토리를 `install` 혹은 `clone` 합니다.

```
git clone https://github.com/PyTorchLightning/lightning-transformers
```

먼저 데이터셋 부터 정의합니다.

`nli`에 맞게 task를 `nlp/text_classification`로 설정하고, `dataset_name`과 `dataset_config_name`을 정의합니다.
그리고 모형 backbone으로는 `klue/bert-base`로 설정하고, gpu 갯수와 `max_epochs` 등 가장 기본적인 파라미터만 셋팅합니다.

```
python train.py task=nlp/text_classification dataset.cfg.dataset_name=klue dataset.cfg.dataset_config_name=nli backbone.pretrained_model_name_or_path=klue/bert-base trainer.gpus=1 trainer.max_epochs=10
```

[KLUE-NLI 리더보드](https://klue-benchmark.com/tasks/68/leaderboard/task)에서 KLUE-BERT-base의 `ACC`는 81.63%로 보고되고 있습니다. 위의 스크립트 실행 결과 약 81.4%의 정확도를 보입니다. 위의 스크립트에서 배치 사이즈와 learning rate, 시퀀스 최대 길이 등을 간단히 변경한 것만으로도 아래의 스크립트로도 벤치마크 성능보다 뛰어난 82%의 정확도를 얻을 수 있습니다.

```
python train.py task=nlp/text_classification dataset.cfg.dataset_name=klue dataset.cfg.dataset_config_name=nli backbone.pretrained_model_name_or_path=klue/bert-base trainer.gpus=1 trainer.max_epochs=10 training.batch_size=64 training.lr=5e-05 dataset.cfg.max_length=64
```

> 현재 `lightning-transformers`에서는 text_classification 시 data에서 `idx`와 `label`을 제외하고 첫번째 field와 두번째 field를 label을 예측하는 데 사용하고 있습니다.([https://github.com/PyTorchLightning/lightning-transformers/blob/master/lightning_transformers/task/nlp/text_classification/data.py#L59](https://github.com/PyTorchLightning/lightning-transformers/blob/master/lightning_transformers/task/nlp/text_classification/data.py#L59) 참조) 따라서 위 코드로 실행 시 데이터셋을 재정의하거나 해당 라인을 아래와 같이 수정한 후 실행하면 됩니다.

```
zip(example_batch[input_feature_fields[1]], example_batch[input_feature_fields[2]])
```

다음 편으로는 `ligtning-transformer`으로 특정 task를 해결하기 위해 `tokenizer`, `dataset`, `model` 등을 직접 정의하는 방법에 대하여 알아보겠습니다.

----

본 블로그는 `hydra-core==1.1.0rc1`, `lightning-transformers==0.1` 버전을 기준으로 작성되었습니다.

## References
- [https://github.com/PyTorchLightning/lightning-transformers](https://github.com/PyTorchLightning/lightning-transformers)
- [https://github.com/PyTorchLightning/lightning-transformers/blob/master/docs/source/structure/conf.rst](https://github.com/PyTorchLightning/lightning-transformers/blob/master/docs/source/structure/conf.rst)
- [https://hydra.cc/docs/next/advanced/override_grammar/basic/](https://hydra.cc/docs/next/advanced/override_grammar/basic/)
- [https://klue-benchmark.com/](https://klue-benchmark.com/)