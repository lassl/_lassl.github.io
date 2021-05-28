---
layout: post
title: Causal Language Modeling with lightning-transformers
author: boseop.kim
categories: [implementation]
tags: [lightning-transformers, clm]
---

- https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/language_modeling.html 를 예시로 동작 방식을 조금 상세하게 argument의 전달을 상세히 쓴 글입니다.
<br/>

---
## Introduction
lightning-transformers를 활용하여, Causal Language Modeling을 구현하는 방법은 [LANGUAGE MODELING](https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/language_modeling.html)에 간략히 소개되어있고, 본 포스트에서는 위의 링크에서 간략히 소개된 내용들이 lightning-transformers 라이브러리 내부에서 어떤 코드에 해당하는 지를 분석하였습니다.

## Datasets
실제로 huggingface datasets 라이브러리를 lightning-transformers에서 사용하고있기때문에, https://huggingface.co/datasets 에 등록되어있는 데이터라면 매우 쉽게 사용할 수 있습니다. 예를 들어 아래의 command로 lightning-transformers 라이브러리를 사용한다고 합시다.

```bash
python train.py task=nlp/language_modeling dataset=nlp/language_modeling/wikitext
```

위의 command에서 `task=nlp/langague_modeling`은 `conf` directory 하위에 존재하는 `conf/task/nlp/language_modeling.yaml`과 `conf/dataset/nlp/language_modeling/wikitext.yaml`를 `train.py`에 argument로 전달하는 것을 의미합니다. (전체 lightning-transformers 코드 구성에 관한 설명은 [Masked Language Modeling with lightning-transformers
](https://lassl.github.io/implementation/lightning-transformers-mlm.html)를 참고해주세요!)

```bash
├── conf
│   ├── __init__.py
│   ├── backbone
│   │   └── nlp
│   │       ├── default.yaml
│   │       └── seq2seq.yaml
│   ├── config.yaml
│   ├── dataset
│   │   ├── default.yaml
│   │   └── nlp
│   │       ├── default.yaml
│   │       ├── language_modeling
│   │       │   ├── default.yaml
│   │       │   └── wikitext.yaml
...
│   ├── task
│   │   ├── default.yaml
│   │   └── nlp
│   │       ├── default.yaml
│   │       ├── language_modeling.yaml
│   │       ├── multiple_choice.yaml
│   │       ├── question_answering.yaml
│   │       ├── summarization.yaml
│   │       ├── text_classification.yaml
│   │       ├── token_classification.yaml
│   │       └── translation.yaml
...
│   ├── tokenizer
│   │   └── autotokenizer.yaml
...
```

각각의 yaml 파일을 분석해보겠습니다. 먼저 `conf/dataset/nlp/language_modeling/wikitext.yaml` 아래와 같으며, lighting-transformers가 사용하고있는 [hydra](https://github.com/facebookresearch/hydra)에서 지원하는 문법이 적용되어 있음을 확인할 수 있습니다. 특히 아래의 yaml 파일에서 `__target__`에 python class 명을 적으면 [hydra](https://github.com/facebookresearch/hydra)가 알아서 instantiate를 해주는 기능입니다. (자세한 내용은 [Instantiating objects with Hydra
](https://hydra.cc/docs/patterns/instantiate_objects/overview)를 참고해주세요!)

- `conf/dataset/nlp/language_modeling/wikitext.yaml`

    ```yaml
    # @package dataset 
    defaults:
    - nlp/default
    _target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingDataModule
    cfg:
    dataset_name: wikitext
    dataset_config_name: wikitext-2-raw-v1
    block_size: 512
    ```

실제로 위의 yaml 파일에 정의된대로 `cfg` 하위에 정의된 `dataset_name`, `dataset_config_name`, `block_size` key에 대응되는 value로 `LanguageModelingDataModule` class를 instantiate하게 됩니다.

- `lightning_transformers.task.nlp.language_modeling.LanguageModelingDataModule`

    ```python
    class LanguageModelingDataModule(HFDataModule):
        """
        Defines ``LightningDataModule`` for Language Modeling Datasets.

        Args:
            *args: ``HFDataModule`` specific arguments.
            cfg: Contains data specific parameters when processing/loading the dataset
                (Default ``LanguageModelingDataConfig``)
            **kwargs: ``HFDataModule`` specific arguments.
        """
        cfg: LanguageModelingDataConfig

        def __init__(self, *args, cfg: LanguageModelingDataConfig = LanguageModelingDataConfig(), **kwargs) -> None:
            super().__init__(*args, cfg=cfg, **kwargs)

        def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
            column_names = dataset["train" if stage == "fit" else "validation"].column_names
            text_column_name = "text" if "text" in column_names else column_names[0]

            tokenize_function = partial(self.tokenize_function, tokenizer=self.tokenizer, text_column_name=text_column_name)

            dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=self.cfg.load_from_cache_file,
            )

            convert_to_features = partial(self.convert_to_features, block_size=self.effective_block_size)

            dataset = dataset.map(
                convert_to_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                load_from_cache_file=self.cfg.load_from_cache_file,
            )

            return dataset
    ...
    ```

`LanguageModelingDataModule`이 상속하는 class인 `HFDataModule`의 `load_dataset` method에서 cfg 하위에 정의된 value들을 사용합니다.

- `lightning_transformers.core.nlp.data.HFDataModule`

    ```python
    class HFDataModule(TokenizerDataModule):
        """
        Base ``LightningDataModule`` for HuggingFace Datasets. Provides helper functions and boilerplate logic
        to load/process datasets.

        Args:
            tokenizer: ``PreTrainedTokenizerBase`` for tokenizing data.
            cfg: Contains data specific parameters when processing/loading the dataset (Default ``HFTransformerDataConfig``)
        """
        cfg: HFTransformerDataConfig
        tokenizer: PreTrainedTokenizerBase

        ...

        def load_dataset(self) -> Dataset:
            # Allow custom data files when loading the dataset
            data_files = {}
            if self.cfg.train_file is not None:
                data_files["train"] = self.cfg.train_file
            if self.cfg.validation_file is not None:
                data_files["validation"] = self.cfg.validation_file
            if self.cfg.test_file is not None:
                data_files["test_file"] = self.cfg.test_file

            data_files = data_files if data_files else None
            if self.cfg.dataset_name is not None:
                # Download and load the Huggingface dataset.
                return load_dataset(
                    path=self.cfg.dataset_name,
                    name=self.cfg.dataset_config_name,
                    cache_dir=self.cfg.cache_dir,
                    data_files=data_files
                )
        ...
    ```

또 다른 yaml 파일인 `conf/task/nlp/language_modeling.yaml` 아래와 같으며 마찬가지로 `__target__`에 해당하는 `lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer`를 `downstream_model_type` key의 value인 `transformers.AutoModelForCausalLM` 전달하여 instantiate 하게됩니다.

- `conf/task/nlp/language_modeling.yaml`

    ```yaml
    # @package task
    defaults:
    - nlp/default
    - override /dataset@_group_: nlp/language_modeling/default
    _target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer
    downstream_model_type: transformers.AutoModelForCausalLM
    ```

- `lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer`

    ```python
    class LanguageModelingTransformer(HFTransformer):
        """
        Defines ``LightningModule`` for the Language Modeling Task.

        Args:
            *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
            downstream_model_type: Downstream HuggingFace AutoModel to load. (default ``transformers.AutoModelForCausalLM``)
            **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        """

        def __init__(self, *args, downstream_model_type: str = 'transformers.AutoModelForCausalLM', **kwargs) -> None:
            super().__init__(downstream_model_type, *args, **kwargs)

        def on_fit_start(self):
            tokenizer_length = len(self.tokenizer)
            self.model.resize_token_embeddings(tokenizer_length)

        def _step(self, batch, batch_idx):
            outputs = self.model(**batch)
            loss = outputs[0]
            return loss

        def training_step(self, batch, batch_idx):
            loss = self._step(batch, batch_idx)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            loss = self._step(batch, batch_idx)
            self.log("val_loss", loss, sync_dist=True)

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            loss = self._step(batch, batch_idx)
            self.log("test_loss", loss, sync_dist=True)

        @property
        def hf_pipeline_task(self) -> str:
            return "text-generation"
    ```

예시로든 command에 아래와 같이 `backbone.pretrained_model_name_or_path=gpt2`를 argument로 전달하면 `conf/backbone/nlp/default.yaml`에서 `pretrained_model_name_or_path` key의 value를 `gpt2`로 설정하여, `train.py`에 argument로 전달하는 것과 같습니다.

```bash
python train.py task=nlp/language_modeling dataset=nlp/language_modeling/wikitext backbone.pretrained_model_name_or_path=gpt2
```

- `conf/backbone/nlp/default.yaml`

    ```yaml
    # @package backbone
    defaults:
    - /tokenizer@_group_: autotokenizer # use AutoTokenizer by default
    pretrained_model_name_or_path: gpt2
    ```

이와 같이 전달되면 lightning-transformers는 중간과정들을 거쳐 결과적으로 위의 `LanguageModelTransformer`가 상속하는 `HFTransformer`의 `__init__` method로 instantiate할 때, `backbone.pretrained_model_name_or_path`으로 전달되어 value에 맞게 class를 instantiate하게되는 것을 확인할 수 있습니다.

- `lightning_transformers.core.nlp.model.HTTransformer`

    ```python
    class HFTransformer(TaskTransformer):
        """
        Base class for task specific transformers, wrapping pre-trained language models for downstream tasks.
        The API is built on top of AutoModel and AutoConfig, provided by HuggingFace.

        see: https://huggingface.co/transformers/model_doc/auto.html

        Args:
            downstream_model_type: The AutoModel downstream model type.
                See https://huggingface.co/transformers/model_doc/auto.html
            backbone: Config containing backbone specific arguments.
            optimizer: Config containing optimizer specific arguments.
            scheduler: Config containing scheduler specific arguments.
            instantiator: Used to instantiate objects (when using Hydra).
                If Hydra is not being used the instantiator is not required,
                and functions that use instantiation such as ``configure_optimizers`` has been overridden.
            tokenizer: The pre-trained tokenizer.
            pipeline_kwargs: Arguments required for the HuggingFace inference pipeline class.
            **model_data_kwargs: Arguments passed from the data module to the class.
        """

        def __init__(
            self,
            downstream_model_type: str,
            backbone: HFBackboneConfig,
            optimizer: OptimizerConfig = OptimizerConfig(),
            scheduler: SchedulerConfig = SchedulerConfig(),
            instantiator: Optional[Instantiator] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            pipeline_kwargs: Optional[dict] = None,
            **model_data_kwargs,
        ) -> None:
            self.save_hyperparameters()
            model_cls: Type["AutoModel"] = get_class(downstream_model_type)
            model = model_cls.from_pretrained(backbone.pretrained_model_name_or_path, **model_data_kwargs)
            super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, instantiator=instantiator)
            self._tokenizer = tokenizer  # necessary for hf_pipeline
            self._hf_pipeline = None
            self._hf_pipeline_kwargs = pipeline_kwargs or {}
    ...
    ```

또한 `conf/backbone/nlp/default.yaml`에서 `/tokenizer@_group_: autotokenizer`에 대응되는 yaml 파일인 `conf/tokenizer/autotokenizer.yaml`을 확인해보면, `conf/backbone/nlp/default.yaml`의 `pretrained_model_name_or_path` key의 value인 `gpt2`가 `conf/tokenizer/autotokenizer.yaml`의 `pretrained_model_name_or_path` key의 value로 전달되어, 알맞게 tokenizer도 instantiate하는 것을 확인할 수 있습니다.

- `conf/tokenizer/autotokenizer.yaml`

    ```yaml
    # @package tokenizer
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: ${backbone.pretrained_model_name_or_path}
    use_fast: true
    ```

## Language Modeling Using Your Own Files
huggingface datasets library로 바로 이용할 수 있는 corpus가 아닌 custom corpus를 사용해야하는 경우에는 line by line 형식으로 준비하여 활용할 수 있습니다.

```bash
text,
this is the first sentence,
this is the second sentence,
```

위와 같이 corpus가 line by line 형식으로 준비되면 아래와 같이 `dataset.cfg.train_file`, `dataset.cfg.validation_file`에 그 경로를 전달하여 활용할 수 있습니다.

```bash
python train.py task=nlp/language_modeling dataset.cfg.train_file=train.csv dataset.cfg.validation_file=valid.csv
```

위와 같이 전달하면 `conf/dataset/nlp/default.yaml`의 `cfg` 하위에 정의된 key인 `train_fle`, `validation_file`에 경로들을 value로 기입하는 것과 같습니다. 결과적으로는 관련있는 yaml 파일들이 아래와 같은 상태로 `train.py`에 전달되는 것과 같습니다.

- `conf/task/nlp/language_modeling.yaml`

    ```yaml
    # @package task
    defaults:
    - nlp/default
    - override /dataset@_group_: nlp/language_modeling/default
    _target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingTransformer
    downstream_model_type: transformers.AutoModelForCausalLM
    ```

- `conf/dataset/nlp/language_modeling/default.yaml`

    ```yaml
    # @package dataset
    defaults:
    - nlp/default
    _target_: lightning_transformers.task.nlp.language_modeling.LanguageModelingDataModule
    cfg:
    block_size: null
    ```

- `conf/dataset/nlp/default.yaml`

    ```yaml
    # @package dataset
    defaults:
    - /dataset/default
    _target_: lightning_transformers.core.nlp.HFDataModule
    cfg:
    dataset_name: null
    dataset_config_name: null
    train_file: train.csv
    validation_file: valid.csv
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
