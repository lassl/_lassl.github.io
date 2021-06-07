---
layout: post
title: Masked Language Modeling with lightning-transformers
author: minho.ryu
categories: [implementation]
tags: [lightning-transformers, mlm]
---
<img src="https://user-images.githubusercontent.com/19511788/118396034-8b12ba80-b688-11eb-8d2e-c6adef157ce8.png">
<center> Masked Language Modeling <a href="http://jalammar.github.io/illustrated-bert/" target="_blank">(Photo from illustrated-bert)</a></center>

- lightning transformers 라이브러리를 활용하여 Masked Language Modeling을 구현한 방법을 설명한 글입니다.

<br />

---
### Lightning Transformers
lightning-transformers 라이브러리는 [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Transformers](https://github.com/huggingface/transformers), 그리고 [Hydra](https://github.com/facebookresearch/hydra) 를 이용하여 작성되었습니다. 이 글에서는 hydra의 작동방식과 pytorch-lightning 사용법에 대해 알고있다고 가정하도록 하겠습니다.

lightning-transformers 의 코드 구성은 아래와 같습니다.

    .
    ├── conf                            # Config files for hydra
    │   ├── dataset/nlp                 # dataset config for each nlp task
    │   │   ├── language_modeling       
    │   │   ├── ...   
    │   │   └── translation   
    │   ├── ...
    │   ├── task/nlp                    # model config for each nlp task
    │   │   ├── language_modeling.yaml   
    │   │   ├── ...   
    │   │   └── translation.yaml   
    │   └── ...
    ├── lightning_transformers          # Source files
    │   ├── cli                     
    │   ├── core                        # backbone classes using pytorch-lightning
    │   └── task/nlp                
    │       ├── language_modeling
    │       │   ├── config.py           # Task-specific config class
    │       │   ├── data.py             # Task-specific data class
    │       │   └── model.py            # Task-specific model class
    │       ├── ...
    │       └── translation
    ├── ...                         
    ├── predict.py                      # Script for prediction
    └── train.py                        # Script for training

<br />

---
### Language Modeling in lightning-transformers
Masked Language Modeling 구현 방법에 앞서 Language Modeling이 어떻게 구현되어 있는지 살펴봅시다. 사실 LM과 MLM의 구현방식은 크게 다르지 않으므로 이를 대부분 참조하였습니다.
먼저 위 코드 구성에서 lightning_transformers 폴더 아래 있는 task/nlp/language_modeling 안의 파일들을 보면,

```python
""" language_modeling/config.py """

from dataclasses import dataclass
from lightning_transformers.core.nlp import HFTransformerDataConfig


@dataclass
class LanguageModelingDataConfig(HFTransformerDataConfig):
    block_size: int = 128
```
config.py 파일에서는 Language Modeling 데이터 모듈에서 사용할 config를 정의하였습니다. LanguageModelingDataConfig는 HFTransformerDataConfig를 상속받았는데, 
HFTransformerDataConfig에는 dataset_name, padding, truncation 등의 transformers 라이브러리를 사용할 때 task-dependent 하게 공통으로 사용되는 argument들이 정의되어 있습니다.  

```python
""" language_modeling/data.py """

from functools import partial
from typing import Callable, Optional, Union

from datasets import Dataset
from pytorch_lightning import _logger as log
from transformers import default_data_collator, PreTrainedTokenizerBase

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.language_modeling.config import LanguageModelingDataConfig

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

    @property
    def effective_block_size(self) -> int:
        if self.cfg.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                log.warn(
                    f"The tokenizer picked seems to have a very large `model_max_length` "
                    f"({self.tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing dataset.cfg.block_size=x."
                )
            block_size = 1024
        else:
            if self.cfg.block_size > self.tokenizer.model_max_length:
                log.warn(
                    f"The block_size passed ({self.cfg.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.cfg.block_size, self.tokenizer.model_max_length)
        return block_size

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[PreTrainedTokenizerBase],
        text_column_name: str = None,
    ):
        return tokenizer(examples[text_column_name])

    @staticmethod
    def convert_to_features(examples, block_size: int, **kwargs):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator
```
data.py 파일에서는 데이터를 훈련 및 예측에 사용할 수 있는 형태로 변환하는 process_data 함수가 정의되어있습니다. 
tokenize_function 은 주어진 text 를 먼저 tokenizer 를 통해 input_ids (, token_type_ids, attention_mask) 로 변환해줍니다.
그 다음 convert_to_features 에서는 language modeling 을 할 token 의 갯수를 주어진 block_size 만큼 늘려주기 위해 모든 example text 를 
이어 붙인 다음 block_size 만큼씩 쪼개주는 작업을 진행하고, labels 은 input_ids 을 복사하여 넣어줍니다.

```python
""" language_modeling/model.py """

from lightning_transformers.core.nlp import HFTransformer

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
model.py 파일에서는 transformers 와 pytorch-lightning 라이브러리의 모델 구조를 연결하는 역할을 해줍니다.
이렇게 세 가지 파일을 살펴보았는데 코드를 천천히 보시면 사실 task 관련한 대부분의 구현이 transformers 에 되어 있기 때문에 어렵지 않게 사용하실 수 있음을 볼 수 있습니다.
그럼 Masked Language Modeling 을 구현한 방법으로 넘어가도록 하겠습니다.

<br />

---
### Masked Language Modeling Implementation
위 Language Modeling (LM) 관련 scripts 들을 이해하셨다면 Masked Language Modeling (MLM) 을 구현하는 것은 크게 어렵지 않습니다.
LM 과 마찬가지로 task/nlp 아래 masked_language_modeling 폴더를 만든 뒤 그 안에 config.py, data.py, model.py 스크립트들을 만들었습니다. 
먼저 MLM 의 model.py 의 경우 LM 의 model.py 파일을 그대로 사용하되 downstram_model_type 의 default 값을 transformers.AutoModelForCausalLM 이 아닌
transformers.AutoModelForMaskedLM 으로 바꿔주었습니다. 감사하게도 transformers 라이브러리에서 모델링 부분이 모두 구현되어 있어 가져다 쓰기만 하면 됩니다.

```python
""" masked_language_modeling/model.py """

from lightning_transformers.core.nlp import HFTransformer

class MaskedLanguageModelingTransformer(HFTransformer):
    """
    Defines ``LightningModule`` for the Masked Language Modeling Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load. (default ``transformers.AutoModelForMaskedLM``)
        **kwargs: :class:`lightning_transformers.core.nlp.HFTransformer` arguments.
    """

    def __init__(self, *args, downstream_model_type: str = 'transformers.AutoModelForMaskedLM', **kwargs) -> None:
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
        return "masked-text-generation"
```

두 번째로 볼 부분은 data.py 파일입니다. 여기서도 사실 LM 의 data.py 파일과 크게 다르지 않습니다. 다만 LM 의 경우에는 최대한 길이를 길게 학습하는 것이 유리한 반면 MLM 에서는 line_by_line 으로 학습하는 것이 필요할 때도 있습니다. 이는 LM 은 다음 단어만을 예측하는 것이므로 짧은 것부터 긴 것까지 모두 학습이 되지만 MLM 은 주어진 토큰들을 모두 읽고 예측하는 것이기 때문에 주어진 문장의 길이에 따라 학습이 달라지기 때문입니다. 따라서 line_by_line 으로 학습데이터를 형성할 것인지 아니면 LM 과 마찬가지로 모든 text 를 이어 붙인다음 주어진 max_length 로 쪼갤 것인지 선택할 수 있도록 옵션을 줘야합니다. 추가로 Whole Word Masking 기능을 사용할 수 있도록 옵션으로 추가하였습니다.

```python
""" masked_language_modeling/data.py """
from functools import partial
from typing import Callable, Optional, Union

from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask, PreTrainedTokenizerBase

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.masked_language_modeling.config import MaskedLanguageModelingDataConfig

class MaskedLanguageModelingDataModule(HFDataModule):
    """
    Defines ``LightningDataModule`` for Language Modeling Datasets.

    Args:
        *args: ``HFDataModule`` specific arguments.
        cfg: Contains data specific parameters when processing/loading the dataset
            (Default ``MaskedLanguageModelingDataConfig``)
        **kwargs: ``HFDataModule`` specific arguments.
    """
    cfg: MaskedLanguageModelingDataConfig

    def __init__(
        self, *args, cfg: MaskedLanguageModelingDataConfig = MaskedLanguageModelingDataConfig(), **kwargs
    ) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        column_names = dataset["train" if stage == "fit" else "validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        tokenize_function = partial(
            self.tokenize_function,
            tokenizer=self.tokenizer,
            text_column_name=text_column_name,
            line_by_line=self.cfg.line_by_line,
            padding=self.cfg.padding,
            max_length=self.cfg.max_length,
        )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        if not self.cfg.line_by_line:
            convert_to_features = partial(
                self.convert_to_features, 
                max_seq_length=self.cfg.max_length,
            )

            dataset = dataset.map(
                convert_to_features,
                batched=True,
                num_proc=self.cfg.preprocessing_num_workers,
                load_from_cache_file=self.cfg.load_from_cache_file,
            )

        return dataset

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer: Union[PreTrainedTokenizerBase],
        text_column_name: str = None,
        line_by_line: bool = False,
        padding: Union[str, bool] = "max_length",
        max_length: int = 128,
    ):
        if line_by_line:
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
 
    @staticmethod      
    def convert_to_features(examples, max_seq_length: int, **kwargs):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    @property
    def collate_fn(self) -> Callable:
        if self.cfg.wwm:
            return DataCollatorForWholeWordMask(self.tokenizer, mlm_probability=self.cfg.mlm_probability)
        else:
            return DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=self.cfg.mlm_probability)

```

마지막으로 config.py 파일입니다. 이 파일에서는 MLM 에서 쓰인 argument 를 추가해주면 됩니다. 위 model.py 스크립트에서 추가로 사용한 argument 는 mlm_probability, line_by_line, wwm 세 가지가 있습니다. 따라서 이 세 가지 옵션을 아래와 같이 추가해주면 됩니다.

```python
""" masked_language_modeling/config.py """

from dataclasses import dataclass, field
from lightning_transformers.core.nlp import HFTransformerDataConfig

@dataclass
class MaskedLanguageModelingDataConfig(HFTransformerDataConfig):
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    wwm: bool = field()
        default=False,
        metadata={"help": "Whether or not use whole word masking"},
    )
```

자 이렇게 lightning_transformers/task/nlp 에 필요한 파일들을 모두 추가하신 다음에 마지막으로 conf 폴더 아래에 필요한 yaml 파일들을 추가해주면 됩니다. 이 부분은 LM 과 관련된 파일을 보시면 충분히 이해하실 수 있기 때문에 생략하도록 하겠습니다.
지금까지 lightning-transformers 를 이용해서 MLM 을 구현한 방법에 대해서 살펴보았습니다. 이 번 구현을 진행하면서 이 라이브러리를 구성하고 있는 세 가지 라이브러리에 대한 이해가 있다면 아주 쉽게 task 를 추가하거나 활용하여 수정할 수 있다는 것을 알 수 있었습니다. 정말 하루가 다르게 점점 더 사용 뿐만 아니라 개발 및 연구도 쉬워지는 것 같네요!

p.s. 현 포스트에서 소개드린 masked_language_modeling 구현을 PytorchLightning/lightning-transformers repo 에 <a href="https://github.com/PyTorchLightning/lightning-transformers/pull/173">PR</a>을 날려 merge 되어 contributor가 되었네요! 도움 주신 모든 팀원 여러분들께 감사드립니다 :)
