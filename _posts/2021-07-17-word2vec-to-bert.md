---
layout: post
title: Word2Vec에서 BERT까지의 NLP의 역사
author: minho.ryu
categories: [history]
tags: [word2vec, rnn, bert, attention mechanism, seq2seq]
---

- 본 글은 word2vec의 등장부터 BERT까지 자연어처리가 발전해온 역사를 짚어보는 글입니다.
<br/>

---
## 1. NNLM to Word2Vec
2013년, 처음 word2vec 알고리즘이 나왔을 때, 엄청난 센세이션을 일으켰습니다. 벌써 8년 전이지만 BERT가 나왔을 때 만큼이나 임팩트를 줬던 것 같습니다.
<img src="https://user-images.githubusercontent.com/19511788/120918095-96885d00-c6ed-11eb-970c-c8fc39d70457.png" width="700" class="center">
<center> Figure 1. distributed representation by word2vec </center>

이전에도 단어를 one-hot encoding이 아닌 distributed representation으로 표현하려는 시도는 있었지만 효율적이지 못한 방법으로 대규모 코퍼스에 대해 학습을 진행하기 어려웠습니다.
word2vec 이전 방법들을 간단하게 살펴보자면, 대표적으로 Neural Network Language Model (NNLM), Recurrent Neural Network Language Model (RNNLM)이 있습니다. 
NNLM은 Feed Forward 신경망 모델로써 앞의 n개의 단어를 받아 다음 단어를 예측하는 문제를 푸는 방법으로 학습이 이뤄졌고, RNNLM은 RNN의 속성처럼 차례차례 다음 단어를 예측하는 방법으로 학습을 하였습니다. 그러나 두 방법 모두 예측을 위해 vocab의 갯수에 비례하는 계산량을 요구하는 softmax 함수를 이용함으로써 속도에서 큰 장애물이 있었습니다.

<img src="https://user-images.githubusercontent.com/19511788/122235966-120db980-cef9-11eb-8359-c401aab554fd.png" width="700" class="center">
<center> Figure 2. Neural Network Language Model (NNLM) </center>

word2vec을 제안한 [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) 논문에서는 Continuous Bag of Words (CBoW) 방법과 Skip-Gram 방법을 제시하였습니다. word2vec의 성공은 대용량의 데이터를 혁신적으로 빠른 연산 방법을 제안함으로써 가능했다고 볼 수 있습니다. 기존 NNLM의 경우, projection layerd의 dimension $P$, $N$개의 단어와 $V$의 vocabulary수, hidden layer의 dimension $H$를 이용해 계산복잡도를 구하면 아래와 같습니다.
<center>$Q = N \times P + N \times P \times H + H \times V$</center>

보통 $V >> P,H$ 이므로, 위 수식에서 dominating term은 $N \times P \times H$입니다. 반면, word2vec에서는 연산의 bottleneck인 hidden layer를 제거하였고 두 번째 heavy term인 $H \times V$를 줄이기 위해 Huffman 이진트리를 응용한 Hierarchical Softmax를 이용하여 $O(V)$의 계산복잡도를 $O(log(V))$으로 줄였습니다. 따라서 word2vec의 계산복잡도는 아래와 같습니다.
<center>$Q = N \times P + H \times log(V)$</center>

 또한 두 번째로 나왔던 논문인 [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)에서는 Skip-Gram 모델에 Hierarchical Softmax를 대체하는 Negative Sampling (NEG) 기법과 자주 나오는 단어를 적게 뽑는 Unigram Subsampling 기법을 추가로 사용하여 학습 효율을 더욱 끌어올렸습니다. NEG의 경우 최근에 많이 사용되는 Contrastive Learning 과 거의 동일한 개념이며, 목적 함수는 아래와 같습니다.
<center>$\displaystyle log(\sigma({v^{\prime}_{w_O}}^{T}{v^{\space}_{w_I}})) + \sum_{i=1}^{k}\mathbb{E}_{w_i \sim P_n(w)}[log(-\sigma({v^{\prime}_{w_i}}^{T}{v^{\space}_{w_I}}))]$</center>

위 목적함수를 Maximize하는 방향으로 훈련시키면, 수식에서 볼 수 있듯이 실제 주변 단어로 이루어진 pair에 대해서는 두 벡터가 유사하도록 학습하고, k개의 가짜 pair (negative sample)에 대해서는 두 벡터가 멀어지도록 학습하게 됩니다. 이 NEG 방법은 기존 softmax 함수를 통해 target 단어를 직접 예측하는 방법을 Logistic Regression을 이용한 이진 문제로 변환함으로써 시간복잡도를 낮추면서 학습 속도를 크게 개선하였습니다. softmax 방법은 target 단어는 가깝게 그외에 모든 단어 (V)에 대해 negative signal을 주는 것이라면, NEG 방법은 k개의 단어에 대해서는 negative signal을 주는 방식으로 훨씬 효율적인 방법이라고 볼 수 있습니다. negative sample의 경우 unigram distribution $U_n(w)$의 3/4 제곱에서 추출하는 것이 효과적인 것으로 알려졌으며 k값은 5-20정도가 유용합니다.

<img src="https://user-images.githubusercontent.com/19511788/122509555-f280a900-d03e-11eb-8f54-90105037c532.png" width="700" class="center">
<center> Figure 3. Skip-Gram Model </center>

Unigram Subsampling 기법은 자주 등장하는 단어를 적게 sampling 함으로써 자주 등장하지 않는 단어에 대한 학습이 더 잘 일어나도록 하는 방법입니다. threshold $t$값보다 빈도 비율이 큰 단어에 대해 아래 수식을 적용합니다.

<center>$\displaystyle P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}$</center>

$f(w_i)$는 단어 $w_i$의 빈도 비율를 나타내며, $t$값으로 보통 $10^{-5}$정도의 값을 사용합니다. 이 방법들을 이용하여 단어 임베딩 성능과 학습 속도가 비약적으로 좋아졌고 자연어처리에 딥러닝을 적용하는 큰 발판으로 작용하였습니다. 


---
## 2. Attention Mechanism
word2vec이 공개된 이후 NLP를 위한 딥러닝 모델 구조에도 많은 진화가 나타났습니다. one-hot encoding이 아닌 word2vec (또는 Glove)로 사전학습된 distributed representation의 이용이 당연시 되었고, RNN, LSTM을 이용하여 이전에는 풀기 어려웠던 여러 NLP문제들을 개선해 나갔고, CNN으로도 NLP 문제에서 좋은 성능을 낼 수 있다는 것을 보여주는 논문([Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882))도 나왔습니다. 또한 Sequence to Sequence (seq2seq) 또는 Encoder-Decoder라고 불리는 두 개의 RNN모델을 이용한 구조를 이용하여 번역 문제를 개선해 나가고 있었습니다. 그러던 어느 날, NLP에 다시 한 번 큰 변혁을 가져온 논문이 공개되었습니다. 그 논문은 바로 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)으로, 이 논문에서 기존 seq2seq 모델은 Information Bottleneck 문제가 발생한다고 주장하였습니다. 풀어말하면, 문장의 길이에 상관없이 하나의 고정된 길이의 벡터가 주어진 문장의 모든 정보를 담기엔 부족하다고 주장하였습니다.

<img src="https://user-images.githubusercontent.com/19511788/125164980-e6b47c80-e1cf-11eb-9118-9785f17d95fd.png" width="900" class="center">
<center> Figure 4. Information Bottleneck of Encoder Decoder RNNs </center>

이 문제를 해결하기 위해서, 이제는 너무나 당연하지만 아주 중요한 **Attention Mechanism**이라는 개념이 도입되었습니다. 이 개념을 설명하기 앞서 RNN Encoder-Decoder 구조에 대해서 간략하게 설명하겠습니다. 이 구조에서 Encoder RNN은 input sequece $\mathbf{x}=(x_1,...,x_{T_x})$를 읽고 고정된 길이를 가진 context 벡터 $c$로 변환합니다. 이를 수식으로 표현하면 다음과 같습니다.

<center>$c=q(\{h_1,...,h_{T_x}\})$,</center>
<center>where $h_t=f(x_t,h_{t-1})$</center>

$h_t$는 $t$시점에서의 hidden state를 의미하며, $c$는 주로 Encoder RNN의 마지막 hidden state를 의미합니다. 처음 Encoder-Decoder 구조를 제안한 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)에서는 $f$ 또는 $q$의 함수로 LSTM을 사용하였습니다. 그 다음 Decoder RNN은 벡터 $c$와 이전에 예측 생성한 단어를 기반으로 단어를 차례차례 생성합니다.
<center>$\displaystyle p(\mathbf{y})=\prod_{t=1}^T{p(y_t | \{y_1, ..., y_{t-1}\}, c)}$</center>
<center>$p(y_t | \{y_1, ..., y_{t-1}\}, c) = g(y_{t-1}, s_t, c)$</center>

$s_t$는 Decoder RNN의 hidden state를 나타내며, 함수 $g$는 보통 nonlinear multi-layered network을 이용합니다. 여기에 **Attention Mechanism**은 각 단어를 생성할 때 동일한 $c$를 사용하는 것이 아니라 매 차례마다 다른 $c_i$를 생성하여 사용합니다.

<center>$p(y_i|y_1, ..., y_{i-1})=g(y_{i-1},s_i,c_i), s_i=f(s_{i-1},y_{i-1},c_i)$</center>

context 벡터 $c_i$는 encoder hidden state들의 weighted sum으로 계산되며 각 weight는 encoder hidden state와 decoder hidden state사이의 align 함수와 softmax 함수를 통해 구할 수 있습니다.

<center>$\displaystyle c_i=\sum_{j=1}^{T_x}{\alpha_{ij}h_j}$,</center>
<center>$\displaystyle \alpha_{ij}=\frac{exp(e_{ij})}{\sum_{k=1}^{T_x}{exp(e_{ik})}}$,</center>
<center style="margin-top: 10px;">where $e_{ij}=a(s_{i-1}, h_j)$</center>

위와 같은 **Attention Mechanisim**을 통해서 문장의 길이가 길어질 때 성능이 급격하게 줄어드는 Information Bottleneck 문제를 크게 개선했을 뿐만 아니라 alignment를 학습함으로써 모델이 어떻게 작동하는 지에 대해 해석할 수 있는 방법을 제공하였습니다.

<img src="https://user-images.githubusercontent.com/19511788/125179976-b0a6e500-e22f-11eb-8c32-b7411e09a9e8.png" width="700" class="center">
<center> Figure 5. Effects of Attention Mechanism </center>

<img src="https://user-images.githubusercontent.com/19511788/125179980-c1575b00-e22f-11eb-9b5e-c2e6d74087de.png" width="700" class="center">
<center> Figure 6. Sample alignments from English to French </center>


---
## 3. Seq2Seq to Transformer
Attention Mechanism 이 소개된 이후 많은 variants들이 나왔지만 여전히 LSTM 기반의 모델을 사용하는 것이 일반적이었습니다. 하지만 RNN계열 모델의 경우 Sequence의 각 토큰을 한 번에 한 개씩 밖에 처리하지 못하는 구조적인 결함을 가지고 있었고 GPU의 병렬처리 기능을 효과적으로 활용하지 못하는 한계를 지니고 있어 학습이 오래걸린다는 단점이 여전히 존재하였습니다. 또한 [Convolution Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)같이 병렬처리에는 효과적인 방법들도 존재했지만 거리가 먼 토큰들 간의 dependency를 배우는데 어려움이 있었습니다. 이 때, 이 두 가지 문제를 동시에 해결한 방법을 제시하는 논문이 등장했는데, 그것이 바로 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 즉, **Transformer**입니다. **Transformer**는 Encoder-Decoder 구조를 사용하지만 Recurrence나 Convolution없이 오직 Attention만을 이용해서 global dependency를 학습합니다. 또한 Encoder와 Decoder 모두에서 각각 Self Attention을 통해 Input Token들 간 그리고 Output Token들 간의 Alignment 뿐만 아니라 동시에 Input Sequence와 Output Sequence 간의 Alignment까지 모두 학습함으로써 모델의 표현력을 향상시켰습니다. 모델의 구체적인 구조는 다음과 같습니다.

<img src="https://user-images.githubusercontent.com/19511788/125181888-40ef2500-e244-11eb-98e4-148f0eaf11b3.png" width="700px" class="center">
<center>Figure 7. Transformer Model Architecture</center>

모델 구조에 있어서 중요한 부분들을 살펴보자면, Position Encoding과 Multi-Head Attention이 있습니다. 먼저 Position Encoding에 대해 알아보면, **Transformer**모델은 recurrence나 convolution을 사용하고 있지 않기 때문에 모델 자체적으로는 토큰의 위치에 대한 정보를 처리할 수 없습니다. 토큰의 위치에 상관없이 inter dependency만을 구할 수 있을 뿐이죠. 따라서 위치 정보에 대해 별도로 주입해 줄 필요가 있었고 이에 제시된 방법이 Position Encoding 입니다. Position Encoding은 아주 간단하게 Input Sequence의 각 토큰에 대한 word embedding에 더하는 방식으로 주입할 수 있습니다. 두 번째로 Multi-Head Attention은 크게 두 가지로 나눠 Multi-Head와 Attention으로 나눌 수 있고 Attention은 어떤 Alignment 함수를 사용하였는지에 대한 것입니다. 본 논문에서는 Scaled Dot-Product Attention을 제안하였는데 이는 다음 수식과 같습니다.

<center>$\displaystyle Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$</center>
<img src="https://user-images.githubusercontent.com/19511788/125182693-2c625b00-e24b-11eb-98d1-d4fb89fa7372.png" style="width: 700px" class="center">
<center>Figure 8. Scaled Dot-Product Attention</center>

Multi-Head의 경우, 토큰 간의 하나의 Alignment를 학습하는 것이 아닌 다양한 Alignment를 학습하는 것으로 비슷한 연산량으로 더 많은 representation subspace를 학습할 수 있습니다.

<img src="https://user-images.githubusercontent.com/19511788/125182802-f96c9700-e24b-11eb-90d7-23bfae980e9c.png" style="width: 700px" class="center">
<center>Figure 9. Multi-Head Attention</center>

**Transformer**는 recurrence 제거로 인한 학습 효율의 개선과 함께 기존의 모델들을 압도하는 성능을 보여주었으며 그리고 word2vec 이후 NLP의 역사를 다시 한 번 급격하게 발전시키는 시발점이 됩니다.

---
## 4. Shallow to Deep Contextual Representation
다시 word2vec을 살펴보겠습니다. word2vec은 기존 one-hot encoding에서 성공적으로 distributed representation으로의 전환을 이루어 냈으나 여전히 한 가지 큰 문제점을 가지고 있었습니다. 그것은 단어의 의미가 문맥에 따라 다양하게 바뀔 수 있다는 점입니다. 각 단어에 대해 문맥과 상관없이 하나의 embedding을 가지는 기존 방식은 문맥에 따라 단어의 embedding을 바꿀 수 없다는 한계를 지녔습니다. 이러한 문제를 해결하기 위해서 다양한 방식이 시도되어 왔습니다. 먼저 [Semi-Supervised Sequence Learning](https://arxiv.org/abs/1511.01432)에서는 LSTM을 이용하여 Language Modeling (LM)과 Sequence Autoencoder방법을 통해 pretraining을 진행합니다. Sequence Autoencoder는 아래 그림에서 볼 수 있듯이 Input Sequence를 넣고 <eos> 토큰을 넣으면 Input Sequence를 그대로 출력하는 방식으로 학습을 하는 방법입니다.

<img src="https://user-images.githubusercontent.com/19511788/125183758-01c8d000-e254-11eb-8898-ddf79f766c2b.png" style="width: 700px" class="center">
<center>Figure 10. Sequence Autoencoder</center>

이와 같은 방식을 통해 각 토큰에 대한 embedding을 구할 수 있을 뿐만 아니라 Input Sequence 전체 정보를 처리하기 위한 pretraining 방법으로도 사용할 수 있었고 기존 Doc2Vec과 같은 방식보다도 더 좋은 성능을 보여주었습니다. 하지만 왼쪽에서 오른쪽 단방향으로 학습이 이루어진다는 점에서 양방향의 정보를 사용할 수 없다는 한계가 있습니다. 이러한 단점을 보완하기 위해서 [Deep Contextualized Word Representations](https://arxiv.org/abs/1802.05365) (ELMo)논문에서는 Deep bidirectional LSTM을 이용하여 양방향 LM을 학습하고 학습된 파라미터를 고정하여 pretrained embeddings로서 downstream task에 활용하였습니다.

<img src="https://user-images.githubusercontent.com/19511788/125185191-3e99c480-e25e-11eb-943d-176258b5cd7b.png" style="width: 700px" class="center">
<center>Figure 11. ELMo Strategy</center>

하지만 여전히 ELMo의 경우도 왼쪽에서 오른쪽 그리고 오른쪽에서 왼쪽으로 학습된 정보를 각각 활용할 뿐, 단어의 contextual representation을 구성하기 위해 해당 단어 양쪽의 단어를 한 번에 사용하지 못하는 한계가 존재했습니다. 이렇듯 더 나은 Deep Contexualized Representation을 향한 움직임이 한창인 와중 한편에서는 transformer라는 모델이 발표되었고, 이 모델은 이 움직임에 자연스럽게 활용되는데 그 첫 번째로 나왔던 것이 OpenAI GPT (a.k.a. GPT1)모델입니다. GPT1 모델은 transformer에서 Decoder 부분만을 이용한 Architecture를 가지고 있으며 LM을 통해 학습됩니다.

<img src="https://user-images.githubusercontent.com/19511788/125185506-262aa980-e260-11eb-9763-0d66eb5223a9.png" style="width: 700px" class="center">
<center>Figure 12. OpenAI GPT Architecture</center>

GPT1은 기존 LSTM 기반 모델들보다 대부분의 NLP 문제에 있어서 월등한 성능을 보여주었습니다. 그러나 여전히 같은 문제가 존재하였고 2018년 말, 이 문제를 극복하기 위해 BERT가 처음 세상에 공개되었습니다.

---
## 5. Bidirectional Encoder Representations from Transformers (BERT)
하나의 모델에 대해서 양방향으로 LM을 학습시키게 되면 왼쪽에서 오른쪽으로 학습하고나서 반대 방향으로 학습할 때, 이미 왼쪽에 있는 단어들을 모델이 봤었기 때문에 그 때 습득한 정보를 이용할 수 있습니다. 따라서 일반적인 CLM 방법으로는 양방향 정보를 동시에 이용하면서 "See themselves 문제", 소위 컨닝을 막을 수 없습니다. 이러한 문제를 해결하기 위해서 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 논문에서는 다른 학습 방법을 제안하였는데 그 방법이 바로 **Masked Language Modeling (MLM)** 방법입니다.

<img src="https://user-images.githubusercontent.com/19511788/125188116-9855bb00-e26d-11eb-8d82-420ec1d4807c.png" style="width: 700px" class="center">
<center>Figure 13. Masked Language Modeling with BERT</center>

BERT는 Transformer의 Encoder 부분을 이용한 Architecture를 가지고 있으며, MLM은 위 그림처럼 토큰들을 랜덤으로 [MASK]토큰으로 대체한 뒤 원래 토큰을 예측하는 문제입니다. 이 MLM을 이용하면 마스킹된 양방향 토큰들의 정보를 동시에 이용함과 동시에 컨닝 문제를 피할 수 있었고 그 결과 downstream task에서 그 이전 SoTA모델인 GPT1과도 꽤 큰 차이로 압도하는 성능을 보여주었습니다.

<img src="https://user-images.githubusercontent.com/19511788/125188555-2ed6ac00-e26f-11eb-8fab-9cbba66dc70c.png" style="width: 800px" class="center">
<center>Figure 14. GLUE Test Results by Pretrained Models</center>

구체적인 세부 내용들은 많이 생략하였지만 지금까지 NLP연구의 역사를 통해 어떻게 word2vec에서 BERT까지 진화해왔는지 살펴보았습니다. 제가 여기서 언급했던 모든 페이퍼들은 꼭 한 번씩 읽어볼만한 가치가 있다고 말씀드릴 수 있을 것 같고 이 글은 연구의 흐름을 보여주는 것에 초점을 맞췄다는 것을 다시 한 번 강조드리면서 글을 마치겠습니다.