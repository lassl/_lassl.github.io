---
layout: post
title: transformers에 모델 기여하기
author: doyoung.gwak
categories: [history]
tags: [opensource, transformers, gpt, ner]
---

커뮤니티가 활발하고 리뷰가 활발한, 그리고 인기있는 저장소에 기여해보는 경험은 협업에대한 시야를 넓혀주고, 연차높은 개발자들과 커뮤니케이션 해볼 수 있는 좋은 기회입니다. 큰 규모의 프로젝트 저장소에 기여한다는 것이 쉬운일은 아니겠지만, 많은 부분이 자동화되어있고 사람의 개입이 최소한으로 들어가는 흐름속에서 코드 기여를 할 수 있도록 잘 세팅되어 있습니다. 이런 best practice 프로세스를 기여자 입장에서 경험해보면, 본인의 프로젝트(회사 프로젝트나 개인 프로젝트 등)를 키워갈때나 협업이 필요한 환경에서 작업할 때 규모가 커져가는 프로젝트 속에서 어디로 가야하는지 방향을 잘 설정할 수 있을 것입니다.

이번 글에서는 제가 huggingface/transformers 저장소에 기여해본 경험을 바탕으로 어떤 작업물을 기여했는지와, 어떤 절차로 작업했는지, 그리고 transformers에 기여할 때 몇가지 팁들을 공유드리려고 합니다.

<img src="https://user-images.githubusercontent.com/37643248/132370743-00bf0ff7-0fed-4e36-a0b6-aaef3b5887b9.png" class="center">

## 기여 포인트 찾기

이미 많은 것들이 구현된 저장소에서 스스로 기여포인트를 발견하는 것은 쉽지 않을 수 있습니다. huggingface의 transformers를 사용해보다가 기여하고 싶은 점이 생겨서 기여를 한다면 더할나위 없이 좋겠지만, 누군가가 만들어 놓은 이슈들 중에 구현 혹은 개선 할 수 있는 것들을 찾아보는 방법도 있습니다. 간단하게는 문서의 오탈자를 수정하여 기여할 수도 있습니다.

## 행동강령 확인 및 커뮤니티 분위기 파악

다양하고 수많은 사람들이 오가는 커뮤니티이기 때문에 보통 이런 저장소에는 [행동강령(Code of Conduct) 문서](https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md)가 존재합니다. 건강한 커뮤니티를 위해서 의사소통할 때 갖춰야할 태도나, 해서는 안되는 것들을 설명합니다. 다양한 의견들을 존중, 실수에 대한 포용, 개인과 공동체를 함께 지향, 그리고 해서는 안 되는 행동들에 대한 정리가 있습니다. transformers 커뮤니티에서 행동강령을 위반하면 1. Correction 2. Warning 3. Temporary Ban 4. Permanent Ban의 지침이 따를 수 있습니다.

행동강령의 내용은 worst case에대한 설명입니다. 실제 PR을 날리거나 issue를 만들어서 소통할 때 참고하기 좋은 문서는 아무래도 이전에 만들어져왔던 수많은 issue들과 PR들입니다. 첫 PR 올리는 사람들은 어디까지 디테일하게 문서를 작성하는지, 어떻게하면 리뷰어가 답변을 잘 주는지 등을 분석해보시면 좋습니다. (다른 커뮤니티와 분위기가 어떻게 다른지 궁금하시다면 리눅스 저장소를 참고해보셔도 좋을것 같습니다. 리눅스 저장소는 코드리뷰가 살벌한 것으로 유명합니다.)

## Contribution 가이드 확인

다음으로는 `CONTRIBUTING.md` 입니다. 기여가 많이 일어나는 저장소에는 기여 관련 내용으로 문서가 존재할 가능성이 큽니다. 기여를 하고싶은 사람들이 어떻게 기여할 수 있는지, 기여할만한 거리들 찾는 방법부터 환경세팅, PR가이드 등이 설명되어 있습니다.

huggingface도 마찬가지로 `CONTRIBUTING.md` 가이드가 존재합니다. 

기여 가이드 문서를 확인해보면, 보통 새로운 모델의 논문을 낸 저자들이 huggingface에 contribution하고싶은 경우가 많은지, "Do you want to implement a new model?" 섹션이 꽤 상단에 올라가있네요. 저는 이번에 OpenAI가 이미 기여했었던 GPT2을 위한 새로운 NER 모델 (`GPT2ForTokenClassification`)을 구현해서 추가하는 작업이었기 때문에 이 부분은 넘어가도록 하겠습니다.

"Do you want a new feature (that is not a model)?"에서 가이드를 보면 아래와같이 5가지 유의사항을 알려줍니다.

1. 개선 동기를 먼저 말해달라
2. 기능을 완성된 문장으로 설명해 달라
3. **코드 조각**으로 설명해달라
4. 관련 논문이 있으면 링크를 걸어달라
5. 당신이 생각하기에 도움이 된다면 추가적인 자료들(그림이나 스크린캡처 등)을 올려달라

> 저는 글을 쓰는 시점 기준으로 아무것도 달성하지 못한것 같습니다만,, 워낙 다른 사람들도 필요할법한 기능이라 생각하므로 리뷰어의 코멘트를 조금 더 기다려보도록 하겠습니다 ㅎㅎ

"Start contributing! (Pull Requests)" 섹션에서는 로컬환경에서 어떻게 transformers 기여 환경을 구축하고, 테스트하는지를 간단한 명령어들로 설명합니다. 처음에는 명령어가 왜이렇게 많지 싶다가도, 테스트를 하다보니 이정도면 꽤 간편하게 환경세팅&테스트 할 수 있게 잘 만든것이겠구나 납득이 되었습니다 ㅎㅎ. 테스트 도구로는 pytest를 사용하고, python lint formatter는 black을 사용합니다. 문서 자동 생성을 위해 Sphinx를 사용하는데, 이것들이 기본적으로 어떻게 동작하고 어떤 목적으로 사용되는지는 알아야 했습니다. 조금 찾아보시면 잘 설명된 자료들이 많을테니 이 글에서 따로 설명하진 않겠습니다.

여기서 전체적인 작업 프로세스는 아래와 같습니다.

1. 오픈소스 저장소를 내 계정에 [fork](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)
2. fork 뜬 저장소에서 코드 수정 및 git commit & git push
3. fork 뜬 저장소에서 오픈소스 저장소로 Pull Request 생성 (CI 테스트가 자동으로 실행됨)
4. CI 테스트 결과 확인 및 실패 테스트 성공으로 해결
5. 테스트 통과했다고 리뷰어 태깅하면서 알림주기 (코멘트)
6. 리뷰어에게 리뷰 받고 코드 추가 수정
7. 어프로브 받고 머지되길 기다리기
8. 머지 및 완료 🎉

PR을 올리고 난 다음부터의 과정은 아래에서 설명하도록 하겠습니다.

## PR 올리고 CI 테스트 진행 및 결과 확인

PR을 올리면 아래와같이 자동으로 14개의 ci 테스트가 [CirecleCI](https://circleci.com/)에서 시작됩니다. CircleCI는 facebook, spotify 등 여러 큰 기업들에서도 사용하고 있는 [CI(Continuos Integration)](https://ko.wikipedia.org/wiki/%EC%A7%80%EC%86%8D%EC%A0%81_%ED%86%B5%ED%95%A9) 도구입니다. 빌드 머신을 직접 구축할 필요 없이 CircleCI에 세팅된 빌드 머신으로 지속적 통합 개발 프로세스를 만들 수 있습니다. 진행되었던 테스트는 바로 끝나는 작업은 아니고 아니고 전체 사이클이 2시간 정도 걸렸습니다. 테스트가 진행되는동안은, 귀찮아서 안 읽고 있던 `CONTRIBUTING.md` 가이드나 `README.md`를 좀 찬찬히 읽고 오셔도 괜찮을듯 합니다.

![image](https://user-images.githubusercontent.com/37643248/132370927-437854e9-2f75-439c-a25e-7e03b22d7aa1.png)

<center>PR을 만들고나면 CI 빌드&테스트가 시작된 모습 (혹은 커밋을 하고나도 다시 빌드&테스트가 시작)</center>

CI 체킹이 끝나고 실패한 테스트가 있으면 이렇게 친절하게 메일도 오네요. 처음엔 3개의 실패가 있었습니다. 이제 여기서 각 fail들의 에러메시지를 확인하고 고쳐주는 작업을 해보겠습니다.

여기서 좀 만만해보이는 `run_tests_torch`를 한번 살펴보겠습니다.

| ![image](https://user-images.githubusercontent.com/37643248/132371139-f2e47b55-a061-4e51-939a-feaf43d304ad.png) | ![image](https://user-images.githubusercontent.com/37643248/132371157-5d69f273-3c2b-41fa-a3cd-3badb9feb690.png) |
| :----------------------------------------------------------: | ------------------------------------------------------------ |
|             CI 웹 도구에서 확인한 실패의 흔적들              | 3개 실패, 11개 성공이지만, 이 테스트들 해결하는데 꽤 많은 시간이 소요될줄은 이때만해도 몰랐따 |

링크를 눌러 들어가보니 5000줄의 에러메시지를 확인할 수 있었는데요. 에러 메시지 길이에 압도되지 말고 빠르게 에러부분을 찾으러 가보겠습니다. 오른쪽상단의 더보기(?) 버튼을 눌러 raw text로 확인할 수 있었습니다. ([테스트 요약](https://app.circleci.com/pipelines/github/huggingface/transformers/27229/workflows/d6f82df0-9768-4a63-8fa5-6b05b19452f1/jobs/262650), [원본 에러메시지 링크](https://circleci.com/api/v1.1/project/github/huggingface/transformers/262650/output/110/0?file=true&allocation-id=6129832666290041547797fc-0-build%2F412F2DC))

![image](https://user-images.githubusercontent.com/37643248/132371491-d4a07636-96ce-4b38-bfbc-0347435b325f.png)

<center>너무 긴 테스트 로그 파일을 raw로 확인하는 중 - 실패했던 파일들과 테스트들 이름 확인</center>

fail을 키워드로 검색해보니 어느 파일에서 fail이 나왔는지 확인할 수 있었고, 해당 파일의 테스트함수를 타고 들어가보니 익숙한 에러메시지가 보였습니다. 테스트할 때 불러온 `GPT2Config`에는 `classifier_dropout` 속성이 없었는데, 저는 `classifier_dropout`를 사용하고 있었네요. `classifier_dropout` 속성이 없어도 디폴트값으로 사용할 수 있게 예외처리를 추가하여 git commit, push 해줍니다.

![image](https://user-images.githubusercontent.com/37643248/132371873-46d07318-221d-463d-b145-b6cb86cffbe0.png)

<center>너무 긴 테스트 로그 파일을 raw로 확인하는 중 - 테스트 실패한 코드라인 확인</center>

이것 외에도 chech_code_quality에서 여러번 fail이 되서 고쳐주었는데, python의 lint formatter인 black을 처음 써보는탓에 꽤 헤맸었고, `__init__.py` 체크 테스트에도 클래스를 추가하지 않은 실수가 있어서 테스트코드를 디버깅해가며 어디가 잘못된건지 한참 찾아 헤맸습니다. ci 테스트 로그에서 어떤 명령어를 실행하여 테스트를 수행하다가 fail이 나왔는지 기록이 남아있기 때문에, 그 명령어를 알아내서 로컬에서 테스트해보면 빠르게 테스트해볼 수 있습니다.

M1 macOS에서 환경을 세팅하고 컨트리뷰션 가이드에따라 makefile을 실행해가며 테스트를 했었는데 환경세팅이 다 되지 않았는지 중간에 에러를 뱉고 실행이 멈췄습니다. 이 문제는 centos 장비에서 다시 실행해봤을때 문제없이 잘 되었습니다. 제 macOS에서 환경 설정하다 꼬인건지, centos에 의존성이 있는건지는 추가적인 확인이 필요하겠습니다만, macOS에서 작업하다가 잘 안되면 centos로 넘어가곤 했습니다.

![](https://user-images.githubusercontent.com/37643248/132371966-939030ff-4ef7-4900-a636-ea249d382630.png)

<center>
  빌드&테스트를 모두 통과한 모습 <a href="https://app.circleci.com/pipelines/github/huggingface/transformers/27246/workflows/7fa54ade-55c7-4a16-99ae-f74c390278be">링크</a>
</center>

이렇게 테스트를 다 통과하고나면, 초록색이 나오고, 모든 테스트가 정상적으로 다 통과하는데 2시간이 걸렸습니다(중간에 실패가 뜨면 빨리 끝나기도 합니다).

### Huggingface 맴버로부터 리뷰받기

모든 테스트가 통과되고나서 하루, 이틀 안에 huggingface의 Member로부터 리뷰가 달렸습니다. 당시에 열려있는 PR만 100개가 다 되어가는터라 무관심인채로 1달이 지날수도 있을것 같아서, 테스트가 통과한 뒤에 코멘트도 달아보고, 리뷰어로 지정해볼만한 분들을 다 태깅하기도하면서 소심한 어그로(?)를 끌었습니다.

![](https://user-images.githubusercontent.com/37643248/132372099-b513aba0-ea98-4065-bd8a-1c1a64c20b7c.png)

<center>오른편에 여러번의 테스트 실패 흔적들</center>

![image](https://user-images.githubusercontent.com/37643248/132373282-86b51883-b824-43f5-9c8a-375e08197c88.png)

<center>적절한 리뷰어를 잘 모르겠으면 일단 태깅하고 보기</center>

실제로 저 리뷰어들이 담당자였던건지 **@patrickvonplaten** **@sgugger** 두명이나 리뷰를 해주고 코멘트를 달아주어서 시간을 오래 끌지 않고 처리를 할 수 있었습니다.

그 중에도 **@patrickvonplaten**는 코드레벨까지 내려가서 질문을 하는 모습을 볼 수 있었습니다. 테스트에 연연하다보니 추가되지 않아도 될 클래스가 추가되고 사용되고 있었는데, 제거해도 괜찮지 않냐는 피드백이었고 피드백이 적절하다고 생각하여 바로 수정 & 반영했습니다.

필요없는 클래스를 제거하고보니 다시 빌드 에러가 발생했습니다. 알고봤더니 제가 구현한 구현체에서 버그가 있었습니다. (그 와중에 **@sgugger**는 어프로브를 날려버리네요 ㅎㅎ) 이때 macOS에 임시 테스트 환경을 만들어놨었는데 아무래도 모델 추론 테스트는 gpu가 있는 리눅스 환경이 필요할것 같아서 리눅스 환경에 다시 tranformers 프로젝트를 클론받고 테스트환경을 세팅하여 ci 테스트에서 에러가나는 명령어를 그대로 재현해보고 본격적인 디버깅을 시작했습니다. 회사에 처음 입사하고 새로운 환경에 적응해야하는 기분을 오랜만에 느낄 수 있어서 쫄깃쫄깃한 느낌이 들었네요.

{:class="table table-bordered"}
| ![image](https://user-images.githubusercontent.com/37643248/132373963-5222559a-938d-4420-8eec-ae732f9c6e2f.png) | ![image](https://user-images.githubusercontent.com/37643248/132373787-631cd425-805f-44a1-abbc-783a3044a944.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|  빌드 실패 떴는데도 시원하게 어프로브 날려주는 **@sgugger**  |                  코드레벨까지 들어오는 리뷰                  |

결국 문제를 해결하고 테스트를 통과시켜 작업을 완료하였고, 리뷰를 해준 @sgugger와 @patrickvonplaten를 태깅하며 언급을 한번 했습니다. 코멘트를 달았던 리뷰어에게 다시 리뷰를 요청하는 기능도 잊지 않고 해주었습니다.

이런 과정 끝에 5일만에 PR이 huggingface/transformers에 잘 머지될 수 있었습니다. 제 PR에서는 최종적으로는 **@patrickvonplaten**가 리뷰를 머지시킬지 결정하는 분이셨네요.

![image](https://user-images.githubusercontent.com/37643248/132373768-9e57cb2f-8fd1-46cf-89c7-daaaeb59cdf8.png)

<center>마지막 코멘트와 함께 내 코드가 transformers에 머지된 모습</center>

마지막 merge 후 branch도 깔끔하게 제거해주고 정리했습니다.

