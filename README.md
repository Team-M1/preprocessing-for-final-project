# preprocessing-for-gender-hate-classifier

### 악플 분류기 만들기의 첫번째 단계로, 텍스트에 성별혐오가 있는지를 분류하는 이진 분류기를 만드는 과정입니다.


</br>
학습에 사용된 데이터는 다음과 같습니다.

- [Korean HateSpeech Dataset](https://github.com/kocohub/korean-hate-speech)의 labeled 데이터 모두(train.tsv + dev.tsv)

  해당 데이터에서 `contain_gender_bias`가 `True`이고, `hate`가 `hate`인 데이터는 1로, 나머지는 0으로 레이블하여 사용함

- 직접 수집한 데이터 `(data/혐오표현.csv)`

  인터넷 커뮤니티에서 각자 100개 정도를 목표로 수집하여 직접 레이블을 붙였습니다.


</br>
데이터는 다음 전처리 과정을 거쳤습니다.

1. 정규식 `r"[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+"`으로 필터링 (한국어와 알파벳, 지정된 특수문자만 남김)
2. 4번 이상 반복되는 문자는 3글자만 남김. 예) "ㅋㅋㅋㅋㅋㅋㅋㅋㅋ" -> "ㅋㅋㅋ"
3. 2칸 이상의 공백문자는 1칸만 남김, 텍스트 맨 앞과 뒤의 공백 제거


</br>
학습에 사용된 모델은 [transformers](https://huggingface.co/transformers/)의 `ElectraModel`이며 그 중 `ElectraForSequenceClassification`를 사용하였습니다.

세부적으로, [monologg/KoCharELECTRA](https://github.com/monologg/KoCharELECTRA)(음절 단위 한국어 ELECTRA)의 모델과 토크나이저를 사용했습니다.


</br>
데이터가 약 7:1 정도 비율의 불균형 데이터이므로, [imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler)를 사용하여 학습 과정에서 데이터가 1:1의 비율로 학습될 수 있게 하였습니다.