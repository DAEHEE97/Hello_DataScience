[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/corazzon/boostcourse-ds-510/blob/master/k-beauty-oversea-online-sale-output.ipynb)

## 국가(대륙)별/상품군별 온라인쇼핑 해외직접판매액
* 국가통계포털 : http://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1KE10081&vw_cd=MT_ZTITLE&list_id=JF&seqNo=&lang_mode=ko&language=kor&obj_var_id=&itm_id=&conn_path=MT_ZTITLE

### K-Beauty는 성장하고 있을까? 해외 직접판매를 한다면 어느 국가로 판매전략을 세우면 좋을까?
* K-Beauty란? [K-Beauty - Wikipedia](https://en.wikipedia.org/wiki/K-Beauty)
* e : 추정치, p : 잠정치, - : 자료없음, ... : 미상자료, x : 비밀보호, ▽ : 시계열 불연 ( 단위 : 백만원 )

## 필요 라이브러리 가져오기


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# os 별로 폰트를 다르게 설정해 줍니다.
if os.name == "posix":
    # Mac
    sns.set(font="AppleGothic")
elif os.name == "nt":
    # Window
    sns.set(font="Malgun Gothic")
```


```python
# 레티나 설정을 해주면 글씨가 좀 더 선명하게 보입니다.
# 폰트의 주변이 흐릿하게 보이는 것을 방지합니다.
%config InlineBackend.figure_format = 'retina'
```

## 데이터 로드하기

- e : 추정치, p : 잠정치, - : 자료없음, ... : 미상자료, x : 비밀보호, ▽ : 시계열 불연속


```python
df_raw = pd.read_csv("data/국가_대륙_별_상품군별_온라인쇼핑_해외직접판매액_202201.csv", 
                     encoding="cp949")
df_raw.shape
```


```python
df_raw
```


```python
# "국가(대륙)별" 데이터 빈도수 세기
# 데이터가 45개씩 들어있습니다.

df_raw["국가(대륙)별"].value_counts()
```


```python
# 미국 데이터만 따로 보기
df_raw[df_raw["국가(대륙)별"] == "미국"].head()
```

## 분석과 시각화를 위한 tidy data 만들기
* https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf



집계를 한 데이터를 tidy data 형식으로 만들어야 합니다.






### melt () 함수


- melt를 사용해보겠습니다. 
- melt는 행에 있는 데이터를 열로 옮기는 것입니다.


```python
df = df_raw.melt(id_vars=["국가(대륙)별", "상품군별", "판매유형별"], 
                 var_name="기간", value_name="million")
df
```

## 데이터 전처리
### 기간에서 연도를 분리하기


```python
# 2014년부터 2021년 분기별 데이터 

df["기간"]
```


```python
# 문자열 공백 기준 > 문자열 리스트로 변환 > 0번째 인덱스 반환 > str to int
"2019 4/4 p)".split()
"2019 4/4 p)".split()[0] 
int("2019 4/4 p)".split()[0])

```


```python
# map 안에 lambda 함수를 넣습니다. 
# x.split()한 데이터의 첫번째 인덱스 값을 integer 형으로 변경합니다.

df["기간"].map(lambda x : int(x.split()[0]))

```


```python
# map으로 "연도"라는 열을 만들었습니다.

df["연도"] = df["기간"].map(lambda x: x.split()[0])
#df
```


```python
# 이번에는 "분기" 열을 만들어보겠습니다.
# 공백과 /을 기준으로 데이터를 split하여 4/4에 있는 4와 4를 분리시켜보겠습니다.
# 4/4에 있는 앞의 4를 가져옵니다.

"2019 4/4 p)".split()[1].split("/")[0]

int("2019 4/4 p)".split()[1].split("/")[0])
```


```python
df["분기"] = df["기간"].map(lambda x : int(x.split()[1].split("/")[0]))
```


```python
df["분기"]
```


```python
df["분기"] = df["기간"].apply(lambda  x : x.split()[1].split("/")[0])
df["분기"] = df["분기"].astype(int)
df["연도"] = df["연도"].astype(int)
df.head()
```

### 금액을 수치데이터로 표현하기 위해 데이터 타입 변경하기


```python
import numpy as np
```


```python
# 2016년 이전 데이터는 결측치가 많고, -으로 기록되어 있습니다. 
# -로 표시된 값은 결측치이므로 numpy를 로드하여 np.nan을 사용하여 NaN으로 대치(replace) 합니다.

# astype(float)를 활용하여 데이터를 float 데이터 형태를 확인할 수 있습니다.
# -은 NaN으로 변경하였고, 숫자는 소숫점으로 변경하였습니다.


#replace()한 값을 million" 열에 덮어쓰기합니
df["million"] = df["million"].replace("-", np.nan).astype(float)
df
```

### 필요없는 데이터 제거하기


```python
# 데이터셋의 용량은 약 763KB입니다.
df.info()
```


```python
# 합계 데이터는 따로 구할 수 있기 때문에 전체 데이터에서 제거합니다.

df = df[(df["국가(대륙)별"] != "합계") & 
        (df["상품군별"] != "합계")].copy()
df
```


```python
# 데이터셋의 용량을 약 488KB 로 줄였습니다.

df.info()
```


```python
# 백만원에 결측치가 있습니다. 
# 금액이 없는 데이터도 있다는 것을 참고합니다.

df.isnull().sum()
```

## K-Beauty 시각화

### 전체 상품군 판매액

- NaN으로 표시된 결측치 데이터가 있으면 시각화가 잘 안나타날 수 있으므로,
- NaN으로 나타난 행의 데이터를 제거하고 시각화를 해보겠습니다.

- 판매유형별 계 데이터만 추출


```python
df
```


```python
df_total = df[df["판매유형별"] == "계"].copy()
```


```python
df_total.isnull().sum()
```


```python
# 연도에 따른 판매액을 lineplot

sns.lineplot(data=df_total, x="연도", y="million")

```


```python
# 연도별 판매액 / 상품군 별로 다른 색상으로 시각화
# legend 값을 밖에 표시하는 소스코드를 stack overflow에서 가져옵니다.

plt.figure(figsize=(15, 4))

sns.lineplot(x="연도", y="million", data=df_total, hue="상품군별")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

```


```python
# relplot을 lineplot의 서브플롯을 그리기 위해 사용하겠습니다.
# kind를 "line"으로 설정합니다.


sns.relplot(x="연도", y="million", data=df_total, hue="상품군별", kind="line")
```


```python
# 상품군별로 색상을 다르게 하기 위해 col 옵션을 활용하고, 한 행에 4개를 표시하기 위해 col_wrap 옵션을 사용합니다.



sns.relplot(x="연도", y="million", data=df_total, hue="상품군별", col="상품군별", col_wrap=4, kind="line")
```


```python
# 화장품이 다른 상품군들에 비해 해외직접판매액이 꾸준히 증가하다 감소 하는것 을 알 수 있습니다. 
# 다른 값들은 잘 안나타나므로 화장품을 빼고 보겠습니다.

# ~를 사용하면 데이터가 반전됩니다. 상품군별이 "화장품"인 데이터만 빼고 df_sub 변수에 담습니다.


df_sub =  df[~df["상품군별"].isin(["화장품"])]
df_sub
```


```python
# kind의 기본값이 scatter이므로 line으로 바꿔줍니다. 
# 의류, 패션 상품도 많이 판매함을 알 수 있습니다. 
# 가전 전자 통신기기와 음반 비디오 악기 쪽도 성장세가 있음을 알 수 있습니다.

sns.relplot(x="연도", y="million", data=df_sub, 
            hue="상품군별", col="상품군별", col_wrap=4, kind="line")
```


```python
# 의류, 패션 상품도 빼고 나니 가전고 음반 쪽 판매가 두드러지게 나타납니다.


df_sub = df_total[~df_total["상품군별"].isin(["화장품", "의류 및 패션 관련상품"])].copy()

sns.relplot(x="연도", y="million", data=df_sub, 
            hue="상품군별", col="상품군별", col_wrap=4, kind="line")
```

### 연도별 분기별 화장품 판매액 데이터 시각화




```python
# 화장품 데이터를 가져오기 위해 우선 boolean 식을 만들어 상품군별이 화장품인 데이터를 가져옵니다.
# copy()로 복사하지 않으면 원본 데이터에 영향을 미칠 수 있습니다.


df_cosmetic = df[(df["상품군별"] == "화장품")].copy()
df_cosmetic

```


```python
# "상품군별" 컬럼에 어떤 값들이 들어있는지 unique()로 확인하면 화장품만 있는 것을 확인할 수 있습니다.

df_cosmetic["상품군별"].unique()
```


```python
# 2019년까지 성장 하다, 2020 - 2021 에서 감소 하였습니다.

sns.lineplot(data=df_cosmetic, x="연도", y="million")

```


```python
# 연도별
# 분기 별로 살펴보겠습니다.
# 시각화 그래프를 그려보니 1분기부터 4분기까지 계속 성장하고 있습니다.
plt.figure(figsize=(15, 4))
sns.lineplot(data=df_cosmetic, x="연도", y="million", hue="분기")
```

### 기간별 화장품 판매액 데이터 시각화



```python
# 이번에는 연도가 아닌 기간으로 그래프를 그려보겠습니다.
# 글씨를 겹쳐지지 않게 하기 위해 xticks()를 사용하여 글자를 회전시킵니다.

plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_cosmetic, x="기간", y="million")
```


```python
# 국가, 대륙 별로 한번 출력해보기로 하고, df_cosmetic 데이터 프레임을 살펴봅니다.
df_cosmetic.head()


```


```python
# 중국에서의 가장 판매액이 높습니다.

plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_cosmetic, x="기간", y="million", hue="국가(대륙)별")
```


```python
df_cosmetic["국가(대륙)별"]
```


```python
# 중국을 빼고 시각화 그래프를 그리면 아세안에서 최근 높은 판매량을 기록했음을 알 수 있습니다.

plt.figure(figsize=(15,4))
plt.xticks(rotation=30)

sns.lineplot(data=df_cosmetic[df_cosmetic["국가(대륙)별"] != "중국"], x="기간", y="million", hue="국가(대륙)별")
```


```python
# "계" 데이터를 빼고 그래프를 그리니 온라인 면세점이 성장하고 있다는 사실을 알 수 있습니다.


plt.figure(figsize=(15,4))
plt.xticks(rotation=30)
df_sub = df[df["판매유형별"] != "계"].copy()
sns.lineplot(data=df_sub, x="기간", y="million", hue="판매유형별")
```


```python
# 계 + 온라인 면세점도 빼고 시각화
# 증가후 감소 추세

plt.figure(figsize=(15,4))
plt.xticks(rotation=30)
df_sub = df[(df["판매유형별"] != "계") & (df["판매유형별"] != "면세점")].copy()
sns.lineplot(data=df_sub, x="기간", y="million", hue="판매유형별", ci=None)
```

### 의류 및 패션관련 상품 온라인쇼핑 해외직접판매액


```python
df_fashion = df[df["상품군별"].str.contains("의류")]


plt.figure(figsize=(15, 4))
plt.title("의류 및 패션관련 상품")
plt.xticks(rotation=30)
sns.lineplot(data=df_fashion, x="기간", y="million", hue="국가(대륙)별")
```


```python
plt.figure(figsize=(15, 4))
plt.xticks(rotation=30)
sns.lineplot(data=df_fashion, x="기간", y="million", hue="판매유형별")
```

### 데이터 집계하기


```python
pivot = df_fashion.pivot_table(
    index="국가(대륙)별", values="million", 
    columns="연도", aggfunc="sum")
pivot
```

### 연산결과를 시각적으로 보기


```python
plt.figure(figsize=(15, 7))
sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".0f")
```

## 전체 상품군별로 온라인쇼핑 해외직접판매액은 증가했을까?


```python
sns.barplot(x="연도", y="million", data=df_sub)
```


```python
plt.figure(figsize=(15, 4))
sns.lineplot(x="연도", y="million", data=df_sub, hue="국가(대륙)별")
```

* lengend를 그래프의 밖에 그리기 : [matplotlib - Move legend outside figure in seaborn tsplot - Stack Overflow](https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot)


```python
plt.figure(figsize=(15, 4))
sns.lineplot(x="연도", y="million", data=df_sub, hue="상품군별")
```


```python

```
