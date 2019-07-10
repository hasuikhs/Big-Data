## Python_8(pandas)

### DataFrame 연습

```python
import numpy as np
import pandas as pd

data = [[2, np.nan],
        [7, -3],
        [np.nan, np.nan],
        [1, -2]]
df = pd.DataFrame(data,
                  columns = ["one", "two"],
                  index = ["a", "b", "c", "d"])
display(df)

display(df.sum(axis = 0)) # 결과값은 Series
display(df.sum(axis = 1))
# NaN은 실수 0.0으로 간주
```

```python
# "one" 열 항목만 합계
display(df["one"].sum())

# "b" 행 항목만 합계
display(df.loc["b"].sum())

display(df["one"].mean()) ## 평균을 구할때는 NaN 배제
```

```python
# 결측값 처리
## "one" 컬럼의 결측값은 "one" 컬럼의 평균으로 대체
df["one"].fillna(value=df["one"].mean(), inplace = True)

## "two" 컬럼의 결측값은 "two" 컬럼의 최소값으로 대체
df["two"].fillna(value=df["two"].min(), inplace = True)
```

```python
# random값을 도출해서 DataFrame을 생성
np.random.seed(0)

## 0~9까지의 정수형 난수를 생성(6, 4) 형태로 생성
df = pd.DataFrame(np.random.randint(0,10,(6,4)))
df.columns = ["A", "B", "C", "D"]
df.index = pd.date_range("20190101", periods = 6)

# 순열 랜덤 치환
random_date = np.random.permutation(df.index) 

# 원본은 고정되어 있고 바뀐 결과 DataFrame 리턴
df2 = df.reindex(index = random_date,
                 columns=["B", "A", "D", "C"])

# index(column) 기반의 정렬
df2.sort_index(axis = 1, ascending = True)

# value 기반의 정렬(오름차순)
df2.sort_values(by = ["B", "A"]) # [첫 정렬기준, 두번째 정렬기준]
```

```python
## 새로운 column을 추가
df["E"] = ["AA", "BB", "CC", "CC", "AA", "CC"]

## unique() 중복을 제거하기 위한 함수
type(df["E"].unique()) # 중복제거 후 ndarray 리턴

## 각 values값들의 개수를 세는 함수
df["E"].value_counts() # 결과가 Series로 리턴

## boolean mask를 만들기 위한 함수
df["E"].isin(["AA"])
```

```python
## 테이블 Merge(DB에서의 JOIN)

import numpy as np
import pandas as pd

data1 = {"학번" : [1, 2, 3, 4],
         "이름" : ["홍길동", "김길동", "이순신", "강감찬"],
         "학년" : [2 ,4, 1, 3]}
data2 = {"학번" : [1, 2, 4, 5],
         "학과" : ["컴퓨터", "경영", "철학", "기계"],
         "학점" : [3.4, 1.9, 4.5, 2.7]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)


pd.merge(df1, df2, on = "학번", how ="inner") 
# merge(테이블, 테이블, on = 기준, how = "inner" or "outer"("left", "right"))
```

```python
# 키의 이름이 다른 경우
pd.merge(df1, df2, left_on = "학번", right_on = "학생학번", how="inner")
```

```python
data1 = {"학번" : [1, 2, 3, 4],
         "이름" : ["홍길동", "김길동", "이순신", "강감찬"],
         "학년" : [2 ,4, 1, 3]}
data2 = {"학과" : ["컴퓨터", "경영", "철학", "기계"],
         "학점" : [3.4, 1.9, 4.5, 2.7]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2, index=[1, 2, 4, 5])

# 한쪽 DataFrame에 학번 열이 존재하지 않고 인덱스에 들어있는 경우
pd.merge(df1, df2, left_on = "학번", right_index = True, how = "inner")
```

```python
data1 = {"이름" : ["홍길동", "김길동", "이순신", "강감찬"],
         "학년" : [2 ,4, 1, 3]}
data2 = {"학과" : ["컴퓨터", "경영", "철학", "기계"],
         "학점" : [3.4, 1.9, 4.5, 2.7]}
df1 = pd.DataFrame(data1, index=[1, 2, 3, 4])
df2 = pd.DataFrame(data2, index=[1, 2, 4, 5])

# 양쪽 모두에 학번 열이 존재하지 않고 인덱스에 들어있는 경우
pd.merge(df1, df2, left_index = True, right_index = True, how = "inner")
```

```python
## pandas에서 필요한 데이터 파일을 다운로드
## google -> lending club data 
## 2015년 Loand Data 다운로드 후 적절한 위치에 저장
import numpy as np
import pandas as pd
```

```python
## 1. Data Loading
df = pd.read_csv("./data/LoanData/LoanStats3d.csv",
                 sep = ",",
                 skiprows = 1)
                 
## 총 데이터의 수 확인(몇 줄의 대출정보가 있는지 확인)
display(df.shape)
```

```python
## 필요한 column만 추출해서 새로운 DataFrame 생성
## loan_amnt : 대출 금액
## loan_status : 대출 상태
## grade : 대출 등급
## int_rate : 이자율
## term : 대출기간
df2 = df[["loan_amnt", "loan_status", "grade", "int_rate", "term"]]
df2.head()
```

```python
## 각 컬럼의 값 확인
display(df2["loan_status"].unique())
display(df2["grade"].unique())
display(df2["term"].unique())

## 결측값 제거
display(df2.shape)
df2.dropna(how = "any", inplace = True)
display(df2)
```

```python
## 대출기간별 대출 총액 구하기
print("대출기간 36 개월 대출 총액 : {:,}".format(df2.loc[df2["term"] == " 36 months", "loan_amnt"].sum()))
print("대출기간 60 개월 대출 총액 : {:,}".format(df2.loc[df2["term"] == " 60 months", "loan_amnt"].sum()))

terms = df2["term"].unique()
term_result = {}

for t in terms:
    term_sum = df2.loc[df2["term"] == t, "loan_amnt"].sum()
    term_result[t] = term_sum
    
for key in term_result:
    print("{}".format(key), end = " : ")
    print("{}".format("{:,}".format(term_result[key])))
```

```python
## 대출 중 불량한 상태의 대출에 대해서
## 불량한 상태의 대출( Charged Off, Late (31-120 days), Late (16-30 days), In Grace Period )
## 각 Grade의 대출 건수를 구하라

bad_loan_status = df2["loan_status"].unique()[[1, 3, 4, 5]]

# 불량 대출에 대한 mask 생성
bad_mask = df2["loan_status"].isin(bad_loan_status)
print(bad_mask.sum())

df3 = df2.loc[bad_mask, "grade"]
df3.value_counts().sort_index()  
```
