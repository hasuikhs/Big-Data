## Python_7(pandas)

### 데이터 전달 포멧

```
1. CSV(Comma Seperated Value) 
	-  ","로 데이터를 구분해서 구분된 데이터를 전송, 이용하는 방법
	ex) 홍길동, 서울, 30, 김길동, 인천, 20, 박길동, 제주, 50 ...
	장점) 부가적인 데이터가 작다
		=> 대용량의 데이터를 처리하는데 적합
	단점) 데이터를 사용하려면 기본적인 파서프로그램을 작성
	      데이터의 형태가 변형이 되면 프로그램 자체를 수정
		=> 유지보수 문제
	      구조적인 데이터를 표현하기 적합하지 않다

2. XML
	- 마크업 태그를 이용해서 데이터를 표현하는 방식
	ex) <name>홍길동</name><address>서울</address><age>30</age> ...
	장점) 제공되는 파서를 이용해서 데이터 처리가 용이
	      구조적인 데이터를 표현하기에 적합
	단점) 부가적인 데이터가 너무 큼
		=> 초기에는 문제가 되지 않음

3. JSON(JavaScript Object Notation)
	ex) {"name" : "홍길동", "address" : "서울", "age" : 30 ...}
	장점) 구조적 데이터 처리 적합
	      XML에 비해 부가적인 데이터가 적음
	단점) CSV방식에 비해 부가적인 데이터가 존재
```

1. #### pandas?

   - pandas는 python data 분석의 핵심 module(package)

   - 데이터 분석

     ```
     1. EDA(탐색적 데이터 분석) : 예) 엑셀을 이용한 데이터 분석
                                   python언어로 pandas를 이용해서 EDA 수행
     2. 통계적 데이터 분석 : 통계적 이론을 이용한 분석
     3. 머신러닝 : 기존 데이터를 이용해서 프로그램을 학습, 학습된 결과를 이용해 예측
     ```

   - Pandas는 고유한 자료구조를 2개 사용

     | 자료구조  |                             설명                             |
     | :-------: | :----------------------------------------------------------: |
     |  Series   | numpy의 1차원 배열과 유사, 동일한 데이터 타입의 복수개의 요소로 구성 |
     | DataFrame | Table 형식으로 구성된 자료구조, 2차원 배열과 유사, DB 테이블과 유사 |

2. #### Series

   - ```python
     import numpy as np
     import pandas as pd
     
     arr = np.array([-1, 10, 50 ,99], dtype = np.float32)
     print(arr)	# [-1 10 50 99]
     
     s = pd.Series([-1, 10, 50, 99], dtype = np.float64)
     display(s)	
     print(s.values)	# numpy 1차원 array로 리턴 [-1. 10. 50. 99.]
     print(s.index)	# RangeIndex(start=0, stop=4, step=1)
     print(s.dtype)	# float64
     ```

   - ```python
     s = pd.Series([-1, 10, 50, 99], index=['c','a','k','tt'])
     display(s)
     print(s['a'])		# 다른 형식의 인덱스 사용 가능
     print(s[1])			# 원래 형태의 숫자 인덱스 또한 사용 가능
     print(s[1:3])		# 일반적인 slicing 사용 가능
     print(s['c':'k'])	# slicing 사용 가능 하지만 일반적인 방법과는 다르게 뒤의 인덱스의 값도 포함
     print(s.sum())		# 인자들의 전체 합 158
     ```

   - ```python
     ## A 공장의  2019-01-01 부터 10일간 제품 생산량을 Series에 저장
     ## 단, 생산량의 평균은 50이고 표준편차는 5인 정규분포에서 생산량을 랜덤하게 결정
     ## B공장의 평균은 70이고 표준편차는 8
     import numpy as np
     import pandas as pd
     from datetime import date, timedelta
     from dateutil.parser import parse
     
     start_day = parse("2019-01-01")
     factory_a = pd.Series([int(x) for x in np.random.normal(50, 5, (10,))],
                          index=[start_day + timedelta(days=x) for x in range(10)])
     #display(factory_a)
     
     factory_b = pd.Series([int(x) for x in np.random.normal(70, 8, (10,))],
                          index=[start_day + timedelta(days=x) for x in range(10)])
     #display(factory_b)
     
     display(factory_a + factory_b)
     ```

   - 이전에는 Series라는 자료구조를 만들때 python의 list를 이용했지만 dictionary를 이용하여 만들기

     ```python
     import numpy as np
     import pandas as pd
     
     my_dict = {"서울": 3000, "부산" : 2000, "제주" : 8000}
     s = pd.Series(my_dict)
     
     s.name = '지역별 가격 데이터'
     s.index.name = "지역"
     display(s)
     ```

3. #### DataFrame

   - 거의 대부분의 경우 DataFrame을 이용해서 데이터 분석 dictionary를 이용해서 생성

     ```python
     import numpy as np
     import pandas as pd
     
     data = {"name" : ["kim", "lee", "park", "moon", "kim"],
             "year" : [2015, 2016, 2019, 2019, 2015],
             "point" : [3.1, 4.3, 1.2, 2.3, 3.9]}
     df = pd.DataFrame(data)
     display(df)
     print("DataFrame의 shape : {}".format(df.shape))	# (5, 3)
     print("DataFrame의 요소개수 : {}".format(df.size))# 15	
     print("DataFrame의 차원 : {}".format(df.ndim))	# 2
     print("DataFrame의 index : {}".format(df.index))	
     # RangeIndex(start=0, stop=5, step=1)
     print("DataFrame의 컬럼 : {}".format(df.columns))	
     # Index(['name', 'year', 'point'], dtype='object')
     print("DataFrame의 데이터 : {}".format(df.values))
     #[['kim' 2015 3.1]
     # ['lee' 2016 4.3]
     # ['park' 2019 1.2]
     # ['moon' 2019 2.3]
     # ['kim' 2015 3.9]]
     ```

   - CSV파일을 이용해서 DataFrame을 생성하기

     ```python
     df = pd.read_csv("./data/movielens/ratings.csv")
     ```

     ```python
     ## DataBase로부터 데이터를 얻어내서 DataFrame을 생성
     ## mysql을 이용해서 처리해 보자
     ## python에서 mysql을 사용할 수 있또록 도와주는 Module 설치 및 사용
     # conda install pymysql
     import pymysql.cursors
     import numpy as np
     import pandas as pd
     
     ## mysql 설치 및 기동 과정
     ## mysql을 다운로드 받아서 간단하게 설치-설정
     ## 1. 다운로드 받은 mysql.zip 파일을 압축 해제 후
     
     ## 2. mysql bin 폴더에서 shift + 우클릭 => 여기서 명령창 실행 (command 창의 working directory)
     
     ## 3. 일단 DBMS 기동 => mysqld => 실행 성공후 창을 minimize
     
     ## 4. mysql 서버가 실행되고 있는 상태이기 때문에 다른 command창을 이용해서 mysql console로 진입
     
     ## 5. 새로운 command창 실행 후 mysql -u root
     
     ## 6. mysql console에 진입하면 mysql로 프롬프트가 변경
     
     ## 7. 새로운 사용자를 생성
     ##   mysql> create user python identified by "python"; 
     ##   mysql> create user python@localhost identified by "python";
     ##   외부연결에 대비해 로컬호스트 계정 생성
     
     ## 8. database를 생성
     ##   mysql> create database library;
     
     ## 9. 새로운 사용자에게 database사용 권한을 부여
     ##   mysql> grant all privileges on library.* to python;
     ##   mysql> grant all privileges on library.* to python@localhost;
     
     ## 10. console에서 command 창으로 빠져 나감
     ##   mysql> exit
     ## 11. command 창에서 제공된 script file을 이용해서 database 구축
     ##   > mysql -u python -p library < _tableDump.sql
     ## 데이터베이스를 연결하기 위한 변수를 하나 생성
     conn = pymysql.connect(host="localhost", user="python", password="python", db="library", charset="utf8")
     
     sql = "select bisbn, btitle, bauthor, bprice from books"
     df = pd.read_sql(sql, con=conn)
     display(df)
     ```

   - 3가지 형태의 데이터 파일을 이용

     ```python
     df = pd.read_csv("./data/movielens/ratings.csv")
     
     ## DataFrame에 들어있는 데이터를 JSON 형태로 저장하기
     new_df = df.head()
     display(new_df)
     
     new_df.to_json("./data/movielens/ratings.json")
     ```

   - JSON파일을 읽어서 DataFrame 만들기

     ```python
     file = open("./data/movielens/ratings.json", "r")
     my_dict = json.load(file)
     file.close()
     df = pd.DataFrame(my_dict)
     display(df)
     ```

   - pandas의 DataFrame의 제어

     ```python
     import numpy as np
     import pandas as pd
     
     data = {
             "이름" : ["홍길동", "강감찬", "이순신", "신사임당"],
             "학과" : ["컴퓨터", "경영학", "철학", "미술"],
             "학년" : [1, 3, 2, 4],
             "학점" : [3.1, 2.9, 4.5, 3.9]
            }
     df = pd.DataFrame(data, columns=["학과", "이름", "학년", "학점"], index=["one", "two", "three", "four"])
     display(df)
     ## DataFrame은 기본적인 분석함수를 제공
     display(df.describe()) # count, mean, std, min, (1q, 2q, 3q), max
     ```

     ```python
     import numpy as np
     import pandas as pd
     import warnings
     
     # warning을 출력하지 않기 위한 설정
     warnings.filterwarnings(action="ignore") # off
     # warning.filterwarnings(action="default") # on
     
     data = {
             "이름" : ["홍길동", "강감찬", "이순신", "신사임당"],
             "학과" : ["컴퓨터", "경영학", "철학", "미술"],
             "학년" : [1, 3, 2, 4],
             "학점" : [3.1, 2.9, 4.5, 3.9]
            }
     df = pd.DataFrame(data, columns=["학과", "이름", "학년", "학점"], index=["one", "two", "three", "four"])
     display(df)
     
     ## DataFrame에서 특정 column만 추출
     ## 컬럼 1개만 들고오면 => Series
     # display(df["이름"])
     year = df["학년"] # year => Series => View로 가져옴 => 원본 데이터에 영향 O
     # year = df["학년"].copy() # View가 아닌 복사본을 이용하려면 => 원본 데이터에 영향 X
     year[0] = 100
     display(year)
     display(df)
     
     ### 2개 이상의 컬럼을 추출하기
     ## display(df["이름", "학년"])  # Error
     ### Fancy indexing을 이용
     ## 인덱싱하는 부분에 인덱스 배열을 이용하는 indexing방법
     df[["학과", "이름"]] ## DataFrame 추출
     ```

   - 컬럼 값 수정하기

     ```python
     # df["학년"] = 4              # 스칼라
     # df["학년"] = [1, 1, 2, 2]   # 리스트
     df["학년"] = np.array([1, 1, 3, 2]) # numpy array
     ```

   - 기존에 존재하지 않는 새로운 컬럼을 추가하기

     ```python
     # df["나이"] = [20, 21, 22, np.nan]
     # 나이를 모를때 None 이나 numpy의 np.nan 이용 
     # df["나이"] = pd.Series([20, 21, 23, 24], 
     #                      index=["one", "two", "three", "four"]) 
     # pandas의 Seires는 인덱스를 일치 해야함
     df["나이"] = pd.Series([20, 21, 23], 
                          index=["one", "two", "four"])
     ```

   - ```python
     # 새로운 컬럼을 추가할때 조건을 받아 True False
     df["장학금여부"] = df["학점"] > 3.0
     ```

   - 컬럼 삭제하기

     ```python
     del df["학점"]  # 학점 컬럼을 삭제 일반적인 방법 X
     
     # 컬럼을 삭제하는 일반적인 방법
     # df.drop("학점", axis=1, inplace=True) 
     # 컬럼과 레코드 까지 삭제 가능 (axis=0 행, axis =1 열)
     # inplace=True 원본삭제
     new_df = df.drop("학점", axis=1, inplace=False)
     ```

   - row indexing

     ```python
     # index 번호로 row를 선택 불가능
     # display(df[0])     # X
     # display(df["one"]) # X
     # index 번호로 특정 row를 선택 불가능
     # 하지만 slicing 가능, slicing 결과는 DataFrame
     # display(df[:-1])
     display(df["one" : "three"]) # 임의로 정한 index는 slicing 시에 뒤의 index의 값도 포함
     ```

     ```python
     ## row indexing
     display(df)
     display(df[0:1]) ## df['~~~'] => 컬럼제어할 때
     # 일반적으로 행을 제어할 때
     # df.loc[0] # 0번째 행을 indexing 하려는 의미 같아 보이지만, 사용 불가능
     display(df.loc["one"])
     # index기반으로 indexing은 가능
     # 1개의 행을 선택하는 것이기 때문에 Series로 리턴
     display(df.loc["one":"three"])
     display(df.loc[["one", "three"]]) # Fancy indexing
     ## loc 사용 시 컬럼에 대한 indexing도 가능
     display(df.loc["one":"three", "이름":"학년"])
     display(df.loc["one":"three", ["이름","학년"]]) # "이름","학년" 의 열만 추출하는 Fancy indexing
     
     ## 새로운 행을 추가하기
     df.loc["five",:] = ["물리", "유관순", 2, 3.5] # 기존의 index를 쓰면 선택, 없는 index를 쓰면 추가
     display(df)
     ## row를 삭제하려면
     df.drop(["one", "three"], axis=0, inplace=True)
     display(df)
     ```

   - DataFrame 간단한 문제

     ```python
     # 1. 이름이 '강감찬'인 사람을 찾아서 이름과 학점을 DataFrame으로 출력
     
     df["이름"] == "강감찬" # boolean mask
     display(df.loc[df["이름"] == "강감찬", ["이름", "학점"]])
     
     # 2. 학점이 2.5초과 3.5미만인 사람을 찾아 학과와 이름을 출력
     
     display(df.loc[df["학점"] > 2.5].loc[df["학점"] < 3.5, ["학과", "이름"]])
     
     display(df.loc[(df["학점"] > 2.5) & (df["학점"] < 3.5),["학과", "이름"]])
     
     # 3. Grade라는 컬럼을 추가한 후 학점이 4.0 이상인 사람을 찾아 해당 사람만Grade를 A로 설정
     df["Grade"] = df["학점"] > 4.0
     
     display(df)
     
     display(df.replace(True,"A"))
     display(df.loc[df["Grade"] == True].replace(True,"A"))
     ```

     ```python
     # DataFrame 조작을 위해 간단한 DataFrame을 생성
     # 사용하는 DataFrame의 value값은 [0, 10) 범위의 난수형 정수를 균등분포에서 추출하여 사용
     # 6행 4열짜리 DataFrame을 생성
     import numpy as np
     import pandas as pd
     
     np.random.seed(0)
     df = pd.DataFrame(np.random.randint(0, 10,(6, 4)))
     
     df.columns = ["A", "B", "C", "D"]
     df.index = pd.date_range("20190101", "20190106") # pd.date_range("20190101", periods=6)
     # 컬럼 : ["A", "B", "C", "D"]
     # index : 날짜를 이용 (2019-01-01 부터 하루씩 증가)
     
     # NaN을 포함하는 새로운 컬럼 "E"를 추가
     # [7, np.nan, 4, np.nan, 2, np.nan] 데이터 추가
     
     df["E"] = [7, np.nan, 4, np.nan, 2, np.nan]
     
     display(df)
     #######################
     ## 결측값 처리
     #######################
     ## 결측값을 제거(NaN이 포함된 row를 제거)
     # df.dropna(how = "any", inplace = True)
     # display(df)
     
     # 결측값을 다른 값으로 대체
     #df.fillna(value=0, inplace =True)
     #display(df)
     
     # 결측값을 찾기 위한 mask
     #display(df.isnull())
     
     # 간단한 예제
     # "E" column의 값이 NaN인 모든 row를 찾고 해당 row의 모든 colums의 값을 출력
     
     display(df.loc[df["E"].isnull()])
     ```

   - 통계용어

     ```
     - 평균(mean) : 수학적 확률의 기댓값
                   어떤 사건을 무한히 반복했을 때 얻을 수 있는 평균으로서 기대할 수 있는 값
     
     - 편차(deviation) : 확률변수 X와 평균값의 차이
                         데이터의 흩어진 정도를 수치화 하기에는 부적합, 편차의 합은 0이기 때문
     
     - 분산(variance) : 데이터의 흩어진 점도를 알기 위해서 사용되는 편차 제곱 평균
                       제곱을 사용했기 때문에 단위 표현이 애매해지는 경우가 존재
     
     - 표준편차(standard deviation) : 분산의 제곱근
     
     - 공분산(covariance) : 두개의 확률변수에 대한 관계를 보여주는 값
     - 두개의 확률변수에 대한 공분산 값이 양수 
       => 하나의 확률변수가 증가하면 다른 확률 변수도 증가하는 경향 존재
     - 공분산이 0 : 두 데이터는 독립
     
     - 상관관계(corelation) : 두 대상이 서로 연관성이 있다고 추측되는 관계
     
     - 상관계수(corelation coefficient) : -1과 1 사이의 실수
       * 상관관계는 인과 관계 설명이 불가능
     ```

     
