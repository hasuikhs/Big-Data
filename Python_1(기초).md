## Python_1(기초)

#### 기본연산과 출력

- cell : 코드가 실행되는 객체
- dd : 현재 선택된 셀을 지우기
- b : 셀을 추가하기

1. #### 주석

   ```python
   # 한 줄 주석
   ''' 여러줄 주석 ''' 
   ```

2. #### 내장 상수

   ```python
   # True, False, None(값이 존재하지 않음을 의미)
   a = True
   b = False
   c = None
   ```

3. #### Built-in data type(내장형)

   1. ##### Numeric type(숫자형)

      ```python
      a = 123				# 정수
      b = 3.1415926535	# 실수
      c = 3.14E10			# 실수(지수형태로 표현)
      
      div = 3 / 4
      print(div)			# Java코드라면 0, python은 0.75
      					# 코드를 실행하려면 ctrl + enter
      result = 3 ** 4		# 3의 4제곱
      print(result)		# result = 81
      
      result = 10 % 3		# 나머지 연산
      result = 10 // 3	# 나눗셈의 몫
      ```

   2. ##### Text Sequence Type : str

      - 문자열 생성 방법

      - python은 문자(' ')와 문자열(" ")의 구분이 없음

        ```python
        a = "안녕하세요"
        b = 'hello'
        c = """ 여러 줄에 걸친
        		문자열도
        		가능 함"""
        ```

      - 문자열 연산, indexing, slicing

        ```python
        first = '이것은'
        second = '소리없는'
        third = '아우성'
        
        print(first + second + third)	# 결과 : 이것은소리없는아우성
        
        number = 100
        print(first + str(number))		# python은 변환을 해줘야 문자열 처리
        
        text = 'python'
        print(text * 3)					# 결과 : pythonpythonpython
        # 문자열의 연산도 가능
        
        sample_text= "Show me the Money"
        # str(문자열) => Text Sequence Type
        
        print(sample_text[0])			# 결과 : S
        print(sample_text[-])			# 결과 : y
        # 배열에 음수도 허용 음수를 입력하면 끝의 문자를 출력
        
        print(sample_text[1:3])			# 결과 : ho
        # 슬라이싱 시 앞은 포함 뒤는 불포함
        print(sample_text[:3])			# 결과 : Sho
        # 슬라이싱 시 앞의 숫자가 생략되면 처음부터, 뒤의 숫자가 생략되면 끝까지
        
        # 문자열연산 in, not in
        print("sample" in sample_text)
        
        # sample_text 안에 "sample" 이 있으면 True, 없으면 False
        print("sample" not in sample_text) # 대소문자 구분 함
        
        # 문자열 formatting
        apple = 40
        my_text = "나는 사과를 %d개 가지고 있다" %apple 
        # apple 대신 숫자 직접 입력 가능
        
        print(my_text) # 결과 : 나는 사과를 40개 가지고 있다
        
        apple = 5
        banana = "여섯"
        my_text = "나는 사과 %d개, 바나나 %s개 가지고 있다" %(apple, banana) 
        # 순차적으로 맵핑
        print(my_text) # 결과 : 나는 사과 5개, 바나나 여섯개 가지고 있다
        
        # 문자열 내장 함수
        sample_text = "cocacola"
        print(len(sample_text)) # 문자열의 길이 : len()
        print(sample_text.count("c")) # 특정 문자열의 등장 빈도수
        print(sample_text.find('o')) # 특정 문자열이 처음 등장하는 인덱스 반환
                                     # 만약 찾는 문자열이 없으면 -1 반환
        a = ":"
        b = "abcd"
        # => a:b:c:d
        print(a.join(b)) # => a:b:c:d
        
        a = "   hoBBy  "
        print(a.upper()) # 모두 대문자로 변환
        print(a.lower()) # 모두 소문자로 변환
        print(a.strip()) # 문자열의 앞,뒤 공백을 제거
        ```

4. #### 

5. #### 