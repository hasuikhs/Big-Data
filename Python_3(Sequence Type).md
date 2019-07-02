## Python_3(Sequence Type)

1. #### range

   - range는 숫자 sequence로 주로 for문에서 사용

   - range(10) 처럼 인자가 1개이면 0부터 시작, 1씩 증가

     ```python
     my_range = range(10, 20) # 10부터 19까지 20은 제외
     print(my_range)
     ```

   -  range의 인자가 3개이면 시작, 끝, 증감을 의미

     ```python
     my_range = range(10, 20, 3) # 10, 13, 16, 19 
     print(10 in my_range) # True
     ```

   - range도 list나 tuple처럼 indexing과 slicing이 가능

     ```python
     my_range = range(10, 20, 3)
     print(my_range[-1])
     print(my_range[:2])
     ```

   - range를 이용한 for문

     ```python
     for tmp in range(10, 20, 2) :
         print(tmp)
     ```

2. #### dict

   - 표현법은 JSON표현과 유사 {"name" : "홍길동", "age" : 30}

     ```python
     my_dict = {"name" : "홍길동", "age" : 30}
     print(type(my_dict))  #  <class 'dict'>
     ```

   - 새로운 key : value를 추가할 경우

     ```python
     my_dict[100] = "홍길동"
     print(my_dict)		# {'name': '홍길동', 'age': 30, 100: '홍길동'}
     ```

   -  특정 key를 삭제할 경우

     ```python
     del my_dict["age"]
     print(my_dict)		#  {'name': '홍길동', 100: '홍길동'}
     ```

   -  key값이 중복되는 경우

     ```python
     my_dict = {"name" : "홍길동", "age" : 30, "age" : 40}
     print(my_dict)		# {'name': '홍길동', "age": 40}
     ```

   - keys()

     ```python
     my_dict = {"name" : "홍길동", "age" : 30, "address" : "서울"}
     print(my_dict.keys())
     # 리턴값은 key값들의 리스트처럼 생긴 객체
     # list와 유사하지만 list의 함수는 사용할 수 없다
     # values() : dict의 value값들만 뽑음
     # items() : (key, value)의 형태로 구성된 리스트처럼 생긴 객체를 리턴
     
     my_dict = {"name" : "홍길동", "age" : 30, "address" : "서울"}
     # for 문을 이용하여 모든 key와 그에 대한 value값을 출력
     for key in y_dict.keys() :
         print("{0}, {1}".format(key, my_dict[key]))
     ```

3. #### set

   - set의 가장 큰 특징 : 중복이 없는 저장장소, 순서가 없는 저장구조

     ```python
     my_set = set([1, 2, 3]) # set 생성 => {1, 2, 3}
     print(my_set)
     my_set = set("Hello") # {'H', 'e', 'l', 'o'}
     print(my_set)
     ```

   - 기본적인 set 연산(교집합, 합집합, 차집합)

     ```python
     s1 = {1, 2, 3, 4, 5}
     s2 = {4, 5, 6, 7, 8}
     
     print(s1 & s2)  # 교집합(intersection)
     print(s1 | s2)  # 합집합(union)
     print(s1 - s2)  # 차집합(differences)
     ```

   - 기타 사용가능한 method

     ```python
     my_set = {1, 2, 3, 4, 5}
     # set에 새로운 요소를 추가하려면
     my_set.add(10)
     
     # set에 여러개를 추가하려면
     my_set.update([7, 8, 9])
     
     # set에서 삭제할 경우
     my_set.remove(1)
     ```

4. #### bool

   ```python
   # 논리 상수인 True, False를 사용
   # 다음과 같은 경우는 False로 간주
   # 1. 빈문자열은 논리연산시 False("")
   # 2. 빈리스트([]), 빈튜플(()), 빈딕셔너리({})
   # 3. 숫자 0 False 간주, 나머지 다른 숫자 True 간주
   # 4. None => False간주
   ```

5. #### python의 console 입출력

   ```python
   input_value = input("숫자를 입력하세요!")
   # 입력받은 값은 무조건 문자열
   # eval() : 문자열을 숫자로 형 변환
   
   result = eval(input_value) * 3
   print(result)
   ```

6. #### if문

   ```python
   area = ["seoul", "suwon", "busan"]
   
   if "suwon11" in area :
       # 수행 코드 작성
       pass  # python은 블럭을 비워 둘 수 없음. pass를 사용해 코드란을 채움
   elif "busan" in area :
       print("부산")
   else :
       print("마지막입니다.")
   ```

   