## Python_2(Sequence Type)

1. #### LIST

   - Java의 ArrayList와 상당히 유사

   - list의 생성

     ```python
     # list의 생성
     a = list()				# 공백 리스트 생성
     a = []					# 공백 리스트 생성
     a = [1, 2, 3]			# 일반적인 리스트 생성
     a = [1, 2, 3, "안녕하세요", 3.14, False]	# 여러 타입을 사용
     a = [1, 2,[3, 4, 5], 6, 7]
     ```

   - indexing과 slicing 사용 가능(문자열 사용과 유사)

     ```python
     print(a[0])
     print(a[-2])
     print(a[1:3])
     ```

   - list의 연산

     ```python
     a = [1, 2, 3]
     b = [4, 5, 6]
     print(a + b)	# 연결[1, 2, 3, 4, 5, 6]
     print(a * b)	# [1, 2, 3, 1, 2, 3, 1, 2, 3]
     
     a = [1, 2, 3]
     a[0] = 5
     print(a)		# [5, 2, 3]
     
     a[0] = [9, 9, 9]
     print(a)		# [[9, 9, 9], 2, 3]
     
     a[0:1] = [9, 9, 9] # list안의 list가 아닌 list의 요소를 변경
     print(a)		# [9, 9, 9, 2, 3]
     
     a = [1, 2, 3, 4, 5, 6, 7]
     # 위의 list를 1, 2, 6, 7 로 바꾸려면
     a[2:5] = []
     print(a)
     ```

   - list의 사용 함수

     ```python
     my_list = list([1, 2, 3])
     print(my_list)
     my_list.append(4) # 맨 마지막 인덱스에 값을 추가
     print(my_list)
     my_list.append([5, 6, 7]) # [1, 2, 3, 4, [5, 6, 7]] <- 마지막 인덱스에 해당 인자를 삽입
     print(my_list)
     
     my_list.extend([8, 9, 10]) # [1, 2, 3, 4, [5, 6, 7], 8, 9, 10]  <- 리스트 확장
     print(my_list)
     
     my_list = [7, 3, 1, 8, 2]
     my_list.sort() # 기본적인 오름차순 정렬, 결과를 리턴하지 않고 원본을 제어
     my_list.reverse() # 내림차순 정렬, 내림 차순 정렬을 하려면 먼저 sort()로 오름차순으로 정렬 후 실행
     print(my_list)
     
     my_list(my_list.index(1)) # index()는 찾는 값의 위치를 리턴
     ```

2. #### TUPLE

   - list와 거의 유사

   - 표현 방법이 다르고 , 수정, 삭제가 불가능

   - 리스트는 [], 튜플은 ()

     ```python
     a = ()
     a = (1, 2 ,3)	# [1, 2, 3]
     a = (1,)        # [1]
                  	# 요소가 1개 있을 때 tuple을 표현하려면 
                 	# a = (1,)
     a = (1, 2, 3, 4)# tuple
                   	# tuple은 ()를 생략 가능
     a = 1, 2, 3, 4
     a, b, c = 10, 20, 30
     print(a)
     ```

   - indexing과 slicing 둘 다 사용 가능

     ```python
     a = (1, 2, 3, 4)
     print(a[1]) # 2출력
     print(a[2:4]) # (3, 4)
     ```

   - list와 마찬가지로 +, * 연산이 가능

     ```python
     a = (1, 2, 3)
     b = (5, 6, 7)
     print(a + b) # 결과 (1, 2, 3, 5, 6, 7)
     ```

   - list와 tuple간의 관계

     ```python
     my_list = [1, 2, 3]
     my_list = tuple(my_list)	# list를 tuple로 변환
     print(my_list)
     
     my_tuple = 10, 20, 30, 40
     my_list = list(my_tuple)	# tuple을 list로 변환
     print(my_list)
     ```

     

