# 02_Python(기초)

## 1. 숫자

```python
a = 123			# 정수
b = 3.141592	# 실수
c = 3.14E10		# 실수(지수 형태)

div = 3 / 4		# 0.75

result = 3 ** 4	# 3의 4제곱 : 81

result = 10 % 3	# 나머지 : 1

result = 10 // 3# 나눗셈의 몫 : 3
```

## 2. 문자열

- python은 문자(`''`)와 문자열(`""`)의 구분이 없다.

  ```python
  a = "안녕하세요"
  b = 'hello'
  c = """ 여러 줄에 걸친
  		문자열도 
  		가능 하다."""	# tab 주의
  ```

### 2.1 문자열 연산

```python
first = '이것은'
second = '소리없는'
third = '아우성'
sample_text = "Show me the Money"

print(first + second + third)	# 이것은소리없는아우성

number = 100
print(first + str(number))		# 숫자를 문자열로 캐스팅해줘야 한다.

text = 'python'
print(text * 3)		# pythonpythonpython
```

- in, not in

  ```python
  print('sample' in sample_text)		# False
  print('sample' not in sample_text) 	# True
  ```

### 2.2 Indexing

```python
print(sample_text[0])	# S
print(sample_text[-1])	# y
```

### 2.3 Slicing

- 슬라이싱 시 앞은 포함 뒤는 불포함

  ```python
  print(sample_text[1:3])	# ho
  ```

- 슬라이싱 시 앞의 숫자가 생략되면 처음부터, 뒤의 숫자가 생략되면 끝까지

  ```python
  print(sample_text[0:])	# Show me the money
  print(sample_text[:3])	# Sho
  ```

### 2.4 Formatting

```python
apple = 40
my_text = "나는 사과를 %d개 가지고 있다." %apple
print(my_text)

apple = 5
banana = "여섯"
my_text = "나는 사과 %d개, 바나나 %s개 가지고 있다." %(apple, banana)
print(my_text)
```

### 2.5 문자열 내장 함수

- 문자열의 길이

  ```python
  len(sample_text)
  ```

- 특정 문자열의 빈도수

  ```python
  sample_text.count('me')
  ```

- 특정 문자열이 처음 등장하는 인덱스 반환

  ```python
  sample_text.find('o')
  ```

  - 찾는 문자열이 없으면 -1을 반환한다.

- join

  ```python
  a = ':'
  b = 'abcd'
  print(a.join(b))	# a:b:c:d
  ```

- 기타

  ```python
  a = "   hoBBy  "
  print(a.upper()) # 모두 대문자로 변환
  print(a.lower()) # 모두 소문자로 변환
  print(a.strip()) # 문자열의 앞,뒤 공백을 제거
  ```