## Python_4

1. #### 날짜(date, time, datetime)

   - 현재 날짜

     ```python
     from datetime import date, time, datetime
     
     # 현재 날짜
     today = date.today()
     print("오늘의 날짜 : " + str(today))
     
     print("연도 : {0}, 월 : {1}, 일 : {2}".format(today.year, today.month, today.day))
     
     today = datetime.today() # 1/1000000 초까지
     print(today)
     ```

   - 날짜 연산

     ```python
     from datetime import date, time, datetime, timedelta
     from dateutil.relativedelta import relativedelta  # years, months 사용 하기 위한 패키지
     
     today = date.today() # 오늘의 날짜
     days = timedelta(days = -1)
     # days = timedelta(months = -1) years, months 사용 불가
     print(today + days)
     
     today = datetime.today() # 오늘 날짜와 시간
     delta = timedelta(hours = -3)
     print(today + delta)
     
     today = date.today()
     days = relativedelta(months = -2)
     print(today + days)
     ```

2. #### print문

   ```python
   for tmp in range(10):
       print("tmp : {}".format(tmp), end=", ") # 한줄 출력
   
   # print 함수는 기본적으로 println 처럼 작동
   # print 함수의 마지막 인자로 출력에 대한 제어가 가능
   ```

3. #### for 문

   ```python
   my_list = [1, 2, 3, 4, 5]
   my_sum = 0
   
   # for tmp in my_list:
   #   my_sum += tmp
   
   for tmp in range(len(my_list)):
       my_sum += my_list[tmp]
   
   
   print("합계 : {}".format(my_sum))
   
   my_list = [1, 2, 3, 4, 5]
   new_list = [tmp * 2 for tmp in my_list] # my_list에서 요소를 뽑아서 2배를 하고 tmp에 삽입
   print(new_list)
   
   new_list = [tmp * 2 for tmp in my_list if tmp % 2 == 0]
   print(new_list)
   ```