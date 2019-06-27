## DAY02_LINUX(기본)

#### 명령어

|  명령어  | 설       명                                                  |       예 시        |
| :------: | :----------------------------------------------------------- | :----------------: |
| ls(list) | 기본적인 파일과 디렉토리의 리스트를 보여주는 명령어          |       ls-al        |
|    -a    | 숨김파일을 포함한 모든 파일을 보여주기 위한 옵션             |       ls -a        |
|    -l    | 파일에 대한 자세한 사항(퍼미션, 소유자,  그룹, 파일크기 등)  |       ls -l        |
|   cat    | 파일의 내용을 확인하고 싶을 때 간단히 사용하는 명령어        |     cat 파일명     |
|    cd    | working directory를 이동하기 위해서 사용                     |    cd 디렉토리     |
|   pwd    | 현재 사용중인 디렉토리 표시                                  |                    |
|    cp    | 파일이나 디렉토리를 복사할 때 사용                           | cp abc.txt bbb.txt |
|  touch   | 파일의 크기가 0인 파일을 생성할 때 사용                      |   touch newfile    |
|          | newfile이라는 파일이 없는 경우 파일사이즈가 0인 새 파일 생성 |                    |
|          | 현재 newfile이라는 이름의 파일이 존재할 경우 해당 파일의 수정날짜를          현재 날짜로 변경 |                    |
|    mv    | 파일을 이동하거나, 파일이나 디렉토리의 이름을 변경할 때      | mv aaa.txt bbb.txt |
|  mkdir   | 새로운 디렉토리를 생성                                       |                    |
|  rmdir   | 디렉토리를 삭제하기 위한 명령어 단, 디렉토리가 비어있어야 함 |                    |
|    rm    | 파일이나 디렉토리를 지울때 사용                              |       rm -rf       |
|    -r    | 재귀적으로 특정 디렉토리와 포함된 디렉토리들을 삭제할 때 사용 |                    |
|    -f    | 강제적                                                       |         r          |
|          | 텍스트 파일의 상위나 하위의 10줄을 출력 사용법은 cat과 동일  |                    |
|   more   | 해당 파일의 내용이 긴 경우 페이징 처리를 해서 출력 space로 이동 |                    |
|  clear   | termibal 화면 상의 내용을 지움                               |                    |

- 기본적으로 도시키를 제공(방향키를 이용하여 사용 된 명령어 재이용)
- 자동완성 : 파일이나 디렉토리의 이름의 일부만 입력한 후 TAB키를 이용해서 완성하는 기능
- 자동완성을 사용이 불가할 경우(동일한 파일명이 있는 경우) TAB키를 연타 비슷한 파일명을 가진 파일들의 목록 출력

------

#### gedit

- 윈도우 시스템의 메모장 같은 텍스트에디터

- 한글 사용은 window key + space bar

- 해당 프로그램은 GNOME이라는 윈도우 매니저를 이용하는 경우에만 사용 가능

- 터미널 모드인 경우 gedit 사용 불가능, 전통적으로 vi라는 에디터를 이용

  ```
  vi 에디터는 입력모드와 ex모드가 존재
   - 입력모드로 진입하기 위해서는 i key나 a key를 사용
   - 입력모드를 빠져나와 ex모드로 진입하기 위해서는 esc key 사용
  ```

  |  명령어   |            설  명            |
  | :-------: | :--------------------------: |
  |     x     |        한 글자씩 삭제        |
  |    dd     |          한 줄 삭제          |
  | 숫자 + dd | 숫자에 입력된 만큼의 줄 삭제 |
  |    yy     |          한 줄 복사          |
  | 숫자 + yy | 숫자에 입력된 만큼의 줄 복사 |
  |     p     |           붙여넣기           |
  |   : wq    |         저장 후 종료         |

------

#### 마운트

- 물리적인 장치(HDD, CD/DVD, USB)들을 사용하기 위해서 특정한 위치(디렉토리)에 연결하는 과정

  ```
  CD/DVD에 대한 장치 이름 => /dev 안에 cdrom이라는 이름으로 경로 지정
  현재 자동으로 mount 된 CD/DVD의 위치는 /run/media/root/CD 형태로 마운트
  
  # umount /dev/cdrom(/dev/sr0)
  
  이번에는 특정 mount point(directory)를 이용하여 CD/DVD를 mount
  # mkdir mycdrom (make directory)
  
  mount 명령을 이용해서 CD/DVD를 특정 디렉토리와 연결
  # mount -t iso9660 /dev/cdrom /root/mycdrom
  ```

- ISO파일을 제작해서 mount해서 사용하기

  ```
  ISO파일(.iso) : 국제표준기구(ISO)가 제정한 광학 디스크 압축 표준
  
  LINUX에서는 이런 iso파일을 쉽게 제작 가능
   1. 사용하는 프로그램은 genisoimage
   2. RPM을 이용 해당 프로그램(package)가 설치되어 있는지를 확인
      RPM(RedHat Package Manager)
      => # rpm -qa genisoimage (rpm에게 qa 질문 genisoimage 있냐?)
  이 프로그램을 이용해서 /boot 디렉토리의 내용을 iso파일로 압축
    => # genisoimage -r -J -o boot. iso /boot
  ```

------

#### 사용자와 그룹과 퍼미션

- LINUX는 다중 사용자 시스템

- 기본적으로 root라는 이름의 super user가 존재

- 모든 사용자는 특정 그룹에 속함

- 리눅스 시스템의 모든 사용자는 /etc/passwd 파일에 정의

  cat /etc/passwd 시 나오는 화면 내용 설명

  root             : x              :  0             : 0          : root                     :  /root          : /bin/bash

  사용자이름 : 패스워드 : 사용자ID : 그룹ID : 사용자전체이름 : 홈 디렉토리 : 기본 쉘(명령어 해석기)

- 그룹에 대한 정보는 /etc/group 파일에 

- 새로운 사용자를 추가

  ```
  # useradd testuser : testuser라는 이름의 사용자를 추가
     => 새로운 사용자 추가할 때 특정 옵션을 이용하지 않고 사용자를 추가하게 되면
        사용자 ID는 마지막 등록 사용자 ID에 +1로 생성
  # useradd -u 1111 testuser : 사용자를 추가할 때 특정 사용자ID를 부여
  # useradd -g root testuser : 사용자를 추가할 때 사용자의 그룹을 root 그룹으로 추가
  # useradd -d /newhome testuser : 기본적으로 일반 사용자의 홈디렉토리는 /home/testuser로 잡힘
  ```

  ```
  # 실습
  # useradd -g centos testuser 
      => 새로운 사용자 추가 성공
  # 아직 로그인은 불가능(비밀번호 설정 필요)
  # passwd testuser
  ```

- 사용자 정보 수정

  ```
  # usermod -g root testuser
  ```

- 사용자 삭제

  ```
  # userdel -r testuser
     => -r 옵션을 주면 해당 사용자의 홈 디렉토리도 같이 삭제
  ```

------

#### 파일과 디렉토리의 소유와 퍼미션

-rw-r--r--.  1 root root        47  6월 27 19:12 sample

```
1. 맨 앞의 1칸은 이 파일의 종류를 지칭
   - : 파일을 지칭  d : 디렉토리를 지칭  l : 링크(심볼릭링크)를 지칭
2. 뒤의 9칸은 해당 파일(디렉토리)의 퍼미션을 지칭
   rw-                r--                r--
   소유자의 퍼미션      그룹의 퍼미션        other의 퍼미션
   r : readable       w : writable       x : excutable
```

```
-rw-r--r--.  1 root root        47  6월 27 19:12 sample

=> 소유자와 같은 그룹의 사람만 해당 파일을 읽거나 수정이 가능한 상태로 만들고 
	실행은 안되게 설정하고 싶다?
   rw- rw- --- => 660
파일의 퍼미션 변경
# chmod 660 sample.txt
  
파일의 소유자와 그룹을 변경 가능
# chown centos sample.txt => 해당 파일에 대한 소유자를 변경
# chown centos.centos sample.txt => 해당 파일에 대한 소유자와 그룹을 동시 변경
# chgrp centos sample.txt => 해당 파일에 대한 그룹만 변경
```