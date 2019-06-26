## DAY01_LINUX(기본)

1. ##### VM Ware player 5(가상화 프로그램) 설치(버전은 CPU 사양에 따라)

2. ##### VM Ware player 실행 후 centos iso 마운트 후 설치

3. ##### 세부 설치 옵션

   - 네트워크 : 이더넷 연결
   - 로컬 저장소와 설치 타입 설정(개발자용)

4. ##### 필수 개념과 명령어

   - 관리자 계정 root 로그인 (사용자 목록 보기)

   - 로그인후 바탕화면 우클릭 > 터미널

     ```
     - 종료 : shutdown -P now
     		halt -p
     		init 0 ( runlevel )
     - 재시작 : shutdown -r now
     		  reboot
     		  init 6
     - 로그아웃 : exit, logout
     ```

   - 가상 콘솔

     CTRL + ALT + F1 ~ F6

   - RunLevel : init 명령어 뒤에 붙는 숫자를 의미

     ```
      Runlevel 숫자마다 의미가 정해져 있음
     
       0   : 종료모드
     
       1   : 시스템 복구 모드
     
     2 ~ 4 : Text 기반 다중 사용자 모드
     
       5   : 그래픽 기반 다중 사용자 모드
     
       6   : reboot				  
     ```

   - pwd ( print working directory)

   - cd ( change directory )

   - Terminal을 실행시킨 후 working directory를 /lib/systemd/system으로 이동

   - ls ( list ) : 현재 디렉토리 안의 파일이나 디렉토리의 목록을 출력

     ```
     ls -al   : list all
     
     /lib/systemd/system 안의 runlevel*을 ls로 출력
     
     ls -al runlevel*  : runlevel로 시작하는 list 전부 
     ```

   - 처음 부팅시 어떤 runlevel로 실행할지를 지칭하는 링크(바로가기 아이콘 정도로 이해)가 존재

     ```
     /etc/systemd/system/default.target
     
     이 링크를 다른 target으로 변경
     ln -sf /lib/systemd/system/multi-user.target  /etc/systemd/system/default.target
     
     원상 복귀
     ln -sf /lib/systemd/system/graphical.target   /etc/systemd/system/default.target
     ```

5. ##### 