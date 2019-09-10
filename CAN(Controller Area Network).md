## CAN(Controller Area Network)

- 우리가 일반적으로 사용하는 LAN(Local Area Network) 환경이 아닌 구조적으로 다른 Network 환경
- 1986년 메르세데스 벤츠사에서 로베르트 보쉬사에 3개의 ECU가 통신 가능한 네트워크 구조를 의뢰
- 1986년 보쉬사가 만들어서 자동차기술자 협회에서 발표
- 1991년 CAN 2.0 발표
- 1992년 메르세데스 벤츠에서 CAN을 채택한 자동차 출시
- 1993년 ISO에 의해서 표준화
- ECU(Eletronic Control Unit : 전자적 제어장치)
  - ACU(Airbag Control Unit)
  - BCM(Body Control Module)
  - ECU(Engine Control Unit)
  - TCU(Transmission Control Unit)
  - ABS(Anti-lock Brake System)
- 제조사별 ECU 개수 : Genesis(70개), 벤츠, BMW(80개), 렉서스(100개)

------

##### CAN은 어떤 방식으로 데이터 통신을 하나?(ECU 간의 데이터 통신은 어떻게 하나?)

- 기본적으로 각 ECU를 선으로 연결하는 방식은 좋지 않음

- IO단자가 많이 필요하고 소형화하는데 문제가 발생하여 비용 또한 발생

  - 그래서 다른 방식으로 통신
  - 직렬(Serial) 통신을 함. 각각의 ECU를 서로 직접 연결시키는게 아니라 BUS 개념을 도입
  - CAN은 CAN BUS에 대한 단일 입출력 Interface만 가지고 있는 것이 특징
  - 직렬통신 vs 병렬통신
    - 우리가 사용하는 대표적인 직렬통신 : USB, COM 포트

- 자동차는 기본적으로 네트워크 환경이 굉장히 열악(온도, 충격, 진동)하여 네트워크 통신할 때 많은 오류 발생 가능성 증가

- 자동차와 관련된 CAN 이외의 통신방식이 존재하지만 CAN이 대표적으로 많이 사용되고 언급되는 이유는 안정성

  ![1568079734523](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1568079734523.png)

------

##### CAN의 장점

- CAN은 Multi Master 통신을 함

  - 서버와 클라이언트가 존재하지 않음
  - 모든 ECU가 CAN BUS가 idle하다면 스스로 데이터를 CAN BUS를 통해서 전송 가능

- 노이즈에 매우 강함

  - 차량 자체가 매우 열악한 환경인데 CAN은 두가닥의 꼬인 전선을 이용하여 

    전선의 전압차를 통해 데이터를 전송

- 표준 프로토콜

  - 시장성을 확보 가능

- 하드웨어적으로 오류 보정이 가능

  - CRC를 하드웨어적으로 만들어서 전송
  - 받는측에서 CRC를 이용하여 데이터 프레임의 오류가 있는지를 확인
  - 만약 오류가 있으면 응답을 전송
  - 해당 응답을 확인해서 재전송

- 다양한 통신방식을 지원

  - Broadcast, Multicast, Unicast 모두 지원
  - 통신방식이 address를 기반으로 통신하는 방식이 아님

- ECU 간에는 우선순위가 존재

  - 각 ECU는 고유의 ID가 존재하는데 이 ID값이 작을수록 우선순위가 높음
  - 이 우선순위를 이용하면 급한 message를 먼저 처리 가능

- CAN BUS라는 것을 사용

  - 전선의 양을 줄일 수 있음 -> 비용절감의 효과