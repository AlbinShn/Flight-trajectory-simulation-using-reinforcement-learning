# Flight-trajectory-simulation-using-reinforcement-learning
Flight trajectory simulation using reinforcement learning

이 버전은 deflection angle에 의해 컨트롤하는 case 2 입니다.

저희의 목표는 target까지 최대한 빠르게 이동해 target을 맞추면 됩니다.

이 예제는 카트폴 예제를 참고하였습니다.

1. Fortress Invasion System

    (1) Requirements : 

        state = (x, y, v, psi)
        
        목표 : fortress 파괴, bullet, wall, Obstacle과 충돌하면 게임 종료


    (2) case 1 : 측면 가스분사

        condition : 진공, 좌우 측면에 가스를 분사하여 모멘트를 발생시키는 상황 가정

        Eqn. of motion :

            (x_dot, y_dot, V_dot, psi_dot) = (V_cospsi, V_sinpsi, 0, F/Vm), (Input : F)
            

    (3) case 2 : Deflection에 의한 Torque

        condition : C_nr = 0, none sideslip

        Eqn. of motion : 

            psi_ddot = +-0.3

            (x_dot, y_dot, V_dot, psi_dot) = (V_cospsi, V_sinpsi, 0, psi_ddot * dt), (Input : Psi_ddot)




2. Fortress Invasion environment
    
    (1) Rewad 기준 : 

        1. 직진(1), 좌우 회전(1.2)
        
        2. 성공(100)
        
        3. 40넘게 움직이면서 wall에 충돌(-20) : 
            
            성공할 경우에도 마찬가지, 보통 30 이하의 움직임으로도 충분히 성공가능하다. 더 빠르게 성공시키기 위해 설정
        
        4. 50넘게 움직일 경우(-40)
            
            허용하는 움직임의 개수가 많아질 수록 화면 내부에서 뱅글뱅글 도는 경우가 생긴다. 
            
            이를 방지하기 위해 허용할 수 있는 움직임의 최대값 설정

    (2) 코드 구성

        1. __init__ : 

            초기설정, state, 상,하한선, seed, viewer, step beyond done

        2. seed : sedd 데이터 반환

        3. step : state 업데이트, step byond done 여부 파악, reward 판정, 충돌감지(wall, fortress)

        4. reset

        5. render

        6. close







