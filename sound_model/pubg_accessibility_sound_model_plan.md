# PUBG 접근성 보조용 사운드 이벤트 분류 모델 개발 계획서

작성일: 2026-07-06
목적: 청각적으로 어려움이 있거나 소리를 들을 수 없는 환경에서 PUBG: BATTLEGROUNDS를 플레이하는 사용자가 **게임 내 소리 정보를 시각·촉각 신호로 대체해서 이해할 수 있도록 돕는 접근성 보조 모델**을 설계한다.

---

## 1. 기본 전제

이 계획서는 다음 전제를 기준으로 한다.

```text
1. 배그 인게임 음성 채팅은 사용하지 않는다.
2. 팀 음성 채팅은 Discord 같은 외부 프로그램을 사용한다.
3. Python/모델 입력에는 PUBG 게임 오디오만 들어오게 만든다.
4. 시스템 전체 소리, 마이크 입력, Discord 출력은 모델 입력에서 제외한다.
5. 게임 클라이언트 메모리, 패킷, 파일, 내부 상태는 읽지 않는다.
6. 자동 조준, 자동 반응, 자동 입력, 적 추적 같은 기능은 만들지 않는다.
7. 오직 사용자가 원래 소리로 들을 수 있는 정보를 시각/촉각으로 바꾸는 접근성 보조에 한정한다.
```

이 프로젝트의 핵심 문제는 더 이상 “음성과 게임 소리 분리”가 아니다.

```text
핵심 문제 = PUBG 게임 효과음끼리 겹쳤을 때 각각의 이벤트를 실시간으로 감지하는 것
```

따라서 모델은 단일 분류 모델이 아니라 **멀티라벨 사운드 이벤트 탐지 모델**로 설계한다.

---

## 2. 접근성 중심 설계 원칙

### 2.1 정보 등가성 원칙

이 시스템은 새로운 정보를 만들어내는 도구가 아니라, **청각 정보를 다른 감각 채널로 변환하는 도구**여야 한다.

허용 목표:

```text
총소리가 들림 → 화면에 총소리 아이콘 표시
왼쪽에서 발소리가 들림 → 왼쪽 방향에 발소리 시각 신호 표시
차량음이 커짐 → 차량 접근 강도 표시
폭발음 발생 → 짧은 화면/진동 신호 표시
```

피해야 할 목표:

```text
게임 메모리를 읽어서 적 위치 표시
소리 이후 적 위치를 계속 추적
미니맵/화면 OCR과 결합해 적 위치 추론
자동으로 조준/사격/핑 입력
패킷 분석 또는 게임 클라이언트 변조
사용자가 들을 수 없는 수준의 정보를 과도하게 증폭
```

### 2.2 최소 개입 원칙

출력 UI는 플레이를 대신해 판단하지 않고, 사용자가 원래 청각으로 받을 수 있는 신호를 짧고 명확하게 제공한다.

```text
좋은 출력: “왼쪽 뒤쪽에서 발소리 가능성 높음”
나쁜 출력: “적이 2층 계단 오른쪽에 있음”
```

### 2.3 사용자 제어 원칙

사용자는 다음 항목을 직접 조절할 수 있어야 한다.

```text
표시할 이벤트 종류
민감도
알림 지속 시간
투명도
색상/명도 대비
진동 강도
깜빡임 사용 여부
방향 표시 방식
```

---

## 3. 참고 가능한 공개 자료

### 3.1 PUBG Gun Sound Dataset / BGG

`PUBG-Gun-Sound-Dataset`은 논문 **Enemy Spotted: In-game Gun Sound Dataset for Gunshot Classification and Localization**의 공식 저장소다. PUBG 인게임 총성 데이터를 기반으로 하며, 총기 분류와 총성 위치 추정을 다룬다.

활용 방향:

```text
총성 클래스 초기 학습
총성 방향/거리 추정 구조 참고
CNN / RNN / CRNN / Transformer 백본 비교 참고
```

한계:

```text
발소리, 차량, 폭발, 문, 장전 등 전체 PUBG 효과음을 포괄하지 않음
총성 중심 데이터셋
연구/비상업적 사용 조건 확인 필요
```

### 3.2 BattleSound

BattleSound는 PUBG 기반 사운드 이벤트 탐지 연구로, 무기음 감지와 음성 채팅 감지를 다룬다. 논문에서는 실시간 피드백 생성을 목표로 한 게임 사운드 벤치마크로 설명된다.

활용 방향:

```text
실시간 사운드 이벤트 탐지 구조 참고
weapon sound event detection 설계 참고
noisy game sound 환경에서의 baseline 참고
```

한계:

```text
VOICE / WEAPON / MIXTURE 중심
접근성용 세부 이벤트 분류에는 추가 데이터 필요
발소리/차량/문/장전 등은 별도 수집 필요
```

### 3.3 범용 오디오 사전학습 모델

초기 모델 성능을 빠르게 확보하려면 AudioSet 기반 사전학습 모델을 feature extractor 또는 teacher model로 사용할 수 있다.

후보:

```text
YAMNet: MobileNet 기반, 521개 AudioSet 이벤트 분류
PANNs: AudioSet 기반 pretrained audio neural networks
AST: Audio Spectrogram Transformer
EfficientAT / MobileNet 계열: 실시간 경량화 후보
```

추천 방식:

```text
대형 사전학습 모델 = teacher / pseudo-labeling / feature extraction
실시간 모델 = 작은 CNN/CRNN/student model
```

---

## 4. 전체 시스템 구조

### 4.1 상위 구조

```text
PUBG 프로세스 오디오 캡처
        ↓
오디오 버퍼링
        ↓
전처리 / feature extraction
        ↓
멀티라벨 사운드 이벤트 탐지 모델
        ↓
시간적 후처리
        ↓
접근성 UI / 촉각 피드백
```

### 4.2 런타임 파이프라인

```text
1. PUBG 오디오만 캡처
2. 48kHz stereo 입력 수신
3. 16kHz 또는 32kHz로 리샘플링
4. 최근 1.0초 circular buffer 유지
5. 100ms 간격으로 sliding window 추론
6. log-mel spectrogram 생성
7. 모델 추론
8. 클래스별 확률 산출
9. hysteresis / smoothing / debounce 적용
10. 시각/촉각 신호 출력
```

### 4.3 권장 지연 시간 목표

```text
빠른 이벤트, 예: 총성/폭발       → 150~400ms 내 반응 목표
연속 이벤트, 예: 발소리/차량     → 300~1000ms 내 안정 감지 목표
UI 갱신 주기                     → 50~100ms
모델 추론 주기                   → 100ms
최대 체감 지연                   → 1초 이하
```

---

## 5. 오디오 캡처 설계

### 5.1 권장 방식

가장 좋은 방식은 **PUBG 프로세스 오디오만 캡처**하는 것이다.

```text
PUBG.exe 출력만 캡처
Discord.exe 출력 제외
마이크 입력 제외
브라우저/음악/시스템 알림 제외
```

Windows에서는 다음 두 방향을 고려한다.

### 방식 A: WASAPI Application Loopback

Windows의 application loopback capture는 특정 프로세스 트리에서 렌더링되는 오디오만 캡처할 수 있다. 이 방식이 가능하면 가장 깔끔하다.

장점:

```text
Discord 소리 제외 가능
시스템 전체 loopback보다 입력이 깨끗함
추가 가상 오디오 라우팅이 줄어듦
```

주의:

```text
Windows 10 build 20348 이상 요구
Python만으로 직접 구현하기 어려울 수 있음
C++ 캡처 helper를 만들고 Python과 IPC로 연결하는 구조가 현실적
```

### 방식 B: 가상 오디오 장치 라우팅

PUBG 출력 장치를 가상 오디오 장치로 보내고, Python이 해당 장치를 입력으로 듣는 방식이다.

예시 구조:

```text
PUBG 출력 → Virtual Audio Cable / VB-CABLE / VoiceMeeter input
Discord 출력 → 실제 헤드셋
Python 입력 → PUBG가 연결된 가상 오디오 장치
사용자 청취 → 가상 장치 모니터링 또는 별도 라우팅
```

장점:

```text
구현 난이도 낮음
Python에서 PyAudio / sounddevice / PyAudioWPatch 등으로 접근 가능
실험용으로 빠르게 시작 가능
```

단점:

```text
사용자 설정이 복잡할 수 있음
장치 선택 오류 가능
가상 장치 지연이 추가될 수 있음
배포 UX가 좋지 않을 수 있음
```

### 5.2 개발 단계별 권장

```text
프로토타입: 가상 오디오 장치 라우팅
MVP: WASAPI loopback 또는 가상 장치 자동 설정 가이드
정식화: 프로세스별 오디오 캡처 helper + Python inference engine
```

---

## 6. 이벤트 클래스 설계

처음부터 너무 많은 클래스를 넣으면 데이터 수집과 라벨링이 어려워진다. 접근성 가치가 높은 순서대로 단계적으로 늘린다.

### 6.1 V0 최소 클래스

```text
none / background
footstep
gunshot
vehicle
explosion
```

목표:

```text
가장 중요한 생존 관련 사운드 감지
실시간 멀티라벨 구조 검증
겹친 소리 대응 검증
```

### 6.2 V1 실사용 클래스

```text
footstep
gunshot
vehicle
explosion
reload / bolt / chamber
throwable
 door / window / glass
parachute / glider
water / swim
zone / redzone / ambient hazard
none / background
```

### 6.3 V2 세부 클래스

```text
footstep_near
footstep_far
gunshot_near
gunshot_far
vehicle_near
vehicle_far
explosion_near
explosion_far
reload_self_or_near
throwable_pin_or_bounce
door_open_close
glass_break
```

### 6.4 권장 출력 클래스 구조

초기에는 세부 종류보다 접근성에 중요한 정보를 우선한다.

```text
event_type: footstep / gunshot / vehicle / explosion / reload / door / throwable
activity: active / onset / offset
direction: left / right / front / back / unknown
distance: near / mid / far / unknown
confidence: 0.0 ~ 1.0
```

정확한 총기명 분류는 후순위로 둔다.

이유:

```text
1. 접근성 목적에는 “총성이 있다”가 “M416이다”보다 중요함
2. 정확한 무기 식별은 공정성 논란이 커질 수 있음
3. 먼저 coarse event를 안정화하는 편이 실용적임
```

---

## 7. 모델 구성 계획

## 7.1 최종 권장 구조

최종적으로는 다음과 같은 **멀티태스크 멀티라벨 사운드 이벤트 탐지 모델**을 목표로 한다.

```text
입력: stereo waveform, 최근 1.0초
        ↓
전처리: log-mel spectrogram + stereo spatial features
        ↓
Backbone: lightweight CNN 또는 CRNN
        ↓
Temporal module: GRU / TCN / lightweight Transformer
        ↓
Heads:
  1. event multi-label head
  2. onset/offset head
  3. direction head
  4. distance/intensity head
        ↓
후처리: smoothing, hysteresis, confidence calibration
```

---

## 7.2 입력 feature

### 기본 feature

```text
sample_rate: 32kHz 권장, 16kHz도 가능
channel: stereo 유지 권장
window_length: 1.0초
hop_length: 0.1초
n_fft: 1024 또는 2048
mel_bins: 64 또는 96
feature: log-mel spectrogram
```

16kHz를 쓰면 계산량이 줄어든다. 다만 PUBG의 일부 효과음은 고주파 질감이 구분에 도움이 될 수 있으므로 32kHz도 비교한다.

### stereo feature

방향 추정을 고려하면 stereo 정보를 버리면 안 된다.

추천 입력 채널:

```text
left_logmel
right_logmel
mid_logmel = (L + R) / 2
side_logmel = (L - R) / 2
interaural_level_difference, ILD
interaural_time_difference 후보 feature
```

초기에는 단순하게 다음 4채널부터 시작한다.

```text
[L_logmel, R_logmel, Mid_logmel, Side_logmel]
```

---

## 7.3 모델 후보

### 후보 A: Lightweight CNN, V0 추천

```text
log-mel 입력
→ depthwise separable CNN blocks
→ global pooling
→ sigmoid multi-label output
```

장점:

```text
빠름
구현 쉬움
실시간 추론에 유리
기준선 만들기 좋음
```

단점:

```text
발소리처럼 시간 패턴이 중요한 이벤트에서 한계
onset/offset 탐지가 약할 수 있음
```

사용 시점:

```text
첫 번째 프로토타입
데이터셋 검증
실시간 파이프라인 검증
```

---

### 후보 B: CRNN, V1 추천

```text
log-mel 입력
→ CNN encoder
→ BiGRU 또는 GRU/TCN temporal module
→ frame-wise event probability
→ pooling 또는 sequence output
```

장점:

```text
발소리, 차량음, 연속 이벤트에 강함
시간별 이벤트 탐지가 가능
겹친 소리 처리에 유리
```

단점:

```text
CNN보다 복잡함
튜닝 필요
```

사용 시점:

```text
멀티라벨 이벤트 탐지 본 모델
실시간 접근성 UI 연결 단계
```

---

### 후보 C: Transformer / AST fine-tuning

```text
log-mel spectrogram
→ patch embedding
→ Transformer encoder
→ multi-label output
```

장점:

```text
성능 잠재력 큼
긴 문맥 파악 가능
사전학습 모델 활용 가능
```

단점:

```text
실시간 배포에는 무거울 수 있음
데이터가 적으면 과적합 가능
```

사용 시점:

```text
teacher model
pseudo-label 생성
오프라인 성능 상한선 측정
student model distillation
```

---

### 후보 D: PANNs / YAMNet 기반 transfer learning

```text
waveform 또는 log-mel 입력
→ pretrained audio encoder
→ custom classification head
```

장점:

```text
데이터가 적을 때 유리
빠르게 baseline 확보 가능
pseudo-labeling에 좋음
```

단점:

```text
PUBG 특화 소리에 최적화되어 있지는 않음
방향 추정에는 별도 설계 필요
```

사용 시점:

```text
초기 라벨링 보조
teacher model
소량 데이터 fine-tuning baseline
```

---

## 7.4 최종 추천 모델 로드맵

```text
1단계: log-mel + lightweight CNN + multi-label sigmoid
2단계: log-mel stereo 4ch + CRNN + frame-wise SED
3단계: PANNs/AST teacher로 pseudo-labeling 및 성능 비교
4단계: teacher-student distillation으로 경량 실시간 모델 제작
5단계: direction/distance head 추가
```

---

## 8. 학습 목표와 loss 설계

### 8.1 기본 멀티라벨 분류

각 클래스가 독립적으로 존재할 수 있으므로 softmax가 아니라 sigmoid를 사용한다.

```text
output = sigmoid(logits)
loss = BCEWithLogitsLoss
```

예시:

```text
footstep: 1
gunshot: 1
vehicle: 0
explosion: 0
reload: 0
```

### 8.2 클래스 불균형 대응

PUBG 사운드는 배경음이 많고 이벤트는 짧게 발생한다. 특히 장전, 유리 깨짐, 투척물 소리는 적을 수 있다.

대응 방법:

```text
positive class weighting
focal loss
asymmetric loss
hard negative mining
class-balanced sampling
```

### 8.3 frame-wise SED loss

이벤트의 시작/끝을 감지하려면 100ms 단위 프레임 라벨을 사용한다.

```text
입력: 1초 audio window
출력: 10개 frame × class_count
프레임 단위 BCE loss
```

예시:

```text
0.0~0.1s: background
0.1~0.3s: footstep
0.3~0.5s: background
0.5~0.7s: gunshot
0.7~1.0s: vehicle
```

### 8.4 방향 loss

방향은 처음부터 연속 각도 회귀로 가지 말고, coarse sector 분류로 시작한다.

```text
left
right
front
back
front-left
front-right
back-left
back-right
unknown
```

loss:

```text
CrossEntropyLoss for direction sector
단, event confidence가 높은 frame에만 direction loss 적용
```

### 8.5 거리/intensity loss

거리 역시 정확한 미터 단위 회귀보다 coarse 분류가 안전하다.

```text
near
mid
far
unknown
```

거리 추정은 오디오 설정, 볼륨, EQ, 헤드셋, 게임 내 믹싱의 영향을 크게 받으므로 “절대 거리”보다 “청각적 강도”로 표현하는 편이 더 안정적이다.

---

## 9. 데이터셋 구축 계획

## 9.1 데이터 소스

### 공개 데이터

```text
BGG / PUBG Gun Sound Dataset: 총성, 방향, 거리 관련 참고
BattleSound: weapon sound event detection 구조 참고
AudioSet 기반 pretrained model: transfer learning 참고
```

### 자체 수집 데이터

접근성 모델에는 자체 수집 PUBG 게임 오디오가 반드시 필요하다.

권장 수집 환경:

```text
훈련장
커스텀 매치
리플레이
일반 매치 녹화
다양한 맵
다양한 지형
다양한 건물 내부/외부
다양한 사운드 설정
다양한 거리/방향
```

수집 시 원칙:

```text
게임 오디오만 녹음
Discord/마이크 음성 제외
개인정보/음성 포함 방지
파일명과 메타데이터 정리
비상업적 연구/접근성 목적 명시
사용 허가/약관 검토
```

---

## 9.2 라벨링 단위

### clip-level label

```text
1초 클립 전체에 어떤 이벤트가 있었는지 표시
```

장점:

```text
라벨링 쉬움
초기 모델에 적합
```

단점:

```text
이벤트 발생 시점이 부정확함
실시간 UI 반응이 둔해질 수 있음
```

### frame-level label

```text
100ms 단위로 이벤트 존재 여부 표시
```

장점:

```text
이벤트 시작/끝 탐지 가능
실시간 UI에 적합
```

단점:

```text
라벨링 비용 큼
```

추천:

```text
V0: clip-level multi-label
V1: 주요 클래스만 frame-level
V2: 전체 클래스 frame-level + direction/distance
```

---

## 9.3 권장 라벨 스키마

```yaml
clip_id: match_0001_012345
source: training_ground
map: erangel
sample_rate: 48000
channels: stereo
audio_settings:
  hrtf: on
  master_volume: 100
  effects_volume: 100
  voice_volume: 0
labels:
  - event: footstep
    start: 0.23
    end: 0.48
    direction: front_left
    distance: near
    confidence: verified
  - event: gunshot
    start: 0.61
    end: 0.70
    direction: right
    distance: far
    confidence: verified
```

---

## 9.4 데이터 split 원칙

랜덤으로 클립만 섞으면 같은 매치/같은 환경의 소리가 train/test에 동시에 들어가서 성능이 과대평가될 수 있다.

권장 split:

```text
match/session 기준 분리
map 기준 일부 분리
날짜 기준 분리
오디오 장치/EQ 설정 기준 일부 분리
```

예시:

```text
train: session 001~070
valid: session 071~085
test: session 086~100
```

---

## 10. 데이터 증강 계획

겹친 소리를 잘 처리하려면 실제 겹침 데이터와 합성 겹침 데이터를 모두 사용한다.

### 10.1 Mix augmentation

```text
footstep + gunshot
footstep + vehicle
gunshot + explosion
vehicle + gunshot
reload + footstep
door + footstep
throwable + explosion
```

라벨은 합쳐진다.

```text
audio = footstep + gunshot
label = {footstep: 1, gunshot: 1}
```

### 10.2 SNR augmentation

소리 크기 비율을 다양하게 바꾼다.

```text
총소리 큼 + 발소리 작음
차량음 큼 + 발소리 작음
배경음 큼 + 장전음 작음
```

### 10.3 거리/공간감 augmentation

```text
volume scaling
high-frequency rolloff
reverb
stereo panning
left/right balance 변화
EQ 변화
```

### 10.4 실사용 환경 augmentation

```text
게임 볼륨 차이
헤드셋 EQ 차이
Windows loudness equalization on/off
압축/녹화 코덱 열화
프레임 드랍에 따른 오디오 버퍼 변동
```

주의:

```text
augmentation이 실제 PUBG 사운드 특성을 망가뜨릴 정도로 과하면 안 됨
방향 추정 학습 시 panning augmentation은 실제 HRTF와 충돌할 수 있음
```

---

## 11. 실시간 추론 설계

## 11.1 기본 sliding window

```text
window_size = 1.0 sec
hop_size = 0.1 sec
```

동작 예시:

```text
0.0~1.0초 분석 → 1.0초 근처 출력
0.1~1.1초 분석 → 1.1초 근처 출력
0.2~1.2초 분석 → 1.2초 근처 출력
```

### 11.2 빠른 이벤트용 보조 branch

총성, 폭발은 1초 전체를 기다리지 않아도 된다.

추천 구조:

```text
Fast branch: 0.25~0.5초 window, 총성/폭발/onset 감지
Stable branch: 1.0초 window, 발소리/차량/장전 안정 감지
```

결합 방식:

```text
총성/폭발 = fast branch 우선
발소리/차량 = stable branch 우선
최종 출력 = 두 branch confidence fusion
```

### 11.3 후처리

모델 확률은 프레임마다 흔들릴 수 있으므로 그대로 UI에 표시하면 피로감이 커진다.

필수 후처리:

```text
moving average smoothing
hysteresis threshold
debounce
minimum event duration
cooldown
confidence calibration
```

예시:

```text
footstep_on_threshold = 0.65
footstep_off_threshold = 0.35
minimum_duration = 200ms
cooldown = 100ms
```

---

## 12. 접근성 UI / 피드백 설계

## 12.1 시각 UI

권장 UI는 화면 가장자리의 방향성 링이다.

```text
왼쪽 소리      → 화면 왼쪽 가장자리 pulse
오른쪽 소리    → 화면 오른쪽 가장자리 pulse
전방 소리      → 화면 상단 중앙 pulse
후방 소리      → 화면 하단 중앙 pulse
소리 강도      → pulse 크기 또는 투명도
이벤트 종류    → 작은 아이콘 또는 패턴
```

### 이벤트별 표시 예시

```text
footstep: 짧고 반복적인 작은 pulse
gunshot: 날카로운 순간 flash
vehicle: 부드럽고 지속적인 bar
explosion: 넓은 원형 shock pulse
reload: 작은 gear/magazine 아이콘
door: 문 아이콘 짧은 표시
```

### 색상 설계

색상만으로 구분하지 않는다.

```text
색상 + 모양 + 위치 + 움직임 패턴 조합
고대비 모드 제공
색각 다양성 고려
깜빡임 강도 제한
```

## 12.2 촉각 피드백

가능하면 게임패드, 웨어러블, 휴대폰 진동 등을 통해 촉각 신호를 제공한다.

예시:

```text
왼쪽 발소리 → 왼쪽 진동 모터 약하게 반복
오른쪽 총성 → 오른쪽 진동 모터 짧고 강하게
차량 접근 → 양쪽 저주파 진동 지속
폭발 → 짧은 강한 양쪽 진동
```

주의:

```text
진동이 많으면 피로해짐
사용자가 이벤트별로 끌 수 있어야 함
긴 세션에서 손목/손 피로 테스트 필요
```

## 12.3 UI 출력 제한

공정성과 접근성을 위해 다음 제한을 둔다.

```text
이벤트 지속 표시 시간 제한
소리 발생 후 오래된 위치 추적 금지
지도/미니맵 위 적 위치 표시 금지
정확한 미터 거리 표시 지양
정확한 총기명 표시는 후순위 또는 옵션화
confidence 낮은 신호는 표시하지 않거나 흐리게 표시
```

---

## 13. 평가 지표

## 13.1 모델 성능 지표

```text
per-class precision
per-class recall
per-class F1
macro F1
micro F1
mAP / average precision
event-based F1
segment-based F1
false positives per minute
false negatives per minute
```

## 13.2 실시간 지표

```text
end-to-end latency
average inference time
P95 inference time
CPU usage
GPU usage
memory usage
audio dropout rate
UI update stability
```

## 13.3 겹친 소리 평가

별도 테스트셋을 만든다.

```text
single_event_test
2_event_overlap_test
3_event_overlap_test
low_snr_test
far_sound_test
high_noise_test
new_map_test
new_audio_setting_test
```

특히 다음 조합을 반드시 평가한다.

```text
footstep + gunshot
footstep + vehicle
footstep + explosion
gunshot + vehicle
reload + footstep
door + footstep
```

## 13.4 접근성 평가

청각적 접근성이 필요한 사용자에게 실제 도움이 되는지를 별도로 검증한다.

평가 항목:

```text
이벤트 인지율
방향 인지율
반응 시간
UI 피로도
오탐으로 인한 방해 정도
사용자 신뢰도
설정 이해도
장시간 사용 가능성
```

정량/정성 평가를 같이 한다.

```text
정량: 감지율, 반응 시간, 오탐 빈도
정성: 사용자 인터뷰, 불편 항목, 선호 UI 패턴
```

---

## 14. 개발 단계별 로드맵

## Phase 0: 범위 확정과 안전 설계

목표:

```text
접근성 목적과 비목적 명확화
약관/공정성 위험 검토
오디오 캡처 방식 결정
이벤트 taxonomy 초안 작성
```

산출물:

```text
scope.md
fair_play_boundary.md
event_taxonomy_v0.yaml
audio_capture_design.md
```

완료 기준:

```text
게임 메모리/패킷/자동입력 미사용 원칙 문서화
PUBG 오디오만 캡처하는 기술 경로 확인
V0 이벤트 클래스 확정
```

---

## Phase 1: 오디오 캡처 프로토타입

목표:

```text
Python 또는 helper program으로 PUBG 오디오만 안정적으로 수집
Discord/마이크가 입력에서 제외되는지 검증
```

작업:

```text
가상 오디오 장치 라우팅 실험
WASAPI application loopback sample 검토
WAV 파일 저장 기능 구현
실시간 circular buffer 구현
오디오 레벨 meter 구현
```

산출물:

```text
capture_prototype.py
record_pubg_audio.py
audio_device_checklist.md
sample_recordings/
```

완료 기준:

```text
PUBG 사운드만 녹음됨
Discord 음성 제외 확인
1시간 이상 녹음 중 dropout 없음
timestamped WAV 저장 가능
```

---

## Phase 2: V0 데이터셋 구축

목표:

```text
최소 클래스 5개 데이터셋 구축
clip-level multi-label 라벨링
```

V0 클래스:

```text
background
footstep
gunshot
vehicle
explosion
```

권장 수량:

```text
각 클래스 최소 500~1000개 positive clip
background 최소 2000개 clip
overlap clip 최소 1000개
```

산출물:

```text
dataset_v0/
metadata.csv
labels_clip_v0.csv
split_train.csv
split_valid.csv
split_test.csv
```

완료 기준:

```text
train/valid/test session 기준 분리
각 클래스 최소 성능 평가 가능
라벨 품질 spot-check 완료
```

---

## Phase 3: V0 모델 학습

목표:

```text
log-mel + lightweight CNN baseline 구축
실시간 추론 가능한지 검증
```

모델:

```text
Input: 1s log-mel
Backbone: small CNN
Output: 5-class sigmoid
Loss: BCEWithLogitsLoss
```

산출물:

```text
train_v0.py
model_cnn_v0.pt
evaluate_v0.py
metrics_v0.json
confusion_analysis.md
```

완료 기준:

```text
major class macro F1 0.80 이상 목표
false positives per minute 측정
100ms hop 실시간 추론 가능
```

---

## Phase 4: V1 멀티라벨 SED 모델

목표:

```text
프레임 단위 이벤트 탐지
겹친 소리 처리 개선
```

모델:

```text
Input: stereo 4ch log-mel
Backbone: CNN encoder
Temporal: GRU 또는 TCN
Output: frame-wise multi-label probabilities
```

산출물:

```text
labels_frame_v1.csv
train_sed_v1.py
model_crnn_v1.pt
evaluate_event_f1.py
```

완료 기준:

```text
겹친 소리 테스트셋 성능 별도 보고
총성/폭발 onset latency 400ms 이하 목표
발소리 false positive 억제
```

---

## Phase 5: 방향/강도 추정 추가

목표:

```text
왼쪽/오른쪽/전방/후방 등 coarse direction 표시
near/mid/far 또는 intensity 표시
```

방향 클래스:

```text
front
front_left
left
back_left
back
back_right
right
front_right
unknown
```

산출물:

```text
direction_labels_v1.csv
train_multitask_v2.py
model_multitask_v2.pt
direction_eval.md
```

완료 기준:

```text
left/right 구분 우선 안정화
front/back은 별도 검증
불확실할 때 unknown 출력 가능
```

---

## Phase 6: 실시간 앱 / 접근성 UI

목표:

```text
모델 추론 결과를 실제 접근성 UI로 표시
```

구성:

```text
audio capture service
inference service
post-processing module
overlay UI
settings UI
optional haptic output
```

산출물:

```text
realtime_app/
settings.json
overlay_ui/
haptic_module/
latency_report.md
```

완료 기준:

```text
1초 이하 체감 지연
UI 끊김 없음
이벤트별 on/off 설정 가능
오탐이 사용자에게 과도한 방해를 주지 않음
```

---

## Phase 7: 사용자 테스트와 개선

목표:

```text
청각 접근성이 필요한 사용자에게 실제 도움이 되는지 검증
```

작업:

```text
사용성 테스트 시나리오 설계
접근성 UI 선호도 조사
오탐/미탐 체감 영향 조사
장시간 사용 피로도 조사
```

완료 기준:

```text
사용자가 이해하기 쉬운 UI 패턴 확정
사용자별 민감도 preset 정의
문제 이벤트 클래스 개선 계획 도출
```

---

## 15. 권장 프로젝트 구조

```text
pubg-accessibility-sound/
├── README.md
├── docs/
│   ├── scope.md
│   ├── fair_play_boundary.md
│   ├── audio_capture_design.md
│   ├── event_taxonomy.md
│   └── labeling_guide.md
├── capture/
│   ├── record_pubg_audio.py
│   ├── wasapi_helper/
│   └── device_check.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── labels/
│   └── splits/
├── models/
│   ├── cnn_baseline.py
│   ├── crnn_sed.py
│   ├── multitask_sed.py
│   └── postprocess.py
├── training/
│   ├── train_v0.py
│   ├── train_sed.py
│   ├── train_multitask.py
│   └── evaluate.py
├── realtime/
│   ├── inference_engine.py
│   ├── audio_buffer.py
│   ├── overlay_bridge.py
│   └── config.yaml
├── ui/
│   ├── overlay/
│   ├── settings/
│   └── assets/
├── tests/
│   ├── test_audio_buffer.py
│   ├── test_preprocess.py
│   └── test_postprocess.py
└── experiments/
    ├── exp001_cnn_v0/
    ├── exp002_crnn_v1/
    └── exp003_multitask_direction/
```

---

## 16. 예시 설정 파일

```yaml
project:
  purpose: accessibility
  input_policy: pubg_audio_only
  prohibit_game_memory_access: true
  prohibit_auto_input: true

capture:
  mode: wasapi_application_loopback
  target_process: TslGame.exe
  sample_rate_in: 48000
  channels: 2

preprocess:
  sample_rate: 32000
  window_sec: 1.0
  hop_sec: 0.1
  n_fft: 1024
  mel_bins: 96
  feature_channels:
    - left_logmel
    - right_logmel
    - mid_logmel
    - side_logmel

model:
  architecture: crnn_sed
  output_type: multilabel_framewise
  classes:
    - footstep
    - gunshot
    - vehicle
    - explosion
    - reload
    - door
    - throwable
    - background

postprocess:
  smoothing_ms: 300
  min_event_duration_ms: 150
  cooldown_ms: 100
  thresholds:
    footstep_on: 0.65
    footstep_off: 0.35
    gunshot_on: 0.75
    gunshot_off: 0.30
    vehicle_on: 0.60
    vehicle_off: 0.40

ui:
  mode: directional_ring
  show_confidence: false
  max_event_display_ms: 1200
  high_contrast: true
  flash_intensity_limit: medium
  haptic_enabled: optional
```

---

## 17. 주요 리스크와 대응

## 17.1 약관/공정성 리스크

리스크:

```text
실시간 보조 도구가 비인가 프로그램 또는 불공정 보조로 해석될 수 있음
```

대응:

```text
게임 클라이언트/메모리/패킷 접근 금지
자동 입력 금지
오디오 캡처 기반 접근성 보조로 한정
시각/촉각 변환 이상의 정보 제공 금지
온라인 사용 전 공식 허용 여부 문의 권장
연구/접근성 테스트 모드 우선
```

## 17.2 오탐 리스크

리스크:

```text
발소리 오탐이 많으면 사용자가 잘못 판단함
```

대응:

```text
confidence threshold 조정
hysteresis 적용
낮은 confidence는 흐리게 표시
사용자별 민감도 preset 제공
false positives per minute를 핵심 지표로 관리
```

## 17.3 미탐 리스크

리스크:

```text
중요 소리를 놓치면 접근성 보조 가치가 떨어짐
```

대응:

```text
중요 클래스 recall 우선 튜닝
footstep/gunshot은 별도 threshold
겹친 소리 테스트셋 강화
실제 사용자 피드백 기반 재학습
```

## 17.4 환경 변화 리스크

리스크:

```text
맵, 지형, 헤드셋, EQ, 게임 패치에 따라 소리가 달라짐
```

대응:

```text
다양한 환경 데이터 수집
audio augmentation
도메인별 validation set 유지
패치 이후 회귀 테스트
사용자 동의 기반 익명 오류 샘플 수집
```

## 17.5 개인정보 리스크

리스크:

```text
음성 채팅이나 개인정보가 녹음될 수 있음
```

대응:

```text
PUBG 프로세스 오디오만 캡처
Discord/마이크 입력 제외 테스트
녹음 파일 자동 익명화
음성 포함 여부 검사
사용자 로컬 저장 우선
외부 업로드는 명시적 동의 필요
```

---

## 18. 바로 시작할 작업 목록

### 1주차

```text
1. 이벤트 taxonomy v0 확정
2. PUBG 오디오만 녹음하는 캡처 방식 실험
3. 10~30분 샘플 오디오 수집
4. footstep/gunshot/vehicle/explosion/background 라벨링 가이드 작성
5. log-mel spectrogram 시각화 노트북 작성
```

### 2주차

```text
1. V0 데이터셋 1000~3000 clip 구축
2. CNN baseline 학습
3. class별 precision/recall/F1 측정
4. 오탐 샘플 분석
5. synthetic overlap augmentation 실험
```

### 3~4주차

```text
1. CRNN 기반 SED 모델 구현
2. 100ms frame-level 라벨 일부 구축
3. 실시간 inference loop 구현
4. overlay UI mockup 제작
5. 지연 시간/CPU/GPU 사용량 측정
```

### 2개월차

```text
1. 데이터 다양화
2. 방향 추정 coarse head 추가
3. 사용자별 민감도 preset 추가
4. 접근성 사용자 테스트 설계
5. 공정성/약관 검토 문서 정리
```

---

## 19. MVP 정의

최소 실사용 가능 버전은 다음 기능을 가진다.

```text
입력:
  PUBG 게임 오디오만 캡처

모델:
  footstep / gunshot / vehicle / explosion / background 멀티라벨 감지
  1초 이하 지연

출력:
  방향성 없는 이벤트 아이콘 또는 간단한 화면 가장자리 pulse
  이벤트별 on/off 가능
  민감도 조절 가능

제한:
  게임 메모리 접근 없음
  자동 입력 없음
  정확한 적 위치 표시 없음
  오래된 소리 추적 없음
```

MVP 이후 방향 표시를 추가한다.

```text
MVP+:
  left/right/front/back coarse direction
  near/mid/far intensity
  haptic feedback
```

---

## 20. 최종 권장 방향

이 프로젝트는 다음 순서로 진행하는 것이 가장 안정적이다.

```text
1. PUBG 오디오만 깨끗하게 캡처한다.
2. 이벤트 클래스를 적게 잡고 V0 모델을 빠르게 만든다.
3. 멀티라벨 구조로 겹친 소리를 처리한다.
4. 발소리/총성/차량/폭발에 집중한다.
5. 실시간 지연과 오탐을 먼저 줄인다.
6. 방향/거리 추정은 후순위로 붙인다.
7. UI는 접근성 중심으로 단순하고 사용자 조절 가능하게 만든다.
8. 온라인 사용이나 배포 전에는 공식 허용 여부와 공정성 이슈를 검토한다.
```

가장 먼저 만들 모델은 다음과 같다.

```text
log-mel spectrogram
+ lightweight CNN
+ sigmoid multi-label output
+ 1초 window / 0.1초 hop
+ footstep, gunshot, vehicle, explosion, background
```

그다음 버전에서 다음으로 확장한다.

```text
stereo 4ch log-mel
+ CRNN frame-wise SED
+ event onset/offset
+ coarse direction
+ visual/haptic accessibility feedback
```

---

## 21. 참고 자료

1. PUBG Gun Sound Dataset / BGG — https://github.com/junwoopark92/PUBG-Gun-Sound-Dataset
2. Enemy Spotted: In-game Gun Sound Dataset for Gunshot Classification and Localization — https://arxiv.org/abs/2210.05917
3. BattleSound: A Game Sound Benchmark for the Sound-Specific Feedback Generation in a Battle Game — https://www.mdpi.com/1424-8220/23/2/770
4. Microsoft Application Loopback Audio Capture Sample — https://learn.microsoft.com/en-us/samples/microsoft/windows-classic-samples/applicationloopbackaudio-sample/
5. Microsoft WASAPI Loopback Recording — https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording
6. Microsoft WASAPI Overview — https://learn.microsoft.com/en-us/windows/win32/coreaudio/wasapi
7. TensorFlow Hub YAMNet Tutorial — https://www.tensorflow.org/hub/tutorials/yamnet
8. PANNs AudioSet Tagging CNN — https://github.com/qiuqiangkong/audioset_tagging_cnn
9. Audio Spectrogram Transformer official implementation — https://github.com/YuanGongND/ast
10. PUBG Rules of Conduct — https://pubg.com/en/clause/rules_of_conduct
