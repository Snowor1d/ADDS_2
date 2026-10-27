"""TRAIN_MAP_SOURCE = "dataset": 고정 번호 맵 목록으로 학습.

에피소드마다 DATASET_MAPS 에서 맵 하나를 균등하게 뽑는다. RV3 의 "300maps"
학습이 이 모드(1000-1299)다.

주의: 1000-1299 중 69개(1231-1299)는 맵 150-249 의 57개와 실제 건물 맵
300/303/306/307 의 사본이다(D4 대칭까지 픽셀 2% 이내). 따라서 이 목록으로
학습한 모델을 그 맵들에서 평가하면 미지 환경 평가가 아니다. RV4 테스트
스위트(data/suites)는 학습 소스와의 중복을 검사해 이런 맵을 제외했다.

상수만 정의한다.
"""

DATASET_MAPS = tuple(range(1000, 1300))
# 맵별 추출 가중치. None 이면 균등.
DATASET_MAP_WEIGHTS = None
# 기하 증강: 에피소드마다 D4 대칭(회전 4 x 반사 2) 중 하나를 맵에 적용.
# 출구·장애물이 함께 변환된다. RV3 는 MAP_DATA_AUGMENTATION=True 로 항상 켜져
# 있었다.
DATASET_AUGMENTATION = True
DATASET_AUGMENTATION_TRANSFORMS = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "reflect", "reflect_rotate_90", "reflect_rotate_180", "reflect_rotate_270",
)
# 에피소드마다 군중 수 (양끝 포함, 균등)
DATASET_CROWD_RANGE = (30, 30)
