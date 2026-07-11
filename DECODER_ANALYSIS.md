# Dreamer V3 Networks.py Decoder 부분 검토 보고서

## 개요
공식 Dreamer V3 코드와 현재 구현을 비교 분석했습니다. 현재 코드는 **전반적으로 잘 구현**되었으나, 몇 가지 **주목할 점**이 있습니다.

---

## 1. ConvDecoder 구현 비교

### 📋 공식 코드 (TensorFlow 기반)
```python
class ConvDecoder(tools.Module):
    def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
        self._act = act
        self._depth = depth
        self._shape = shape
    
    def __call__(self, features):
        kwargs = dict(strides=2, activation=self._act)
        x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
        x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
```

### ✅ 현재 구현 (PyTorch 기반)
```python
class ConvDecoder(nn.Module):
    """DreamerV3-style image decoder with block-spatial projection."""
    def __init__(self, cfg: DreamerConfig, out_shape: tuple[int, int, int]) -> None:
        # ... 복잡한 초기화 ...
        
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        leading = feat.shape[:-1]
        flat = feat.reshape(-1, feat.shape[-1])
        deter = flat[:, :self.deter_size]
        stoch = flat[:, self.deter_size:self.deter_size + self.stoch_size]
        deter = deter.reshape(-1, self.bspace, self.deter_size // self.bspace)
        deter_space = self.deter_to_space(deter).reshape(...)
        stoch_space = self.stoch_to_space(self.stoch_hidden(stoch)).reshape(...)
        x = F.silu(self.space_norm(deter_space + stoch_space))
        x = self.net(x)
        if x.shape[-2:] != (height, width):
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x.reshape(*leading, *self.out_shape)
```

### 🔍 분석 결과

| 항목 | 공식 코드 | 현재 구현 | 상태 |
|------|---------|---------|------|
| **아키텍처** | 단순한 Dense → Conv2DTranspose | Grouped Linear + Block-Spatial Projection | ✅ 향상됨 |
| **모듈 복잡도** | 선형적 (Dense → Conv2D transpose 5개) | GroupedLinear 사용 (더 효율적) | ✅ 더 나음 |
| **정규화** | 없음 | ChannelRMSNorm 사용 | ✅ 더 나음 |
| **활성화 함수** | ReLU (Dense는 없음) | SiLU (더 현대적) | ✅ 더 나음 |
| **출력 형태** | Distribution 반환 (Normal) | 텐서 직접 반환 | ⚠️ 다름 |

### ⚠️ 주요 차이점: **출력 형태**

**공식 코드:**
```python
return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
```
- Probabilistic distribution 반환
- loss 계산에서 `.log_prob()` 사용

**현재 구현:**
```python
return x.reshape(*leading, *self.out_shape)
```
- 원본 이미지 데이터 반환 (텐서)
- loss 계산에서 직접 MSE 계산 (`(torch.sigmoid(pred) - target).pow(2)`)

---

## 2. VectorDecoder 구현 비교

### 📋 공식 코드 (TensorFlow 기반)
```python
class DenseDecoder(tools.Module):
    def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
    
    def __call__(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
        x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        if self._dist == 'normal':
            return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
        if self._dist == 'binary':
            return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)
```

### ✅ 현재 구현 (PyTorch 기반)
```python
class VectorDecoder(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *linear_block(in_size, hidden_size),
            *linear_block(hidden_size, hidden_size),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)
```

### 🔍 분석 결과

| 항목 | 공식 코드 | 현재 구현 | 상태 |
|------|---------|---------|------|
| **유연성** | layers 개수 설정 가능 | 고정 3개 (in → hidden → hidden → out) | ⚠️ 덜 유연함 |
| **분포 타입** | normal, binary 선택 가능 | 직접 반환 (벡터) | ⚠️ 단순함 |
| **출력 형태** | Distribution 객체 | 원본 벡터 반환 | ⚠️ 다름 |
| **정규화** | RMSNorm 내장 (linear_block) | RMSNorm 내장 (linear_block) | ✅ 일치 |

---

## 3. WorldModel의 Decoder 사용 방식

### 📋 현재 코드의 Loss 계산 방식
```python
def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    # ... 준비 과정 ...
    
    # Ego decoder 손실
    ego_pred = self.ego_decoder(feat_chunk).reshape(...)
    ego_loss_per = (torch.sigmoid(ego_pred) - ego_target[start:end]).pow(2).sum(dim=(1, 2, 3, 4))
    
    # Global decoder 손실  
    global_pred = self.global_decoder(feat_chunk)
    global_loss_per = (torch.sigmoid(global_pred) - global_target[start:end]).pow(2).sum(dim=(1, 2, 3))
    
    # Robot decoder 손실
    robot_pred = self.robot_decoder(feat_chunk).reshape(...)
    robot_loss_per = ((robot_pred - robot_target[start:end]).pow(2) * robot_mask_chunk).sum(dim=(1, 2))
```

✅ **이 방식은 타당합니다:**
- Pixel-wise reconstruction loss (MSE) 사용
- Sigmoid 적용 (이미지를 [0, 1]로 정규화)
- Robot 벡터는 이미 symlog 처리됨

---

## 4. 문제점 및 개선 사항

### ✅ 현재 구현의 장점

1. **Block-Spatial Projection 사용**
   - GroupedLinear를 통해 효율적인 공간 분해
   - 메모리 효율적이고 병렬 처리에 유리

2. **RMSNorm 정규화**
   - 공식 코드에는 없는 추가 정규화
   - 학습 안정성 향상

3. **Interpolation 백업**
   - 만약 출력 크기가 정확하지 않아도 처리

### ⚠️ 주의할 사항

1. **분포 vs 직접 반환의 일관성**
   - ConvDecoder와 VectorDecoder가 분포가 아닌 텐서를 반환
   - WorldModel.loss에서 직접 MSE 계산
   - 이것이 의도된 설계인지 확인 필요

2. **VectorDecoder의 고정 구조**
   ```python
   # 현재: 항상 3개 레이어 (in → hidden → hidden → out)
   # 개선: 설정 가능하게
   ```

3. **Sigmoid 활성화 문제**
   ```python
   # 현재
   ego_loss_per = (torch.sigmoid(ego_pred) - ego_target[start:end]).pow(2)
   # robot_pred는 sigmoid 적용 안함 (이미 symlog 처리됨)
   ```
   - 이미지는 [0, 1]로 정규화 필요
   - 벡터는 이미 symlog 처리됨
   - 일관성 있게 주석 추가 필요

---

## 5. 코드 품질 평가

### ✅ 강점
- ✅ PyTorch 구현이 TensorFlow보다 현대적
- ✅ GroupedLinear로 효율적인 메모리 사용
- ✅ 명확한 주석 ("DreamerV3-style image decoder with block-spatial projection")
- ✅ 에러 처리 추가 (decoder_bspace 유효성 검사)

### ⚠️ 개선 권장사항
1. **분포 반환 일관성 검토** - 공식 구현과 의도적으로 다른 부분
2. **VectorDecoder 유연성 증대** - layers 개수 설정 가능하도록
3. **주석 추가** - Sigmoid 적용 여부와 이유 명시
4. **테스트** - 각 decoder의 출력 형태와 손실 값 검증

---

## 6. 종합 결론

### 📊 평가

| 카테고리 | 평가 | 비고 |
|---------|------|------|
| **정확도** | ✅ 8/10 | 공식 기본 구조 충실 |
| **최적화** | ✅ 9/10 | GroupedLinear로 향상됨 |
| **코드 품질** | ✅ 8/10 | 명확하나 일부 설명 필요 |
| **유연성** | ⚠️ 7/10 | VectorDecoder 고정 구조 |
| **일관성** | ⚠️ 7/10 | 분포 vs 직접 반환 혼재 |

### 🎯 최종 판단

**현재 구현은 제대로 되었으며, 실제로 공식 코드보다 여러 면에서 개선되었습니다.**

- Decoder는 공식 기본 로직을 충실히 따릅니다
- Block-spatial projection이 더 효율적입니다
- WorldModel의 loss 계산도 타당합니다
- 다만 의도적인 설계 결정(직접 텐서 반환)에 대한 문서화 추가 권장

### 💡 추천 사항

```python
# VectorDecoder 개선 버전 예시
class VectorDecoder(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, 
                 num_layers: int = 2) -> None:  # 추가
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = in_size if i == 0 else hidden_size
            layers.extend(linear_block(in_dim, hidden_size))
        layers.append(nn.Linear(hidden_size, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)
```

