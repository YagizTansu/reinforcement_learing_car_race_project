# Proje Detaylı Anlatım — Adım Adım

---

## Projenin Genel Amacı

Bu bir **üniversite Pekiştirmeli Öğrenme (Reinforcement Learning)** projesi. Amaç:

> Bir arabanın 2D bir yarış pistinde **minimum sürede** tur atmasını öğretmek.

Bunun için **PPO (Proximal Policy Optimization)** algoritması kullanılıyor. Ve asıl araştırma sorusu şu:

> **Policy (karar) ağının boyutu** nasıl etkiler:
> 1. Ulaşılan son performansı?
> 2. Kaç bölümde öğreniyor (yakınsama hızı)?
> 3. Ne kadar sürede öğreniyor (duvar saati süresi)?

---

## Mimari — 7 Modül, 1 Proje

```
src/
  track.py       → Pist oluşturma (geometri)
  car.py         → Arabanın fizik motoru
  frenet.py      → Koordinat dönüşümü
  env.py         → Gymnasium ortamı (RL'nin kalbi)
  render.py      → Görselleştirme
  train.py       → Eğitim başlatıcı (CLI)
  evaluate.py    → Eğitilmiş modeli test et
  plots.py       → Sonuç grafikleri
```

---

## Bölüm Sırası

| # | Bölüm | Dosya | Konu |
|---|-------|-------|------|
| 1 | **Pist** | `track.py` | Spline, yay uzunluğu, eğrilik |
| 2 | **Araba** | `car.py` | Kinematik bisiklet modeli, fizik |
| 3 | **Frenet Koordinatları** | `frenet.py` | Pistye göre konum |
| 4 | **RL Ortamı** | `env.py` | Gözlem, aksiyon, ödül, bölüm |
| 5 | **Eğitim** | `train.py` | PPO, hiperparametreler, kayıt |
| 6 | **Değerlendirme** | `evaluate.py` | Deterministik test |
| 7 | **Grafikler** | `plots.py` | Sonuçların analizi |

---

## Bölüm 1 — `track.py`: Pist Oluşturma

### Amaç
Bir kapalı döngü yarış pisti oluşturmak. Sadece birkaç kontrol noktası ver, gerisi otomatik hesaplanıyor.

### Pipeline (4 adım)
```
Kontrol Noktaları → Periyodik Kübik Spline → Uniform Yeniden Örnekleme → Teğet & Eğrilik
```

**1. Kontrol noktaları:** 8–12 adet (x, y) nokta. Monaco pistini taklit eden sabit bir set var, ya da her bölümde rastgele oluşturuluyor.

**2. Periyodik Kübik Spline:** Bu noktaları pürüzsüz bir kapalı eğriye dönüştürüyor. "Periyodik" demek: son nokta ilk noktayla tam olarak birleşiyor, yani sonsuz dönüş yapılabilir. `scipy.interpolate.CubicSpline` kullanılıyor, `bc_type='periodic'`.

**3. Uniform yeniden örnekleme:** Spline boyunca her **2 metre**'de bir nokta alınıyor (arc-length spacing). Bu çok önemli çünkü yavaşça gidersen pek çok nokta, hızlı gidersen az nokta alırsın — bunu önlemek için uniform örnekleme var.

**4. Teğet ve eğrilik hesabı:**
- **Teğet vektör:** Her noktada hareket yönü. Sonlu fark yöntemiyle:

$$\hat{t}_i = \frac{p_{i+1} - p_{i-1}}{|p_{i+1} - p_{i-1}|}$$

- **İşaretli eğrilik:** Menger formülü. Pozitif = sola dönüş, negatif = sağa dönüş.

$$\kappa_i = \frac{2 \cdot \text{cross}(p_i - p_{i-1},\ p_{i+1} - p_i)}{|p_i - p_{i-1}| \cdot |p_{i+1} - p_i| \cdot |p_{i+1} - p_{i-1}|}$$

### `Track` dataclass — Ne saklıyor?

```python
Track:
  points           # (N, 2) — merkez hattı noktaları [x, y] (metre)
  cum_arc_length   # (N,)   — s_0=0, s_i = baştan i. noktaya kadar uzunluk
  tangents         # (N, 2) — birim teğet vektörler
  signed_curvature # (N,)   — işaretli eğrilik (m⁻¹)
  half_width       # 6.0 m  — pistin yarı genişliği (sabit)
  total_length     # toplam uzunluk (metre)
```

### Public fonksiyonlar
| Fonksiyon | Açıklama |
|-----------|----------|
| `track()` | Sabit pisti oluşturur ve döndürür |
| `build_track(ctrl_pts)` | Verilen kontrol noktalarından Track oluşturur |

### Sabitler
| Sabit | Değer | Açıklama |
|-------|-------|----------|
| `HALF_WIDTH` | 6.0 m | Pistin yarı genişliği |
| `RESAMPLE_DS` | 1.0 m | Örnekleme aralığı |
| `CURVATURE_SCALE` | 20.0 | Eğrilik normalize çarpanı (state vektöründe) |

---

## Bölüm 2 — `car.py`: Kinematik Bisiklet Modeli

### Amaç
Arabanın fizik motorudur. Her `step()` çağrısında arabaya **ivme** ve **direksiyon açısı** veriyorsun, bir sonraki konumu, yönü ve hızı hesaplayıp yeni `CarState` döndürüyor.

### Neden "bisiklet modeli"?
Araba 4 tekerlekli ama model bunu 2 tekerleğe indirgeyerek basitleştirir:
- Ön tekerlekler → tek bir ön tekerlek
- Arka tekerlekler → tek bir arka tekerlek

RL için yeterince gerçekçi, hesabı basit.

### `CarState` dataclass

```python
CarState:
  x        # Kartezyen x konumu (metre)
  y        # Kartezyen y konumu (metre)
  heading  # Yaw açısı (radyan, +x ekseninden CCW)
  v        # İleri hız (m/s), her zaman [0, v_max] aralığında
```

### Fiziksel sabitler (değiştirilemez)

| Sabit | Değer | Açıklama |
|-------|-------|----------|
| `V_MAX` | 30.0 m/s | Maksimum hız (~108 km/h) |
| `A_MAX` | 8.0 m/s² | Maksimum ivme/fren |
| `STEER_MAX` | 0.45 rad | Maksimum direksiyon açısı (~25.8°) |
| `WHEELBASE` | 3.0 m | Dingil mesafesi (L) |
| `DT` | 0.05 s | Simülasyon adımı (= 20 Hz) |

### `step()` — Denklemler (Forward Euler)

**1. Kayma açısı (beta) — kütle merkezindeki slip angle:**

$$\beta = \arctan\!\left(\tan(\delta) \cdot \frac{l_r}{L}\right), \quad l_r = L/2$$

**2. Konum güncelleme:**

$$x' = x + v \cdot \cos(\psi + \beta) \cdot dt$$
$$y' = y + v \cdot \sin(\psi + \beta) \cdot dt$$

**3. Yaw güncelleme:**

$$\psi' = \psi + \frac{v}{l_r} \cdot \sin(\beta) \cdot dt$$

**4. Hız güncelleme:**

$$v' = \text{clip}(v + a \cdot dt,\ 0,\ v_{\max})$$

### Neden Forward Euler?
$dt = 0.05$ s çok küçük olduğu için yeterince doğru. RK4 gibi daha hassas yöntemlere gerek yok.

### Referans
Kong et al., *"Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design"*, IV 2015.

---

## Bölüm 3 — `frenet.py`: Frenet Koordinatları

### Problem: Neden Kartezyen koordinatlar yetmez?
`(x, y)` verince ajan "pistin ortasında mıyım, kenarında mıyım?" bilemez. Frenet sistemi soruyu değiştirir: **"Piste göre neredesin?"**

### Frenet çerçevesi — 3 sayı: `(s, d, theta_e)`

| Değişken | Anlamı | Tipik Aralık |
|----------|--------|------|
| `s` | Pist başından arabaya kadar yay uzunluğu (metre) | [0, ~1046] |
| `d` | Merkez hattından yanal sapma. Pozitif = sol | [-6, +6] |
| `theta_e` | Arabanın yönü − pist teğetinin yönü (radyan) | [-π, π] |

### 3 Fonksiyon

**1. `closest_point_index(track, position)`**
- Tam tarama: O(N) — tüm noktalara bakar
- Warm-start: önceki indeks verilirse ±%10 pencere → çok daha hızlı. `env.py` her adımda bunu kullanır.

**2. `to_frenet(track, position, heading)`**
Adım adım:
1. En yakın merkez hattı noktası `p*` → indeks `i*`
2. `s = cum_arc_length[i*]`
3. Normal vektör: teğeti 90° CCW döndür → `n = (-t_y, t_x)`
4. `d = dot(position − p*, n)` — sol = pozitif, sağ = negatif
5. `theta_e = heading − arctan2(t_y, t_x)`, `[-π, π]`'ye wrap

**3. `lookahead_curvatures(track, s, distances)`**
Ajanın **ileriye bakması** için. 10 mesafede `[5,10,15,20,30,40,55,70,90,110]` m ötedeki eğrilikleri döndürür. Bu 10 değer `obs[4:14]`'ü oluşturur — ajan sayesinde viraj öncesi yavaşlamayı öğrenebilir.

```python
target_s = (s + distances) % total_length  # pist sonunda wrap
indices  = round(target_s / ds) % N
return track.signed_curvature[indices]
```

### Neden `s` state vektöründe yok?
`s` "pistin kaçıncı metresindeyim" sorusunu cevaplar — ama bu bilgi farklı pistlere genellenemez. `d` ve `theta_e` ise pistten bağımsız, evrensel bilgilerdir. Bu **generalization** sağlar.

---

## Bölüm 4 — `env.py`: RL Ortamı

*(Bölüm 3 tamamlandığında burası doldurulacak)*

---

## Bölüm 5 — `train.py`: Eğitim

*(Bölüm 4 tamamlandığında burası doldurulacak)*

---

## Bölüm 6 — `evaluate.py`: Değerlendirme

*(Bölüm 5 tamamlandığında burası doldurulacak)*

---

## Bölüm 7 — `plots.py`: Grafikler ve Analiz

*(Bölüm 6 tamamlandığında burası doldurulacak)*
