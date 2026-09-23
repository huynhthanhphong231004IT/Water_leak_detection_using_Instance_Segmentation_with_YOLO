<h2 align="center">
  Author: Huynh Thanh Phong (ReoRioll)
</h2>

<p align="center">
   Computer Science of College of Information and Communication Technology of Can Tho University (Course 48)<br>
</p>

<p>

<b>Researchs:</b> Artificial Intelligence in Education - Mathematics in Deep Learning and Machine Learning<br>

<mark><b><b>Name Project:</b></b> </mark> Instance Segmentation and Dynamic Prediction for Water Leaks Using YOLO and Euler Integration. <br>

<mark><b><b>Link Data:</b></b> </mark> https://www.kaggle.com/datasets/reorioll/leakwater-data <br>

</p>
<p align="center">
   <b>Presional link Information</b>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Facbook: https://www.facebook.com/huynh.thanh.phong.561667 <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Kaggle: https://www.kaggle.com/reorioll <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Youtobe: https://www.youtube.com/@ReoRioll-2304CICTCTU <br>
</p>
<br>


## Kết quả kiểm thử trên video giao thông thực tế
<p align="center">
  <img src="Images/Predict_output_img/Predict_output_img_01.jpg" width="600">
  <img src="Images/Predict_output_img/Predict_output_img_02.png" width="600">

  <br>
  <i>Đánh giá hệ thống nhận diện và cảnh báo mức độ rò rỉ của nước.</i>
</p>

```python
!git clone https://github.com/https://github.com/huynhthanhphong231004IT/Water_leak_detection_using_Instance_Segmentation_with_YOLO.git
```

```python
from Code.Value import Value as Val
from Code.Detection import WaterLeak_InstanceSegmentation
from Code.DetectionImage import WaterLeak_ImageSegmentation
if __name__ == "__main__":
    model_path = Val.PATH_MODEL

    # Dự đoán trên hình ảnh
    img_input_path = Val.PATH_INPUT_IMAGE
    img_output_path = Val.PATH_OUTPUT_IMAGE
    WaterLeak_ImageSegmentation.Predict(model_path, img_input_path, img_output_path)

    # Dự đoán trên video
    video_input_path = Val.PATH_INPUT_VIDEO
    video_output_path = Val.PATH_OUTPUT_VIDEO
    WaterLeak_InstanceSegmentation.Predict(model_path, video_input_path, video_output_path)

```

## Biểu thức tổng quan về tích phân số Euler

#### 1. Công thức tính lưu lượng rò rỉ (Q)

$$Q = (A_{\text{pixel}} \times k_{\text{pixelToM2}}) \times C_f \times 1000 \quad \text{(lít/giây)}$$

\- $A_{\text{pixel}}$: Diện tích vết leak nhận diện từ YOLO (pixel).

\- $k_{\text{pixelToM2}}$: Tỷ lệ quy đổi pixel sang $m^2$.

\- $C_f$: Hệ số tốc độ lưu lượng dòng chảy.

\- $1000$: Hằng số quy đổi từ $m^3$ sang lít (L) ($1\text{ m}^3 = 1000\text{ lít}$).

#### 2. Công thức tính tổng thể tích nước tích lũy (V) & Điều kiện cảnh báo

$$V(t) = \int_{0}^{t} Q(\tau) \, d\tau \quad \text{(lít)}$$

Sử dụng <mark>phương pháp tích phân số Euler</mark> để cộng dồn thể tích nước chảy qua từng khoảng thời gian $\Delta t$ giữa các frame.

$$V^{(t)} = V^{(t-1)} + Q(t) \cdot \Delta t \quad \text{(lít)}$$

\- $V^{(t)}$: Tổng thể tích tích lũy tại thời điểm hiện tại.

\- $V^{(t-1)}$: Thể tích tích lũy ở bước/frame trước đó.

\- $\Delta t$: Khoảng thời gian giữa 2 lần nhận diện (`dt = current_time - last_time`).

Điều kiện phát cảnh báo (Alert):

$$\text{Cảnh báo} = \begin{cases} 
\mathbf{TRUE} & \text{khi } V_{\text{tổng}} > V_{\text{ngưỡng}} \\ 
\mathbf{FALSE} & \text{khi } V_{\text{tổng}} \le V_{\text{ngưỡng}} 
\end{cases}$$


\- $V_{\text{ngưỡng}}$: Ngưỡng thể tích giới hạn cho phép (`Spillway_crest`).



