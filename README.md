===========================================================================

Cài đặt
-----------------
1. Download repo.
```
git clone https://github.com/NGANnganngan/projectML.git
```

2. Cài đặt môi trường cần thiết (khuyến nghị sử dụng anaconda).
```
conda create -n ENVNAME python=3.11.4
```
Trong đó : 
* `ENVNAME`: Tên môi trường
* khuyến nghị sử dụng python 3.11.4

3. Cài thư viện cần thiết
```
pip install -r requirement.txt
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install chardet
```
Lưu ý: 
* project sử dụng CUDA nên yêu cầu GPU NVIDIA trong quá trình hoạt động
* project cài đặt CUDA phiên bản 11.8

4. Tải file pretrained theo link sau và đưa file vào folder dự án.

https://drive.google.com/file/d/1-5guB3OcJ78a__HNCi0bhTmHg1dQo6PL/view?fbclid=IwAR0iZZZkCcngWySyn4KXCVFvXgBvOKtnwePJ5iw01sTGqhIIFEBuwmI8LyQ

Sử dụng
-----------------
1. Kích hoạt môi trường
```
activate ENVNAME
```
Trong đó : 
* `ENVNAME`: Tên môi trường

2. Chạy lệnh sau, mỗi lệnh trên một terminal riêng biệt
```
python -m http.server 
```
```
python main.py
```
