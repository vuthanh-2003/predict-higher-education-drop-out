Dự án này tuning hai mô hình ensemble learning là Random Forest và XGBoost để dự đoán sinh viên nghỉ học đại học dựa trên dữ liệu về nhân khẩu học và quá trình học tập.
Bộ dữ liệu sử dụng ở đây là: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention

Cấu trúc của dự án: 

<img width="950" height="340" alt="image" src="https://github.com/user-attachments/assets/6c1dc66a-04a2-41e8-9ad6-be1ca6641a2b" />

Để xây dựng mô hình dự đoán sinh viên bỏ học, chúng ta sử dụng 34 biến chứa thông tin về nhân khẩu học, quá trình học tập và tình hình kinh tế của 4424 quan sát.

<img width="613" height="685" alt="image" src="https://github.com/user-attachments/assets/d5382d8f-7777-45e1-9045-043bb4825304" />

Biến Target nhận 3 giá trị: "Graduate" nếu sinh viên đã tốt nghiệp, "Enrolled" nêu sinh viên đang theo học và "Dropout" nếu sinh viên đã bỏ học.

<img width="725" height="648" alt="image" src="https://github.com/user-attachments/assets/dc0afb3f-6be8-4bc6-8396-e71ba850775c" />

Dự án này tập trung vào việc dự đoán sinh viên bỏ học nên ta sẽ bỏ đi các quan sát chứa thông tin sinh viên đang học. Ta còn 3630 quan sát.

<img width="406" height="674" alt="image" src="https://github.com/user-attachments/assets/521ac556-c53a-4590-8541-374007467aea" />


Phân phối của một số biến:

<img width="1416" height="309" alt="image" src="https://github.com/user-attachments/assets/425ba3ad-c65f-4829-9345-8fb219ffc5e5" />

Ta thấy đa số các biến đều không có phân phối chuẩn.

Trong bước feature engineering. Dự án này sử dụng mutual information để lọc các biến có ít tương quan với biến cần dự đoán là Target. Ta chọn ngưỡng là 0,01 tức chọn các biến mà mutual information giữa chúng và biến Target lớn hơn hoặc bằng 0,01.

<img width="1077" height="540" alt="image" src="https://github.com/user-attachments/assets/5c129c60-ea04-45b7-ad18-cce67725c07a" />
<img width="245" height="594" alt="image" src="https://github.com/user-attachments/assets/4f1aa0a2-f5bc-4d44-8e32-4a855f4e2776" />
<img width="223" height="499" alt="image" src="https://github.com/user-attachments/assets/75caf503-1dda-485c-81f1-7dd8d6f17bd6" />

Các biến được chọn là: app_2nd, app_1st, grade_2nd, garde_1st, tuition_date, eval_2nd, eval_1st, age_enroll, Course, apply_mode, scholarship, enroll_1st, enroll_2nd, Debtor, dad_quali, pre_quali, mom_occu, mom_quali, dad_occu.

<img width="712" height="519" alt="image" src="https://github.com/user-attachments/assets/cbd71fc8-d794-43b9-8d8d-f661976dd761" />

Qua xem xét biểu đồ tương quan giữa một vài biến định lượng. Ta thấy có các cặp biến tương quan mạnh với nhau là (app_2nd, app_1st), (grade_2nd, grade_1st), (eval_2nd, eval_1st), (enroll_2nd, enroll_1st). Ta sẽ gộp mỗi cặp biến này thành 1 biến bằng cách tính trung bình mỗi cặp. Ta có các biến mới là: app_mean, grade_mean, eval_mean, enroll_mean.

<img width="600" height="446" alt="image" src="https://github.com/user-attachments/assets/f1617724-6afd-4dc9-8d6f-01233f83aeb4" />

Ta thấy có sự chênh lệch dữ liệu khi số lượng quan sát nhận giá trị "Graduate" ở biến Target nhiều hơn so với số quan sát nhận giá trị "Dropout", đây là sự mất cân bằng dữ liệu. Trong dự án này ta dùng phương pháp SMOTENC một phương pháp tạo ra dữ liệu để giải quyết vấn đề này. Phương pháp Target Encoding được sử dụng để mã hóa các biến định tính trong dự án này bao gồm: ['tuition_date', 'Course', 'apply_mode', 'scholarship', 'Debtor', 'Gender', 'dad_quali', 'mom_quali', 'pre_quali', 'dad_occu', 'mom_occu'].

<img width="1398" height="198" alt="image" src="https://github.com/user-attachments/assets/cc9c5405-4ed0-48e8-b1de-dfbe0bd5309f" />

* Mô hình XGBoost
  Các hyperparameters được tuning của mô hình XGBoost là:
  *  n_estimators: Số cây quyết định trong mô hình [100, 200, 300]
  *  max_depth: Độ sâu tối đa của cây [3,5,7]
  *  learning_rate: Tỷ lệ học tập [0.01, 0.05, 0.1]
  *  subsample: Tỷ lệ % quan sát được lấy để xây dựng mỗi cây [0.5, 0.8, 1.0]
  *  colsample_bytree: Tỷ lệ % biến được lấy để xây dựng mỗi cây [0.5, 0.8, 1.0]
  
  <img width="321" height="138" alt="image" src="https://github.com/user-attachments/assets/cdc907ad-503c-45b8-b2dd-83d18dc2f3b6" />

Sau GridSearch, ta có mô hình tốt nhất là:

<img width="802" height="26" alt="image" src="https://github.com/user-attachments/assets/7df4cfd2-8a2e-4376-a333-dc168bccb17a" />

<img width="788" height="468" alt="image" src="https://github.com/user-attachments/assets/104270b9-16e1-48ab-bbe5-6d82e73f0346" />

Ta thấy tuition_date, app_mean là 2 biến có độ quan trọng nhất đối với mô hình XGBoost. 

* Mô hình Random Forest
  Các hyperparameters đưuọc tuning của mô hình Random Forest là:
  *  n_estimators: Số cây quyết định trong mô hình [100,200,300]
  *  max_depth: Độ sâu tối đa của mỗi cây [3,5,7]
  *  min_samples_split: Số mẫu tối thiểu có trong một node để node đó có thể tiếp tục phân chia [2,5,10]
  *  min_samples_leaf: Số mẫu tối thiểu có trong một leaf node [1,2,4]
  *  max_features: Số biến được xem xét ngẫu nhiên tại mỗi lần phân chia ['sqrt', 'log2]

<img width="304" height="145" alt="image" src="https://github.com/user-attachments/assets/8366dee7-7665-408f-b4ad-1a3c9bb287d3" />

Hyperparameters tối ưu của mô hình này là:

<img width="842" height="24" alt="image" src="https://github.com/user-attachments/assets/533dae61-d2a4-445d-b497-8852c2b7c660" />

<img width="809" height="603" alt="image" src="https://github.com/user-attachments/assets/37b07553-d4fd-4ad0-8e12-c7b14a25dded" />

Khá giống với XGBoost, hai biến quan trọng nhất là tuition_date và app_mean

<img width="291" height="296" alt="image" src="https://github.com/user-attachments/assets/e0dde601-5852-4770-8523-3123b03d6692" />

<img width="801" height="603" alt="image" src="https://github.com/user-attachments/assets/9f4e32b2-33ad-40ef-a723-16ac9fb5ec16" />

Ta thấy hiệu suất của 2 mô hình gần như là tương đương nhau, XGBoost có hiệu suất nhỉnh hơn một chút với bộ dữ liệu này.
