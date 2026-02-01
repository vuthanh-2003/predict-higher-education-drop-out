Dự án này tuning hai mô hình ensemble learning là Random Forest và XGBoost để dự đoán sinh viên nghỉ học đại học dựa trên dữ liệu về nhân khẩu học và quá trình học tập.
Bộ dữ liệu sử dụng ở đây là: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention

Để xây dựng mô hình dự đoán sinh viên bỏ học, chúng ta sử dụng 34 biến chứa thông tin về nhân khẩu học, quá trình học tập và tình hình kinh tế của 4424 quan sát.

<img width="613" height="685" alt="image" src="https://github.com/user-attachments/assets/d5382d8f-7777-45e1-9045-043bb4825304" />

Biến Target nhận 3 giá trị: "Graduate" nếu sinh viên đã tốt nghiệp, "Enrolled" nêu sinh viên đang theo học và "Dropout" nếu sinh viên đã bỏ học.

<img width="725" height="648" alt="image" src="https://github.com/user-attachments/assets/dc0afb3f-6be8-4bc6-8396-e71ba850775c" />

Dự án này tập trung vào việc dự đoán sinh viên bỏ học nên ta sẽ bỏ đi các quan sát chứa thông tin sinh viên đang học. Ta còn 3630 quan sát.

<img width="406" height="674" alt="image" src="https://github.com/user-attachments/assets/521ac556-c53a-4590-8541-374007467aea" />

