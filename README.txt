================================================================================
VECTORCAST TEST SCRIPT GENERATOR (PYTHON TOOL)
AUTHOR: SonBB@fpt.com
================================================================================

1. GIỚI THIỆU
-------------
Tool này được viết bằng Python để tự động sinh kịch bản kiểm thử (Test Script .tst) 
cho công cụ VectorCAST dựa trên mã nguồn C.

Các tính năng chính:
- Phân tích cú pháp C (Function, If/Else, Switch, Loop).
- Tự động tính toán logic MC/DC để phủ các nhánh điều kiện phức tạp.
- Xử lý các vòng lặp (Loop) và mảng (Array) thông minh.
- Tự động tạo Stub cho các hàm phụ thuộc và xử lý con trỏ (Pointer).
- Xử lý các Guard Clause (return sớm) để đảm bảo luồng test đi đúng hướng.

2. YÊU CẦU
----------
- Python 3.x đã được cài đặt.
- Các thư viện chuẩn của Python (re, os, argparse, logging) - không cần cài thêm pip.

3. CÁCH SỬ DỤNG (COMMAND LINE)
------------------------------
Mở Terminal (CMD hoặc PowerShell) tại thư mục chứa tool và chạy lệnh theo cú pháp:

  python test_v11.py <c_file> <header_folder> --unit <unit_name> [OPTIONS]

Tham số bắt buộc:
  c_file          : Đường dẫn đến file mã nguồn .c cần test.
  header_folder   : Đường dẫn đến thư mục chứa các file .h (để đọc #define).
  --unit          : Tên của Unit Under Test (UUT) trong VectorCAST.

Tham số tùy chọn:
  --env           : Tên môi trường test (Mặc định: TEST_ENV).
  --output        : Tên file kết quả đầu ra (Mặc định: Result_Final.tst).
  --verbose       : Bật chế độ log chi tiết để debug (Hiện các bước tính toán logic).

4. VÍ DỤ THỰC TẾ
----------------

a) Chạy cơ bản (Header nằm cùng thư mục hiện tại - dùng dấu chấm "."):
   python test_v11.py "vehicle_control.c" . --unit vehicle_control

b) Chỉ định đường dẫn tuyệt đối và tên môi trường:
   python test_v11.py "C:\Project\Src\vehicle_control.c" "C:\Project\Inc" --unit vehicle_control --env TEST_SYS_1

c) Xuất ra file tên khác và bật chế độ Debug (xem log tính toán):
   python test_v11.py "vehicle_control.c" . --unit vehicle_control --output MyTest.tst --verbose
