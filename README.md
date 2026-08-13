# Python eKYC Service

Dịch vụ Computer Vision & Biometrics Microservice sản xuất hoàn chỉnh (Production-Ready) cho bài toán **eKYC Căn cước Việt Nam** (Căn cước công dân mẫu cũ & Căn cước mẫu mới 2024).

---

## 🛠️ Công Nghệ & Thư Viện (Tech Stack)
- **Python 3.11+**
- **FastAPI & Uvicorn**
- **PaddleOCR** (Bóc tách OCR Tiếng Việt)
- **OpenCV & NumPy** (Xử lý ảnh & Tính toán ma trận)
- **Pydantic v2 & Pydantic Settings** (Validation & Cấu hình)
- **InsightFace / ArcFace** (Face Detection, 5-point Alignment & 512-d Embedding Vector)
- **QRCodeDetector / pyzbar** (Giải mã QR Căn cước Việt Nam)
- **Anti-Spoofing CNN & Gesture Verification** (Video Liveness Detection)
- **Pytest & HTTPX** (Kiểm thử đơn vị & Tích hợp)

---

## 🚀 Khởi Chạy Ứng Dụng (Running the Service)

### 1. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 2. Thiết lập Biến Môi Trường (Optional)
```bash
cp .env.example .env
```

### 3. Khởi chạy Server Uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ khởi chạy tại: `http://localhost:8000`
Tài liệu Swagger UI API Docs tại: `http://localhost:8000/docs`

---

## 🧪 Kiểm Thử Hệ Thống (Running Tests)

Chạy toàn bộ Unit Tests và Integration Tests qua `pytest`:

```bash
pytest -v
```

---

## 📌 Danh Mục Endpoints API chính

| HTTP Method | Endpoint Path | Chức Năng |
| :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra tình trạng hoạt động dịch vụ |
| `POST` | `/api/v1/ekyc/card` | Bóc tách dữ liệu OCR, QR, MRZ & Validate thẻ |
| `POST` | `/api/v1/ekyc/face/verify` | So khớp khuôn mặt (Ảnh thẻ vs Ảnh Selfie) |
| `POST` | `/api/v1/ekyc/face/liveness` | Kiểm tra thực thể sống từ Video |
| `POST` | `/api/v1/ekyc/verify` | Full Orchestrated eKYC Pipeline |

---

## 🔒 Quy Tắc Chuẩn Hóa Dữ Liệu & Bảo Mật PII
- **Chuẩn hóa 10 trường thông tin duy nhất**:
  1. `identityNumber`
  2. `fullName`
  3. `dateOfBirth`
  4. `gender`
  5. `nationality`
  6. `placeOfBirth`
  7. `placeOfOrigin`
  8. `placeOfResidence`
  9. `dateOfIssue`
  10. `dateOfExpiry`
- **Không sử dụng**: `issueDate`, `expiryDate`.
- **Bảo mật PII**: Hệ thống tự động Masking thông tin cá nhân trên Logs (`001095****01`, `N*** V** A`) và không lưu trữ file tạm của khách hàng.
