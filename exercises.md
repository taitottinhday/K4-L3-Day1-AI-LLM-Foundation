# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, phản hồi có xu hướng đa dạng và sáng tạo hơn nhưng cũng ít ổn định hơn; ở 0.0, câu trả lời nhất quán và tập trung nhất. Các mức 0.5–1.0 cân bằng giữa tính tự nhiên và độ tin cậy, còn 1.5 dễ xuất hiện chi tiết bất ngờ hoặc lan man.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi chọn khoảng 0.2–0.3 để câu trả lời ổn định, chính xác và nhất quán với chính sách hỗ trợ, nhưng vẫn đủ tự nhiên khi giao tiếp. Với các nội dung bắt buộc tuyệt đối chính xác, tôi sẽ hạ về gần 0.0.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload tạo khoảng 10.000 × 3 × 350 = 10,5 triệu output token/ngày. Theo bảng giá trong bài, GPT-4o tốn khoảng 105 USD/ngày, còn GPT-4o-mini khoảng 6,30 USD/ngày, nên GPT-4o đắt hơn khoảng 16,7 lần (chỉ xét output). GPT-4o xứng đáng cho phân tích tài liệu phức tạp hoặc quyết định cần chất lượng suy luận cao; mini phù hợp cho FAQ, phân loại và các câu hỏi hỗ trợ đơn giản có lưu lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Persona giáo viên tiểu học tạo câu trả lời ngắn, dùng từ phổ thông và ví dụ như một cuốn sổ được nhiều người cùng giữ. Persona chuyên gia tài chính thường dài hơn và dùng các khái niệm như sổ cái phân tán, cơ chế đồng thuận, mật mã học và tài sản số. System prompt vì vậy không thay đổi câu hỏi, nhưng định hướng rõ đối tượng độc giả, độ sâu, giọng điệu và cách tổ chức câu trả lời của model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với một đoạn tiếng Việt khoảng 100 từ, công thức số từ / 0,75 ước lượng khoảng 133 token, trong khi `tiktoken` có thể cho khoảng 180–220 token tùy nội dung, tức cao hơn chừng 35–65%. Nguyên nhân là tokenizer thường tách từ có dấu hoặc các âm tiết tiếng Việt thành nhiều mảnh token, trong khi nhiều từ tiếng Anh phổ biến đã có sẵn dưới dạng một token hoàn chỉnh. Vì vậy số từ chỉ là phép ước lượng thô và không nên dùng để tính hóa đơn chính xác.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi model tạo câu trả lời dài hoặc có độ trễ cao, vì người dùng thấy nội dung ngay và cảm nhận hệ thống phản hồi nhanh hơn. Non-streaming phù hợp hơn khi kết quả ngắn, cần nhận toàn bộ nội dung trước khi kiểm duyệt/định dạng, hoặc khi dịch vụ phía sau cần một JSON hoàn chỉnh để xử lý thay vì hiển thị từng phần.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giãn nhanh khoảng cách giữa các lần thử, giúp API có thời gian phục hồi và giảm lượng request dồn lên hệ thống đang quá tải. Nếu hàng nghìn client đều chờ cố định một giây, chúng có thể retry đồng thời thành từng đợt lớn (thundering herd) và tiếp tục gây quá tải; trong thực tế nên cộng thêm jitter ngẫu nhiên để phân tán các lần retry.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: “Bạn là trợ giảng thân thiện của khóa học AI. Hãy giải thích chính xác, ngắn gọn bằng tiếng Việt; dùng ví dụ đời thường khi có khái niệm kỹ thuật và nói rõ khi bạn không chắc chắn.” Cụm “ngắn gọn bằng tiếng Việt” giữ câu trả lời phù hợp người học và tiết kiệm token; yêu cầu “nói rõ khi không chắc chắn” giúp hạn chế việc trình bày suy đoán như sự thật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ ba lượt nên trợ lý nhanh chóng quên ngữ cảnh cũ. Tôi sẽ bổ sung bộ nhớ tóm tắt: trước khi cắt history, dùng model tạo một bản tóm tắt có cấu trúc về mục tiêu, dữ kiện và quyết định quan trọng, sau đó gửi bản tóm tắt này cùng ba lượt gần nhất trong những request tiếp theo; dữ liệu nhạy cảm cần được lọc và có thời hạn lưu trữ.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
