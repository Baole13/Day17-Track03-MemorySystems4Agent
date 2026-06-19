# Chào mừng các bạn đến với Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

Repo này được chia thành ba phần rõ ràng:

- `src/`: bản scaffold dành cho sinh viên, chứa pseudocode và TODO để hoàn thiện
- `data/`: dữ liệu benchmark ở root để dùng cho cả benchmark chuẩn và stress benchmark

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Cách dùng repo này

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Sau đó làm việc trực tiếp với `src/` và `data/` ở root repo.

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.

## Phân tích kết quả benchmark

Sau khi hoàn thiện lab và chạy `python src/benchmark.py`, kết quả benchmark mẫu hiện tại cho thấy:

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2396 | 18009 | 0.04 | 0.24 | 0 | 0 |
| Advanced | 3707 | 27698 | 0.82 | 0.87 | 371 | 25 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 589 | 23500 | 0.00 | 0.27 | 0 | 0 |
| Advanced | 784 | 10131 | 0.67 | 0.83 | 218 | 26 |

### 1. Vì sao `Advanced` có recall tốt hơn `Baseline`

`Baseline Agent` chỉ giữ short-term memory trong đúng một thread. Khi benchmark hỏi recall ở thread mới, agent này gần như không còn gì để dựa vào, nên `Cross-session recall` chỉ còn `0.04` ở benchmark chuẩn và `0.00` ở stress benchmark.

Ngược lại, `Advanced Agent` có thêm `User.md` để lưu facts ổn định như tên, nơi ở hiện tại, nghề nghiệp hiện tại, style trả lời, đồ uống và món ăn yêu thích. Vì vậy khi sang thread mới, agent vẫn truy xuất lại được profile người dùng, dẫn đến recall tăng rõ rệt lên `0.82` ở benchmark chuẩn và `0.67` ở stress benchmark.

Điểm quan trọng ở đây là `Advanced` không chỉ “nhớ nhiều hơn”, mà còn nhớ theo kiểu có cấu trúc hơn. Thông tin correction như đổi nơi ở từ Đà Nẵng sang Huế hoặc đổi nghề từ backend engineer sang MLOps engineer được cập nhật theo fact hiện hành, thay vì giữ nguyên cả bản cũ và bản mới như nhau.

### 2. Vì sao `Advanced` có thể tốn hơn ở hội thoại ngắn

Ở benchmark chuẩn, `Advanced` có `Agent tokens only` và `Prompt tokens processed` đều cao hơn `Baseline`. Điều này là hợp lý vì mỗi lượt agent phải mang thêm chi phí:

- đọc `User.md`
- cập nhật profile nếu phát hiện fact mới
- duy trì compact summary cùng recent messages

Nói cách khác, long-term memory không miễn phí. Với hội thoại ngắn hoặc số thread còn ít, phần overhead này có thể khiến `Advanced` tốn hơn nhưng chưa tạo ra khác biệt quá lớn về hiệu quả ngữ cảnh.

### 3. Vì sao compact memory chủ yếu tối ưu `Prompt tokens processed`

Ở stress benchmark, điểm khác biệt rõ nhất nằm ở `Prompt tokens processed`:

- `Baseline`: `23500`
- `Advanced`: `10131`

Đây là tín hiệu quan trọng nhất của compact memory. `Baseline` giữ nguyên toàn bộ lịch sử thread nên mỗi lượt sau phải kéo theo ngày càng nhiều context. `Advanced` thì nén phần lịch sử cũ thành summary và chỉ giữ lại một số message gần nhất, nên độ dài prompt tăng chậm hơn nhiều.

Điều đáng chú ý là compact memory không nhất thiết làm `Agent tokens only` thấp hơn trong mọi trường hợp. Agent vẫn phải trả lời đầy đủ, và đôi khi còn tốn thêm token cho phần summary hoặc câu trả lời có cấu trúc hơn. Lợi ích chính của compact nằm ở việc giảm chi phí ngữ cảnh phải xử lý lặp đi lặp lại qua nhiều lượt.

### 4. Memory growth tăng như thế nào và rủi ro gì đi kèm

`Baseline` không có persistent memory nên `Memory growth` bằng `0`. `Advanced` thì có file `User.md`, nên benchmark cho thấy file memory tăng lên `371 bytes` ở standard benchmark và `218 bytes` ở stress benchmark.

Lợi ích của phần memory này là recall chéo session tốt hơn rõ rệt. Tuy nhiên nó kéo theo một số rủi ro thực tế:

- file memory sẽ tiếp tục phình ra theo thời gian nếu không có guardrail
- agent có thể lưu nhầm thông tin nhiễu nếu extraction quá lỏng
- fact cũ có thể trở thành stale fact nếu không xử lý correction đúng

Vì vậy bản triển khai này dùng thêm `conflict handling + confidence-aware persistence`: chỉ lưu các facts có tính khẳng định rõ, bỏ qua câu hỏi hoặc câu gây nhiễu như “chỉ là câu đùa” hoặc “không phải nơi ở hiện tại”, và ưu tiên fact mới nhất khi người dùng đính chính.

### Kết luận ngắn

Kết quả benchmark cho thấy câu chuyện đúng với mục tiêu của lab:

1. `Baseline` đơn giản, rẻ hơn ở một số trường hợp ngắn, nhưng quên gần như hoàn toàn khi qua session mới.
2. `Advanced` nhớ tốt hơn nhờ `User.md`, nhưng phải trả thêm chi phí cho lớp memory bền vững.
3. Khi hội thoại rất dài, compact memory tạo ra lợi thế rõ ở `Prompt tokens processed`.
4. Hệ thống mạnh hơn thì cũng phức tạp hơn, vì phải quản lý growth, correction, và tránh persist sai fact.
