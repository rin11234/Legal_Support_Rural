# Law QA API

This project provides a FastAPI-based web service for answering legal questions using a Hugging Face model.

## Requirements

- Python 3.12
- `uvicorn==0.32.0`
- `fastapi==0.115.2`
- `python-dotenv==1.0.1`
- `huggingface-hub==0.25.2`
- `transformers==4.45.2`
- `torch==2.4.1`

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/Captone2-legal-support/services-ai.git
    cd services-ai
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the `env` directory with your Hugging Face token:
    ```dotenv
    HF_TOKEN='your_huggingface_token'
    ```

## Using Docker

1. Build the Docker image:
    ```sh
    docker build -t law-qa-api .
    ```

2. Run the Docker container:
    ```sh
    docker run -d -p 8000:8000 --name law-qa-api-container law-qa-api
    ```
3. The API will be available at `http://127.0.0.1:8000`.
 
## Running the API

1. Start the FastAPI server:
    ```sh
    uvicorn main:app --reload
    ```

2. The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `GET /`: Returns a welcome message.
- `GET /get_answer`: Accepts a `question` parameter and returns the answer.

## Example

Request:
```sh
curl -X 'GET' \
  'http://127.0.0.1:8000/get_answer?question=Các%20biện%20pháp%20giải%20quyết%20tranh%20chấp%20đất%20đai%20theo%20pháp%20luật%20Việt%20Nam%20là%20gì?' \
  -H 'accept: application/json'

```

```sh
{
  "description": "Mẹ tôi và dượng tôi ở với nhau gần 10 năm nhưng không đăng ký kết hôn",
  "specificSituation": "Nay dượng tôi phản bội mẹ tôi, có vợ mới và muốn chia đôi số tài sản, trong đó tiền vốn là của tôi bỏ ra cho mẹ tôi làm ăn. Ông ta đòi làm đơn kiện nếu mẹ tôi không đồng ý chia đôi số tài sản hiện tại. Trường hợp này phải giải quyết như thế nào? (Số tiền tôi đưa mẹ làm ăn không có giấy tờ gì chứng minh cả)"
}
```
Response:
```sh
{
  "question": "Mẹ tôi và dượng tôi ở với nhau gần 10 năm nhưng không đăng ký kết hôn Nay dượng tôi phản bội mẹ tôi, có vợ mới và muốn chia đôi số tài sản, trong đó tiền vốn là của tôi bỏ ra cho mẹ tôi làm ăn. Ông ta đòi làm đơn kiện nếu mẹ tôi không đồng ý chia đôi số tài sản hiện tại. Trường hợp này phải giải quyết như thế nào? (Số tiền tôi đưa mẹ làm ăn không có giấy tờ gì chứng minh cả)",
  "answer": "Căn cứ Điều 59 Luật Hôn nhân và gia đình năm 2014 quy định: - Vợ, chồng hoặc cả hai người có quyền yêu cầu Tòa án giải quyết ly hôn. - Cha, mẹ, người thân thích khác có quyền yêu cầu Tòa án giải quyết ly hôn khi một bên vợ, chồng do bị bệnh tâm thần hoặc mắc bệnh khác mà không thể nhận thức, làm chủ được hành vi của mình, đồng thời là nạn nhân của bạo lực gia đình do chồng, vợ của họ gây ra làm ảnh hưởng nghiêm trọng đến tính mạng, sức khỏe, tinh thần của họ. - Chồng không có quyền yêu cầu ly hôn trong trường hợp vợ đang có thai, sinh con hoặc đang nuôi con dưới 12 tháng tuổi. - Trong trường hợp vợ đang có thai, sinh con hoặc đang nuôi con dưới 12 tháng tuổi thì chồng không có quyền yêu cầu ly hôn. - Trong trường hợp chồng đang có thai, sinh con hoặc đang nuôi con dưới 12 tháng tuổi thì"
}
```


# Question examples
```json
[
    {
        "specificSituation": "Nay dượng tôi phản bội mẹ tôi, có vợ mới và muốn chia đôi số tài sản, trong đó tiền vốn là của tôi bỏ ra cho mẹ tôi làm ăn. Ông ta đòi làm đơn kiện nếu mẹ tôi không đồng ý chia đôi số tài sản hiện tại. Trường hợp này phải giải quyết như thế nào? (Số tiền tôi đưa mẹ làm ăn không có giấy tờ gì chứng minh cả)",
        "description": "Mẹ tôi và dượng tôi ở với nhau gần 10 năm nhưng không đăng ký kết hôn."
    },
    {
        "specificSituation": "Ngọc Hải, hiện sống và làm việc tại Long An, đang tìm hiểu về pháp luật tố tụng dân sự qua các năm và có thắc mắc về nghĩa vụ nộp án phí sơ thẩm theo Bộ luật tố tụng dân sự 2004.",
        "description": "Quy định về nghĩa vụ nộp án phí sơ thẩm trong tố tụng dân sự"
    },
    {
        "specificSituation": "Địa điểm để đăng ký biện pháp bảo đảm.",
        "description": "Tôi dùng muốn biết địa điểm thực hiện đăng ký biện pháp bảo đảm."
    },
    {
        "specificSituation": "Em học đến hết học kì I năm lớp 8 thì nghỉ học và đi làm công. Em chưa tốt nghiệp cấp 2, năm nay 18 tuổi và sẽ có đơn gọi nhập ngũ vào năm sau. Em rất muốn đi nghĩa vụ quân sự nhưng không biết trình độ học vấn của mình có được gọi hay không.",
        "description": "Chưa học hết lớp 8 có được đi nghĩa vụ quân sự không?"
    },
    {
        "specificSituation": "Tôi muốn biết về quyền hủy bỏ di chúc hợp pháp đã lập.",
        "description": "Điều kiện khi người lập di chúc hợp pháp được hủy bỏ di chúc hợp pháp."
    },
    {
        "specificSituation": "Hiện tại tôi muốn nhận trẻ em Việt Nam làm con nuôi. Vậy theo pháp luật Việt Nam thì tôi phải nộp lệ phí đăng ký nuôi con nuôi là bao nhiêu tiền Việt Nam?",
        "description": "Tôi là công dân nước ngoài đang sinh sống và làm việc tại Việt Nam và đã có đăng ký thường trú tại Việt Nam."
    }
]
```

## Table of Examples

| Question | Answer |
|----------|--------|
| Mẹ tôi và dượng tôi ở với nhau gần 10 năm nhưng không đăng ký kết hôn. Nay dượng tôi phản bội mẹ tôi, có vợ mới và muốn chia đôi số tài sản, trong đó tiền vốn là của tôi bỏ ra cho mẹ tôi làm ăn. Ông ta đòi làm đơn kiện nếu mẹ tôi không đồng ý chia đôi số tài sản hiện tại. Trường hợp này phải giải quyết như thế nào? (Số tiền tôi đưa mẹ làm ăn không có giấy tờ gì chứng minh cả) | Theo quy định tại khoản 2 Điều 59 Luật Hôn nhân và Gia đình năm 2000 thì tài sản chung của vợ chồng gồm tài sản do vợ, chồng tạo ra, thu nhập do lao động, hoạt động sản xuất, kinh doanh và những thu nhập hợp pháp khác của vợ chồng trong thời kỳ hôn nhân; tài sản mà vợ chồng được thừa kế chung hoặc được tặng cho chung và những tài sản khác mà vợ chồng thoả thuận là tài sản chung. Theo quy định tại khoản 2 Điều 59 Luật Hôn nhân và Gia đình năm 2000 thì tài sản chung của vợ chồng gồm tài sản do vợ, chồng tạo ra, thu nhập do lao động, hoạt động sản xuất, kinh doanh và những thu nhập hợp pháp khác của vợ chồng trong thời kỳ hôn nhân; tài sản mà vợ chồng được thừa kế chung hoặc được tặng cho chung và những tài sản khác mà vợ chồng thoả thuận là tài sản chung. Theo quy định tại khoản 2 Điều 59 Luật |
| Quy định về nghĩa vụ nộp án phí sơ thẩm trong tố tụng dân sự Ngọc Hải, hiện sống và làm việc tại Long An, đang tìm hiểu về pháp luật tố tụng dân sự qua các năm và có thắc mắc về nghĩa vụ nộp án phí sơ thẩm theo Bộ luật tố tụng dân sự 2004. | Theo quy định tại Điều 276 Bộ luật tố tụng dân sự 2004, nghĩa vụ nộp án phí sơ thẩm được quy định như sau:\n 1. Người yêu cầu Toà án giải quyết vụ án dân sự phải nộp án phí sơ thẩm nếu vụ án được giải quyết trong cùng vụ án đó.\n 2. Trong cùng vụ án, nếu vụ án được giải quyết trong cùng vụ án đó thì người yêu cầu Toà án giải quyết vụ án phải nộp án phí sơ thẩm nếu vụ án được giải quyết trong cùng vụ án đó.\n 3. Trong cùng vụ án, nếu vụ án được giải quyết trong cùng vụ án đó thì người yêu cầu Toà án giải quyết vụ án phải nộp án phí sơ thẩm nếu vụ án được giải quyết trong cùng vụ án đó.\n 4. Trong cùng vụ án, nếu vụ án được giải quyết trong cùng vụ án đó thì người yêu cầu Toà án giải quyết vụ án phải nộp án phí sơ thẩm nếu vụ án được giải quyết trong cùng vụ án đó. |
| Tôi dùng muốn biết địa điểm thực hiện đăng ký biện pháp bảo đảm. Địa điểm để đăng ký biện pháp bảo đảm. | Tại Điều 18 Nghị định 99/2022/NĐ-CP có quy định về địa điểm đăng ký như sau: “Điều\n 18. Địa điểm đăng ký\n 1. Địa điểm đăng ký được xác định theo quyết định của cơ quan có thẩm quyền về đăng ký biện pháp bảo đảm, pháp luật về đăng ký biện pháp bảo đảm, trừ trường hợp pháp luật về đăng ký biện pháp bảo đảm, pháp luật về đăng ký biện pháp bảo đảm có quy định khác.\n 2. Địa điểm đăng ký được xác định theo quyết định của cơ quan có thẩm quyền về đăng ký biện pháp bảo đảm, pháp luật về đăng ký biện pháp bảo đảm, trừ trường hợp pháp luật về đăng ký biện pháp bảo đảm, pháp luật về đăng ký biện pháp bảo đảm có quy định khác.\n 3. Địa điểm đăng ký được xác định theo quyết định của cơ quan có thẩm quyền về đăng ký biện pháp bảo đảm, pháp luật về đăng ký biện pháp bảo đảm, trừ trường hợp pháp luật về đăng |
| Chưa học hết lớp 8 có được đi nghĩa vụ quân sự không? Em học đến hết học kì I năm lớp 8 thì nghỉ học và đi làm công. Em chưa tốt nghiệp cấp 2, năm nay 18 tuổi và sẽ có đơn gọi nhập ngũ vào năm sau. Em rất muốn đi nghĩa vụ quân sự nhưng không biết trình độ học vấn của mình có được gọi hay không. | Tại Điều 4 Thông tư 148/2018/TT-BQP quy định về tiêu chuẩn tuyển quân như sau:\n 1. Tuổi đời:\n a) Công dân từ đủ 18 tuổi đến hết 25 tuổi.\n b) Công dân nam được đào tạo trình độ cao đẳng, đại học đã được tạm hoãn gọi nhập ngũ trong thời gian một khóa đào tạo của một trình độ đào tạo thì tuyển chọn và gọi nhập ngũ đến hết 27 tuổi.\n 2. Tiêu chuẩn chính trị:\n a) Thực hiện theo Thông tư liên tịch số 01/2016/TTLT-BQP-BCA ngày 15 tháng 4 năm 2016 của Bộ trưởng Bộ Quốc phòng - Bộ trưởng Bộ Công an quy định tiêu chuẩn chính trị trong tuyển chọn công dân vào phục vụ trong Quân đội.\n b) Đối với các cơ quan, đơn vị trọng yếu, cơ mật và lực lượng vệ binh, kiểm soát quân sự chuyên nghiệp thực hiện tuyển chọn theo quy định của Bộ Quốc phòng.\n 3. Tiêu chuẩn sức |
| Điều kiện khi người lập di chúc hợp pháp được hủy bỏ di chúc hợp pháp. Tôi muốn biết về quyền hủy bỏ di chúc hợp pháp đã lập. | Tại Điều 643 Bộ luật Dân sự 2015 có quy định về việc hủy bỏ di chúc như sau:\n 1. Người lập di chúc có thể hủy bỏ di chúc đã lập vào bất cứ lúc nào.\n 2. Trong trường hợp người lập di chúc từ chối nhận di chúc thì di chúc đã lập và bản di chúc bị hủy bỏ.\n 3. Việc hủy bỏ di chúc đã lập được thực hiện như sau:\n a) Người lập di chúc hủy bỏ di chúc đã lập;\n b) Người thừa kế theo di chúc hoặc theo pháp luật của người lập di chúc đã chết;\n c) Cơ quan, tổ chức được chỉ định là người thừa kế không còn tồn tại vào thời điểm lập di chúc;\n d) Những người được chỉ định là người thừa kế không còn tồn tại vào thời điểm lập di chúc; đ) Người thừa kế là cá nhân chết theo pháp luật;\n e) Cơ quan, tổ chức được chỉ định là cơ quan, tổ chức không còn tồn tại vào thời điểm lập |
| Tôi là công dân nước ngoài đang sinh sống và làm việc tại Việt Nam và đã có đăng ký thường trú tại Việt Nam. Hiện tại tôi muốn nhận trẻ em Việt Nam làm con nuôi. Vậy theo pháp luật Việt Nam thì tôi phải nộp lệ phí đăng ký nuôi con nuôi là bao nhiêu tiền Việt Nam? | Theo quy định tại Điều 6 Luật Nuôi con nuôi 2010 thì người nhận con nuôi phải nộp lệ phí đăng ký nuôi con nuôi theo quy định của pháp luật nước nơi người được nhận là con nuôi thường trú. Trường hợp cha dượng nhận con riêng của vợ hoặc chồng làm con nuôi hoặc cô, cậu, dì, chú, bác ruột nhận cháu làm con nuôi thì không phải nộp lệ phí đăng ký nuôi con nuôi. Như vậy, trường hợp bạn nhận con nuôi là cô, cậu, dì, chú, bác ruột nhận cháu làm con nuôi thì phải nộp lệ phí đăng ký nuôi con nuôi theo quy định của pháp luật nước nơi người được nhận là con nuôi thường trú. Theo quy định tại Điều 6 Nghị định 141/2016/NĐ-CP thì lệ phí đăng ký nuôi con nuôi được quy định như sau:\n 1. Mức thu lệ phí đăng ký nuôi con nuôi:\n 400.000 đồng/trường hợp.\n 2. Mức thu lệ phí đăng ký nuôi con nuôi: |
