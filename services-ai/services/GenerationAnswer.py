import re
import torch
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast


def format_answer(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'(\s+(\d+|[a-zA-Z])[\.\):])', r'\n\1', text)

    text = text.strip()

    return text


class GenerationAnswer:
    def __init__(self, model_name="Captone2C2SE13/legal-support"):
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        self.pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

    def get_answer(self, question):
        answer = self.pipe(question)[0]['generated_text']
        return format_answer(answer)


if __name__ == "__main__":
    from datetime import datetime
    qa_model = GenerationAnswer()
    question = "Mẹ tôi và dượng tôi ở với nhau gần 10 năm nhưng không đăng ký kết hôn. Nay dượng tôi phản bội mẹ tôi, có vợ mới và muốn chia đôi số tài sản, trong đó tiền vốn là của tôi bỏ ra cho mẹ tôi làm ăn. Ông ta đòi làm đơn kiện nếu mẹ tôi không đồng ý chia đôi số tài sản hiện tại. Trường hợp này phải giải quyết như thế nào? (Số tiền tôi đưa mẹ làm ăn không có giấy tờ gì chứng minh cả)"
    start_time = datetime.now()
    answer = qa_model.get_answer(question)
    end_time = datetime.now()
    print(answer)
    duration = end_time - start_time
    print("\nThời gian thực hiện:", duration)
