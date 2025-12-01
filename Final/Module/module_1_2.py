from Module.gpt_seq_2_seq import gpt_call

class Module_1_2:
    """
        Phát hiện và trích xuất các claim từ một chuỗi văn bản đầu vào.

        Input
        -----
        client : OpenAI.Client
            - Đối tượng client đã được khởi tạo sẵn để gọi OpenAI (hoặc API tương thích).
            - Bắt buộc phải có phương thức `chat.completions.create(...)`.

        input_text : str
            - Chuỗi văn bản thô do người dùng nhập.
            - Có thể chứa nhiều dòng.
            - Thường là văn bản không chuẩn hóa (slang, viết tắt, teencode, lỗi chính tả, câu cảm thán…).

        Output
        ------
        list[str]
            - Một danh sách Python gồm các câu tiếng Việt đã được làm sạch và chuẩn hóa.
            - Mỗi phần tử là một claim (mệnh đề khẳng định, có thể kiểm chứng đúng/sai).
            - Đầu ra luôn là một list hợp lệ nhờ cơ chế kiểm tra và xử lý JSON trong `gpt_call`.

        Ví dụ Output
        ------------
        Input:
            "MU da thua 5 tran lien tiep roi\nTroi oi nong qua"
        Output:
            ["Manchester United đã thua 5 trận liên tiếp."]

        Input:
            "Trời ơi nóng quá!"
        Output:
            []
        """
    def __init__(self, system_prompt=None):
        # Prompt mặc định nếu không truyền vào
        self.system_prompt = system_prompt or """
            Bạn là mô-đun Claim Detection.

            Nhiệm vụ:
            - Chuyển đầu vào của người dùng về tiếng Việt chuẩn, đầy đủ chính tả, ngữ pháp, và không dùng từ viết tắt, lóng hay teencode.
            - Chỉ giữ các câu chứa claim: phát biểu khẳng định, luận điểm, hoặc thông tin có thể kiểm chứng.
            - Loại bỏ câu cảm thán, chào hỏi, xã giao hoặc câu không chứa claim.
            - Nếu câu có mở đầu như "tôi tin rằng", "theo tôi nghĩ", "có lẽ", bỏ phần mở đầu và giữ nội dung khẳng định.

            Trả về **một JSON array** duy nhất, ví dụ:
            ["Claim 1", "Claim 2", ...]
            """

    def detect_claims(self, client, input_text):
        return gpt_call(self.system_prompt, input_text, client)