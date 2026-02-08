import regex

from llm.tokenizer import Pair, encode

news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 ‘정보의 균형’을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 ‘무엇이 쟁점인지’, ‘누가 무엇을 주장하는지’, ‘사실은 무엇인지’**를 명확히 전달하는 데 집중합니다.
""".strip()

code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

vietnamese_text = r"""
Đi qua giai đoạn này toàn thể khối Sale support chúng tôi đã có được sự củng cố về hoạt động , hoàn thành được các mục tiêu do Chủ tịch tập đòan và Ban giám đốc giao trong năm hoạt động đầu tiên của khối. Chúng tôi xin chân thành gửi lời cảm ơn đến Chủ tịch tập đoàn ,Ban giám đốc và các khối phòng ban đã cho chúng tôi những ý kiến đóng góp, sự ủng hộ và giúp đỡ nhiệt tình
Bắt đầu năm mới 2011 cũng là bắt đầu một nhiệm kỳ mới , Sale support mang theo hoạt động của mình với một khối lượng công việc đòi hỏi sự đầu tư về thời gian và nhân lực khá lớn. Trách nhiệm của từng nhóm trong khối đã được phân chia khá rõ ràng trong khi nguồn nhân lực chưa được ổn định về số lượng cũng như chất lượng, tuy nhiên tất cả thành viên cũng đã có những đóng góp cho công việc chung để đảm bảo hoàn thành các nhiệm vụ được giao .
Đối với nhóm Logistic, đây là nhóm chịu nhiều vất vả và sức ép nhất trong giai đọan này, các nhân viên của chúng tôi thường xuyên phải làm thêm giờ thậm chí là làm đêm do việc vừa phải đảm bảo hoạt động thường ngày vừa phải hoàn thành một số nhiệm vụ yêu cầu trong thời gian chuyển đổi từ địa điểm cũ sang địa điểm mới , rất nhiều công việc phải làm trong giai đoạn này :
o Chuyển toàn bộ hệ thống kho sang địa điểm mới.
o Hoàn thành việc chuẩn hóa hoạt động GDP , giấy phép đã được Sở y tế Hà nội cấp ngày 1 tháng 3 năm 2011.
o Tổng kiểm kho các chi nhánh trên toàn quốc để chuẩn hóa toàn bộ các mặt hàng trong kho theo lô, hạn sử dụng.
o Đào tạo lại qui trình xuất, nhập và kiểm soát kho cho toàn bộ các thủ kho, phân công lại nhiệm vụ các nhân viên kho tại kho tổng của tập đoàn.
o Cập nhật và kiểm soát toàn bộ hàng hóa trên phần mềm bán hàng theo đúng qui định về GSP và GDP.
o Bắt đầu vận hành xe lạnh chạy tuyến Bắc – Nam giao hàng cho các chi nhánh và các nhà phân phối của tập đoàn. Chuyến khởi hành đầu tiên bắt đầu trong tháng 4 đã hoàn thành tốt lịch trình đề ra tạo tiền đề cho các chu kỳ mỗi tháng một chuyến thuận lợi , suôn sẻ.
Nhóm chịu sức ép thứ hai chính là nhóm Thầu vì Quí 1 của năm là giai đoạn cao điểm của mùa thầu trên toàn quốc, với một dàn nhân viên khá mỏng hai nhân viên và một quản lý nhóm nhưng các bạn phải hoàn thành một khối lượng công việc rất lớn và đòi hỏi về mặt chất lượng khá khắt khe : Hồ sơ thầu phải đảm bảo độ chính xác và hoàn thành đúng thời hạn qui định là 100% kèm theo đó là phải xử lý toàn bộ các công văn , báo giá , hợp đồng , thanh lý. Chính vì vậy nên phòng thầu bao giờ cũng là phòng tắt đèn sau cùng của khối văn phòng. Nói như vậy không phải chúng tôi muốn than phiền mà chỉ để các bạn Kinh doanh hiểu và thông cảm để phối hợp theo đúng các kỳ hạn mà chúng tôi đề nghị tạo điều kiện cho chúng tôi có thể hoàn thành tốt được các nhiệm vụ góp phần hỗ trợ cho bộ phận kinh doanh đáp ứng tốt yêu cầu của khách hàng .
Để đầu vào được suôn sẻ thì hoạt động của Sale admin cũng được chúng tôi chú trọng trong việc phối hợp với nhà cung cấp phần mềm bán hàng chuẩn hóa, phân quyền và cập nhật các thao tác mới trong việc xử lý đơn hàng. Việc lập Kế hoạch bán hàng đã được sự phối hợp bước đầu của Khối Kinh doanh để các kế hoạch đề ra ngày càng sát với thực tế, việc nhận đơn hàng ngòai hệ thống các tổng đài trên toàn quốc chúng tôi đã bắt đầu triển khai thêm các kênh thông tin qua các số hotline và hệ thống tin nhắn SMS trong hệ thống mobile của tập đoàn. Công việc phản hồi tới khách hàng trạng thái xử lý đơn hàng đã được chúng tôi bước đầu kiểm soát trên hệ thống phần mềm bán hàng để các nhân viên Sale admin từng bước đảm nhận được tròn vai trò của mình là cầu nối giữa khách hàng với hệ thống xử lý đơn hàng bên trong của tập đoàn. Một phần nữa trong công việc của nhóm Sale admin chính là việc nhập hàng, đây là công việc bước đầu được chúng tôi kiểm soát nhằm đảm bảo cho việc cung cấp hàng đủ theo Kế hoạch bán hàng của Khối kinh doanh cũng như việc dự trữ hàng hợp lý trên toàn bộ hệ thống kho của tập đoàn .
Góp phần hỗ trợ cho Khối kinh doanh và chủ yếu hiện tại cho nghành hàng OTC trong 6 tháng đầu năm 2011 là các bạn trợ lý Marketing , các bạn như những con ong cần mẫn trợ giúp cho các giám đốc Marketing, giám đốc sản phẩm hoàn thành từ các công việc nhỏ nhất là tập hợp số liệu đến các công việc đòi hỏi sự đầu tư về chất xám như dịch các tài liệu tham khảo, lên maquette các tài liệu để phục vụ cho các hoạt động khoa học của tập đoàn .
Đi qua giai đoạn này toàn thể khối Sale support chúng tôi đã có được sự củng cố về hoạt động , hoàn thành được các mục tiêu do Chủ tịch tập đòan và Ban giám đốc giao trong năm hoạt động đầu tiên của khối. Chúng tôi xin chân thành gửi lời cảm ơn đến Chủ tịch tập đoàn ,Ban giám đốc và các khối phòng ban đã cho chúng tôi những ý kiến đóng góp, sự ủng hộ và giúp đỡ nhiệt tình . Chúng tôi tin tưởng rằng với các nhiệm vụ đã thực hiện trong 6 tháng đầu năm sẽ là tiền đề tốt để chúng tôi nâng cao hơn về chất lượng hoạt động trong 6 thàng cuối năm 2011 góp phần cùng với tòan bộ tập đoàn hoàn thành tốt các mục tiêu của năm 2011 để chào mừng kỷ niệm 10 năm xây dựng và trưởng thành của Tập đoàn và 1 năm thành lập khối Sale support.
SSD
""".strip()


def line_to_pair(line: str) -> Pair:
    p = line.strip("\n").split(" ")
    return (int(p[0]), int(p[1]))


def main():
    gpt4_split_pattern = r"""
        # shorten form of words like "will" ('ll), "have" ('ve), etc
        '(?i:[sdmt]|ll|ve|re)
        # 
        |[^\r\n\p{L}\p{N}]?+\p{L}+
        # numbers between 0 - 999
        |\p{N}{1,3}
        |[ ]?[^\s\p{L}\p{N}]++[\r\n]*
        |\s*[\r\n]
        |\s+(?!\S)
        |\s+
        """
    split_pattern = regex.compile(gpt4_split_pattern, regex.VERBOSE)
    with open("cache/merges.txt", "r") as f:
        merges = [line_to_pair(line) for line in f]

    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    for l, r in merges:
        vocab.append(vocab[l] + vocab[r])

    encoded_text = encode(merges, split_pattern, vietnamese_text)
    ratio = len(vietnamese_text.encode("utf-8")) / len(encoded_text)
    print(f"{ratio:<7.2f}")


if __name__ == "__main__":
    main()
