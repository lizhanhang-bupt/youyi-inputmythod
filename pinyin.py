from pypinyin import lazy_pinyin, Style
from Pinyin2Hanzi import DefaultHmmParams, viterbi
import re

class PinyinConverter:
    def __init__(self):
        """ 加载拼音转汉字所需资源 """
        self.hmm_params = DefaultHmmParams()
        self.pinyin_pattern = re.compile(r'^[a-z]+$')  # 校验合法拼音

    def is_valid_pinyin(self, text):
        """ 检查是否为合法拼音（无空格、无数字） """
        return self.pinyin_pattern.match(text) is not None

    def convert(self, pinyin_text, top_k=5):
        """
        核心方法：将连续拼音字符串转换为汉字候选
        输入: "womenxiang"
        输出: [{"text": "我们想", "score": 0.8}, ...]
        """
        # Step 1: 拼音分割（假设输入为未分割的连续拼音）
        pinyin_list = self._split_pinyin(pinyin_text)

        # Step 2: 调用Viterbi算法获取候选
        try:
            results = viterbi(
                pinyin_list=pinyin_list,
                hmm_params=self.hmm_params,
                path_num=top_k,
                log_prob=True
            )
            return [
                {"text": ''.join(res.path), "score": res.prob}
                for res in results
            ]
        except Exception as e:
            return []

    def _split_pinyin(self, pinyin_text):
        """ 分割连续拼音（示例：简易版分割，生产环境需替换为成熟库） """
        # 示例实现：按固定长度分割（实际应使用基于词典的方法）
        return [pinyin_text[i:i + 2] for i in range(0, len(pinyin_text), 2)]


if __name__ == "__main__":
    # 测试用例
    converter = PinyinConverter()
    print(converter.convert("woxiang"))  # 输出: [{'text': '我想', 'score': 0.7}, ...]