# ==========================================
# 這是一個專為華語教師設計的「文章等級與超綱詞檢測系統」，核心功能是根據學生目前的學習進度，
# 分析輸入文章中是否存在超出該進度範圍的詞彙，並提供一個輕量化的 AI 降維重構提示詞生成器，
# 協助教師針對超綱詞彙提出替換建議。
# 整理資料.py 將(當代中文)教材料原始檔整理成 「完整教材詞彙庫.json」，
# 結構為 {冊別: {課別: {完整詞彙: [...]}}}，供本系統使用。
# 執行時，點擊 run.bat 即可啟動 Streamlit 介面，請確保同一目錄下存在「完整教材詞彙庫.json」檔案。
# 必備的檔案包括：
# - APP.py (本檔)
# - 完整教材詞彙庫.json (由整理資料.py 產出)
# - dict.txt.big (JIEBA 的繁體字典，提升斷詞精準度，非必須但強烈建議)
# ==========================================

import streamlit as st
import json
import jieba
import re

# ==========================================
# 1. 核心邏輯轉譯與排序引擎 (解構底層字串無能)
# ==========================================
def chinese_to_arabic(cn_str):
    """降維轉譯：將中文數字強制映射為物理整數，支撐線性因果律排序"""
    cn_num_map = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10}
    if not cn_str:
        return 0
    if cn_str == '十':
        return 10
    if cn_str.startswith('十'): # 十一 ~ 十九
        return 10 + cn_num_map.get(cn_str[1:], 0)
    if len(cn_str) == 3 and cn_str[1] == '十': # 二十一 ~ 九十九
        return cn_num_map.get(cn_str[0], 0) * 10 + cn_num_map.get(cn_str[2], 0)
    if cn_str.endswith('十'): # 二十, 三十...
        return cn_num_map.get(cn_str[0], 0) * 10
    return cn_num_map.get(cn_str, 0)

def sort_key(item):
    """精準正則切片：排除任何排版元資料污染，僅抓取純中文數字進行整數權重計算"""
    vol_name, lesson_name = item
    # 限制僅抓取 [一二三四五六七八九十] 字元，防堵「一冊第一」等滑稽錯誤
    vol_match = re.search(r'第([一二三四五六七八九十]+)冊', vol_name)
    les_match = re.search(r'第([一二三四五六七八九十]+)課', lesson_name)
    
    v_num = chinese_to_arabic(vol_match.group(1)) if vol_match else 0
    l_num = chinese_to_arabic(les_match.group(1)) if les_match else 0
    
    return (v_num, l_num)

# ==========================================
# 2. 資料庫解讀與因果時間軸建構
# ==========================================
@st.cache_data
def load_vocabulary_database(filepath="完整教材詞彙庫.json"):
    """安全載入實體 JSON 數據庫"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"實體檔案遺失：找不到 {filepath}。請確認前置教材處理腳本已成功執行並產出檔案。")
        return {}

def build_lesson_sequence(vocab_db):
    """提取所有冊課實體，並強制作線性時間軸排序"""
    sequence = []
    for vol_name, lessons in vocab_db.items():
        for lesson_name in lessons.keys():
            sequence.append((vol_name, lesson_name))
            
    # 拒絕作業系統預設的 Unicode 排序，強制套用物理整數排序
    sequence.sort(key=sort_key)
    return sequence

def get_accumulated_vocab(vocab_db, sequence, target_index):
    """計算從歷史起點到當前進度（含當前課數）的絕對已知詞彙聯集（Union）"""
    accumulated_vocab = set()
    for i in range(target_index + 1):
        vol, les = sequence[i]
        lesson_words = vocab_db[vol][les].get("完整詞彙", [])
        accumulated_vocab.update(lesson_words)
    return accumulated_vocab

# ==========================================
# 3. GUI 介面實體化與交互濾網
# ==========================================
st.set_page_config(page_title="中原大學國專部文章等級與超綱詞檢測系統", layout="wide")
st.title("文章等級與超綱詞檢測系統")

# 初始化環境
vocab_db = load_vocabulary_database()
if not vocab_db:
    st.stop()

lesson_sequence = build_lesson_sequence(vocab_db)
if not lesson_sequence:
    st.error("邏輯斷層：教材詞彙庫內無有效冊課數據。")
    st.stop()

# 淨化視覺呈現：剝除「第一冊 - 第一冊第一課」的疊床架屋現象
options = []
for vol, les in lesson_sequence:
    clean_lesson = re.sub(f"^{vol}", "", les)
    options.append(f"{vol} - {clean_lesson}")

# -- 介面佈局 --
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 學習維度設定")
    selected_option = st.selectbox("請選擇學生目前的最高進度 (含本課)：", options)
    target_index = options.index(selected_option)
    
    st.subheader("2. 文本輸入區")
    target_article = st.text_area("請貼上欲檢測的文章：", height=250)

with col2:
    st.subheader("3. 專業、專門詞保留機制")
    st.markdown("請輸入**容許超綱**的保留詞彙 (如人名、特定地名、專業用詞、專門詞)，請以**半形空格**分隔：")
    reserved_words_input = st.text_input("保留詞彙：", placeholder="例如：張怡君 進化論 台北捷運")

# ==========================================
# 4. 核心比對引擎 (絕對關閉 HMM)
# ==========================================
st.markdown("---")
if st.button("執行文章比對 (強制關閉 HMM)", type="primary"):
    if not target_article.strip():
        st.warning("檢測文本為空，系統拒絕進行無意義的空無運算。")
    else:
        # A. 依據時間軸拉出絕對已知詞彙
        known_vocab = get_accumulated_vocab(vocab_db, lesson_sequence, target_index)
        
        # B. 記憶體突入：強制將已知詞彙高頻注入 JIEBA 字典，杜絕歷史已知詞被切碎
        for w in known_vocab:
            jieba.add_word(w, freq=99999)
            
        # C. 清洗文章標點符號，執行無妥協斷詞
        clean_article = re.sub(r'[^\w\s]', '', target_article)
        tokens = jieba.cut(clean_article, HMM=False)
        
        article_vocab = set()
        for t in tokens:
            t = t.strip()
            if t:
                article_vocab.add(t)
                
        # D. 差集邏輯運算：文章詞彙 - 歷史已知詞彙 = 初步超綱詞
        out_of_bounds_words = article_vocab - known_vocab
        
        # E. 執行人工安全豁免過濾
        reserved_vocab = set([w.strip() for w in reserved_words_input.split(" ") if w.strip()])
        final_target_words = out_of_bounds_words - reserved_vocab
        
        # 將狀態物理鎖定於 Session State 中，供下方的 Prompt 引擎提取
        st.session_state['final_target_words'] = final_target_words
        st.session_state['selected_level'] = selected_option
        st.session_state['target_article'] = target_article
        
        # -- 檢測報告輸出 --
        st.subheader("檢測結果報告")
        st.write(f"**文章總獨立詞數：** {len(article_vocab)}")
        st.write(f"**初步判定超綱詞數：** {len(out_of_bounds_words)}")
        st.write(f"**扣除人工保留詞後，實際需重構之生詞數：** {len(final_target_words)}")
        
        if final_target_words:
            st.error("【超綱詞彙現形】")
            st.write(" ｜ ".join(final_target_words))
        else:
            st.success("核心因果檢驗通過：本文所有詞彙皆完全收斂於該學生目前的學習邊界之內。")

# ==========================================
# 5. AI 文章調整提示詞生成器
# ==========================================
st.markdown("---")
if st.button("生成 AI 降維重構提示詞 (Prompt)"):
    if 'final_target_words' not in st.session_state or not st.session_state['final_target_words']:
        st.warning("請先執行文章比對。若無超綱詞，則無觸發重構提示詞之動機。")
    else:
        # 1. 僅提取超綱詞與文章，拒絕夾帶龐大的已知字典
        words_str = "、".join(st.session_state['final_target_words'])
        article_str = st.session_state['target_article']
        
        # 2. 建構高結構化、低 Token 消耗的 Prompt
        prompt_template = f"""
你現在是一位專業的華語教師。
你的核心任務是：保留文本原意，不進行文章美化或更動文意，只針對下列「超綱詞彙」評估是否可以用更基礎、更常見的詞彙代換。

【背景事實】
- 以下為目標文章：
{article_str}

- 這是本篇文章出現的【超綱詞彙】：
{words_str}

【執行指令】
請針對每一個【超綱詞彙】，進行以下分析，並以表格呈現結果：
1. 超綱詞彙：(列出該詞)
2. 替換建議梯度：請提供 1 到 3 個不改變語意因果的簡單替換詞。請按「最簡單口語」到「一般常用詞」的順序排列。
3. 邏輯說明：若該詞屬專有名詞、教材關鍵語法，或一旦替換會嚴重扭曲原意，請在此欄位直接寫明「建議保留原詞」並附上一句話的教學解釋。

請拒絕無意義的寒暄與心理學名詞，直接輸出表格。
"""
        st.subheader("請複製以下輕量化 Prompt 貼給大語言模型：")
        
        # 使用 st.text_area 讓教師容易複製，Token 消耗量極低
        st.text_area("AI 提示詞：", value=prompt_template, height=350)
