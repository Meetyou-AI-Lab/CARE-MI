
QA2D_CHATGPT_SYSTEM_PROMPT = """
Given chinese question and answer pair, combine and modify them to produce corresponding declarative sentences.
"""

QA2D_CHATGPT_USER_PROMPT = """
You are an accurate NLP annotator.
Given chinese question and answer pair, combine and modify them to produce corresponding declarative sentences.
Keep the information mentioned in the chinese question and answer pair unchanged.
Do the least modification during the generation.
Make sure that the generated declarative sentences are fluent.
For examples:

Q:淋病是何种类型的炎症
A:急性化脓性炎症
D:淋病是急性化脓性炎症

Q:下肢浮肿（+）
A:轻度妊高征
D:下肢浮肿（+）提示轻度妊高征

Q:严重肝功能不全的病人不宜用
A:泼尼松
D:严重肝功能不全的病人不宜用泼尼松

Q:肠扭转引起的坏死
A:湿性坏疽
D:肠扭转引起的坏死是湿性坏疽

Q:肾脏中抗原抗体复合物的检测
A:免疫比浊法
D:肾脏中抗原抗体复合物的检测应使用免疫比浊法

Q:两侧坐骨棘间径≥10cm
A:女型骨盆
D:两侧坐骨棘间径≥10cm说明是女型骨盆

Q:此时局部最佳处理方法
A:冲洗上药
D:此时局部最佳处理方法是冲洗上药

Q:应当对
A:孕妇进行产前诊断
D:应当对孕妇进行产前诊断

Q:是因为母乳中
A:含白蛋白、球蛋白较多
D:是因为母乳中含白蛋白、球蛋白较多

Q:治疗24小时后仍有自觉症状
A:剖宫产
D:治疗24小时后仍有自觉症状，应采取剖宫产

Q:月经期使用清洁卫生巾
A:避免感染
D:月经期使用清洁卫生巾可避免感染

Q:1小时后儿头下降0.5cm
A:胎头下降延缓
D:1小时后儿头下降0.5cm提示胎头下降延缓

Q:子宫出现Hegar征
A:孕6周时开始
D:子宫出现Hegar征从孕6周时开始

Now, given the following sample, generate the declarative sentences:

"""

D2FD_CHATGPT_SYSTEM_PROMPT = """
Given the following sample, generate the negated declarative sentences.
"""

D2FD_CHATGPT_USER_PROMPT = """
You are an accurate NLP annotator.
Given chinese declarative statement and answer pair, generate corresponding negated declarative sentences.
Do the least modification during the generation.
Make sure that the generated negated declarative sentences are fluent.
For examples:


S:比较甲、乙两地新生儿的死因构成比，宜绘制圆图。
N:比较甲、乙两地新生儿的死因构成比，不宜绘制圆图。

S:行人工破膜后9小时宫口开9cm提示活跃期延长。
N:行人工破膜后9小时宫口开9cm不提示活跃期延长。

S:胎儿和婴幼儿期生长遵循头尾发展律。
N:胎儿和婴幼儿期生长不遵循头尾发展律。

S:治疗该病，目前首选阿昔洛维。
N:治疗该病，目前不首选阿昔洛维。

S:习惯性晚期流产最常见于子宫颈内口松弛。
N:习惯性晚期流产不常见于子宫颈内口松弛。

S:胎头最低点在坐骨棘水平说明胎头已经衔接。
N:胎头最低点在坐骨棘水平不说明胎头已经衔接。

Now, given the following sample, generate the negated declarative sentences:

"""

QG_CHATYUAN_TF_PROMPT = {
    'pre': "请帮我把：",
    'post': "。这个陈述转化成一个问题。只输出一个问题即可。直接在最后加上'吗？'作为问题。输出的问题必须以'？'结尾。保留陈述句的内容。"
}