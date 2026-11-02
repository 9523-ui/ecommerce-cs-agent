"""
生成「知识库膨胀」实验用的干扰条目。

设计意图（面试可讲）：
真实业务里知识库不是从 31 条随机长到 1 万条，而是按
「品类 x 政策类型 x 渠道」不断派生出措辞高度相似的新条目。
这类近邻条目正是检索退化的主因，所以干扰项必须造得像真的，
而不是随机文本。

两类干扰：
1. 同域近邻：电商客服语境，但品类/渠道/时效被替换，共享大量词面
2. 跨域条目：HR / IT / 行政 / 财务等其他业务域，共享通用疑问词

答案统一为占位文本，确保「正确条目」始终只有原始 31 条 FAQ，
因此 top-1 是否命中目标 FAQ 的标注没有歧义。

固定随机种子，实验可复现。
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "distractors.json"
random.seed(20261103)

PRODUCTS = [
    "手机数码", "家用电器", "服装鞋帽", "生鲜食品", "美妆个护", "母婴用品",
    "图书文具", "运动户外", "家居建材", "汽车配件", "宠物用品", "珠宝钟表",
    "保健食品", "乐器耗材", "办公设备", "农资绿植",
]
CHANNELS = ["微信小程序", "APP 客户端", "网页端", "线下门店", "直播间的链接", "分销渠道", "企业采购通道"]
TIMES = ["3 个工作日", "5 个工作日", "7 个自然日", "15 天", "30 天", "48 小时", "24 小时"]
POLICIES = ["无理由退货", "换货", "价保", "运费险理赔", "发票开具", "保修",
            "补发", "取消订单", "修改地址", "发货时效", "退款到账", "积分抵扣"]

# 同域近邻：句式刻意贴近真实 FAQ 的提问方式
NEAR_TEMPLATES = [
    "{p}类商品的{pol}是怎么规定的？",
    "{p}的{pol}要几天？",
    "通过{ch}下单，{pol}还适用吗？",
    "关于{p}的{pol}，有什么需要特别注意的限制？",
    "{p}订单想办理{pol}，入口在哪里？",
    "{pol}超过{t}之后还能申请吗？",
    "{p} category 的{pol}和普货一样吗？",
    "客服你好，{ch}买的{p}走{pol}流程要准备什么材料？",
    "{pol}的时效在{ch}和网页端是不是不一样？",
    "双11 期间{p}的{pol}会顺延到{t}吗？",
]

# 跨域：其他业务域的问答，共享「怎么 / 多少 / 可以 / 几天」等通用词
CROSS_DOMAIN = [
    ("员工差旅住宿的报销标准是多少？", "按职级分档，需提供发票与行程单。"),
    ("公司团建经费怎么申请？", "由部门负责人提交预算，行政部审批。"),
    ("年假有几天？", "按司龄分档，入职满一年起算。"),
    ("会议室怎么预约？", "在 OA 系统选择时段并提交。"),
    ("服务器报警了应该找谁处理？", "联系运维值班，按 P1/P2 分级响应。"),
    ("VPN 连不上怎么排查？", "检查账号权限与客户端版本，必要时提 IT 工单。"),
    ("季度绩效考核什么时候启动？", "每季度末由 HR 统一发起。"),
    ("办公电脑坏了走什么流程？", "提交资产维修单，IT 部门检测后处理。"),
    ("社保基数每年几月调整？", "通常在每年 7 月统一申报调整。"),
    ("合同盖章需要哪些审批？", "法务初审后由授权人用印。"),
    ("新员工入职要准备哪些材料？", "身份与学历证明、银行卡、体检报告。"),
    ("工资条在哪里查？", "登录人力系统「薪酬」模块查看。"),
    ("门禁卡丢了怎么补办？", "前台登记并缴纳工本费后补发。"),
    ("项目立项报告怎么写？", "按模板填写背景、预算与里程碑。"),
    ("客户合同归档在哪里？", "统一存入合同管理系统，纸质件入档案室。"),
]
CROSS_SUFFIX = ["", "（{ch}）", "（适用于{p}）", "（{t}内有效）"]


def make(n_near: int = 8000, n_cross: int = 2000):
    items = []

    # 1) 同域近邻
    for i in range(n_near):
        tpl = random.choice(NEAR_TEMPLATES)
        q = tpl.format(
            p=random.choice(PRODUCTS),
            pol=random.choice(POLICIES),
            ch=random.choice(CHANNELS),
            t=random.choice(TIMES),
        )
        items.append({
            "id": f"N{i:05d}",
            "category": "干扰-同域近邻",
            "question": q,
            "answer": "该条目为规模实验的干扰数据，不属于标准 FAQ 答案。请按对应品类的适用规则处理。",
        })

    # 2) 跨域
    for i in range(n_cross):
        q, a = random.choice(CROSS_DOMAIN)
        suffix = random.choice(CROSS_SUFFIX)
        if suffix:
            q += suffix.format(
                ch=random.choice(CHANNELS),
                p=random.choice(PRODUCTS),
                t=random.choice(TIMES),
            )
        items.append({
            "id": f"X{i:05d}",
            "category": "干扰-跨域",
            "question": q,
            "answer": a + "（该条目为规模实验的干扰数据，不属于电商客服知识库。）",
        })

    random.shuffle(items)
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(items)


if __name__ == "__main__":
    n = make()
    print(f"已生成 {n} 条干扰项 -> {OUT}")
    print(f"文件大小 {OUT.stat().st_size / 1024:.0f} KB")
    sample = json.loads(OUT.read_text(encoding="utf-8"))
    for x in sample[:6]:
        print("  ", x["id"], x["question"])
