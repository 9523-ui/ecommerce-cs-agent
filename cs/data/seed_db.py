"""
客服业务库初始化脚本
运行: .venv/Scripts/python.exe -m cs.data.seed_db
生成: cs/data/cs.db （订单 / 物流 / 售后工单）
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs.db")

SCHEMA = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS tickets;

CREATE TABLE orders (
    order_no      TEXT PRIMARY KEY,   -- 订单号
    customer      TEXT NOT NULL,      -- 客户姓名
    phone         TEXT,               -- 手机号后四位
    product       TEXT NOT NULL,      -- 商品名
    quantity      INTEGER DEFAULT 1,  -- 数量
    amount        REAL NOT NULL,      -- 实付金额
    status        TEXT NOT NULL,      -- 待发货/已发货/已签收/已取消/退款中/已退款
    created_at    TEXT NOT NULL,      -- 下单时间
    paid          INTEGER DEFAULT 1   -- 是否已付款 1/0
);

CREATE TABLE shipments (
    order_no      TEXT PRIMARY KEY,
    carrier       TEXT,               -- 承运商
    tracking_no   TEXT,               -- 运单号
    state         TEXT,               -- 已揽收/运输中/派送中/已签收/异常
    last_update   TEXT,               -- 最后更新时间
    note          TEXT                -- 异常说明
);

CREATE TABLE tickets (
    ticket_id     TEXT PRIMARY KEY,
    order_no      TEXT,
    issue_type    TEXT,               -- 物流异常/质量问题/少件错发/退款问题/其他
    description   TEXT,
    status        TEXT DEFAULT 'open',-- open/processing/resolved
    created_at    TEXT
);
"""

ORDERS = [
    # order_no, customer, phone, product, qty, amount, status, created_at, paid
    ("CS20261101001", "李伟",   "8821", "无线降噪耳机 Pro",      1,  899.00, "已签收", "2026-10-28 10:12:00", 1),
    ("CS20261101002", "王芳",   "3345", "全棉四件套 1.8m",       2,  436.00, "已发货", "2026-10-30 21:40:00", 1),
    ("CS20261101003", "张敏",   "9902", "儿童学习桌 可升降",      1, 1280.00, "待发货", "2026-11-01 09:05:00", 1),
    ("CS20261101004", "陈强",   "1177", "机械键盘 87键 青轴",    1,  329.00, "已签收", "2026-10-25 14:30:00", 1),
    ("CS20261101005", "刘洋",   "5566", "电动牙刷 声波款",       1,  259.00, "退款中", "2026-10-27 19:20:00", 1),
    ("CS20261101006", "赵雪",   "2200", "羊绒围巾 米白色",       1,  168.00, "已签收", "2026-10-29 11:00:00", 1),
    ("CS20261101007", "孙鹏",   "7788", "空气净化器 除甲醛",     1, 1599.00, "已发货", "2026-10-31 08:15:00", 1),
    ("CS20261101008", "周琳",   "4411", "运动跑鞋 42码",        1,  499.00, "已签收", "2026-10-26 16:45:00", 1),
    ("CS20261101009", "吴昊",   "6633", "咖啡豆 意式拼配 500g",  3,  237.00, "待发货", "2026-11-01 20:30:00", 0),
    ("CS20261101010", "郑爽",   "8899", "轻薄笔记本 14寸 16G",   1, 4999.00, "已签收", "2026-10-20 13:10:00", 1),
]

SHIPMENTS = [
    ("CS20261101001", "顺丰速运", "SF1234567890123", "已签收", "2026-10-29 16:22:00", None),
    ("CS20261101002", "中通快递", "ZT7788990011223", "运输中", "2026-11-02 07:30:00", None),
    ("CS20261101004", "圆通速递", "YT9900112233445", "已签收", "2026-10-26 12:05:00", None),
    ("CS20261101006", "顺丰速运", "SF5566778899001", "已签收", "2026-10-30 10:40:00", None),
    ("CS20261101007", "京东物流", "JD1122334455667", "异常",   "2026-11-01 18:00:00", "收件地址电话无人接听，包裹滞留网点"),
    ("CS20261101008", "中通快递", "ZT3344556677889", "已签收", "2026-10-27 15:30:00", None),
    ("CS20261101010", "京东物流", "JD9988776655443", "已签收", "2026-10-21 11:15:00", None),
]

TICKETS = [
    ("T20261101001", "CS20261101007", "物流异常", "客户反馈物流显示异常，包裹滞留网点", "processing", "2026-11-01 18:30:00"),
]


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", ORDERS)
    conn.executemany("INSERT INTO shipments VALUES (?,?,?,?,?,?)", SHIPMENTS)
    conn.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?)", TICKETS)
    conn.commit()

    print(f"已生成 {DB_PATH}")
    for t in ("orders", "shipments", "tickets"):
        n = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n} 行")
    conn.close()


if __name__ == "__main__":
    main()
