"""MySQL 连接诊断脚本。

按顺序探测：
  1) TCP 端口可达性（与 MySQL 配置无关，先排除网络层）
  2) DNS / 主机解析（若使用域名）
  3) pymysql 同步握手（验证账号密码 / 协议握手）
  4) aiomysql 异步握手（验证项目实际使用路径）
  5) 连接池创建 + 并发查询（验证 DB_POOL_MIN/MAX 配置）
  6) 必填表是否存在（验证 database / schema 是否正确）

用法：
  python tests/db_diag.py
  python tests/db_diag.py --host 1.2.3.4 --port 3306 --user root --password xxx --name mydb
"""
from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import time
import traceback

# 允许 `python tests/db_diag.py` 直接运行
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _section(title: str) -> None:
    print("\n" + "=" * 64)
    print(f" {title}")
    print("=" * 64)


def _ok(name: str) -> None:
    print(f"  [OK]   {name}")


def _fail(name: str, detail: str) -> None:
    print(f"  [FAIL] {name}")
    print(f"         ↳ {detail}")


def _warn(name: str, detail: str) -> None:
    print(f"  [WARN] {name}")
    print(f"         ↳ {detail}")


# ---------- 1) TCP reachability ----------

def check_tcp(host: str, port: int, timeout: float = 5.0) -> bool:
    _section(f"1) TCP {host}:{port}")
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            _ok(f"TCP {host}:{port} open ({int((time.time()-t0)*1000)} ms)")
            return True
    except OSError as e:
        _fail(f"TCP {host}:{port} unreachable", f"{type(e).__name__}: {e}")
        print("\n  → 排查建议：")
        print("     · 确认 MySQL 服务已启动：systemctl status mysqld")
        print("     · 确认监听端口：ss -tlnp | grep 3306")
        print("     · 防火墙 / 安全组：sudo ufw status / 云控制台安全组规则")
        print("     · 云数据库白名单：把当前机器的公网 IP 加入 RDS 白名单")
        return False


# ---------- 2) DNS ----------

def check_dns(host: str) -> bool:
    _section(f"2) DNS {host}")
    if host.replace(".", "").isdigit():
        _ok(f"{host} 是 IP，无需 DNS 解析")
        return True
    try:
        ip = socket.gethostbyname(host)
        _ok(f"{host} → {ip}")
        return True
    except socket.gaierror as e:
        _fail(f"DNS 解析失败: {host}", f"{e}")
        return False


# ---------- 3) pymysql ----------

def check_pymysql(host: str, port: int, user: str, password: str, database: str) -> bool:
    _section(f"3) pymysql 同步握手 {user}@{host}:{port}/{database}")
    try:
        import pymysql
    except ImportError:
        _fail("pymysql 未安装", "pip install pymysql")
        return False
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connect_timeout=8,
            read_timeout=10,
            write_timeout=10,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION(), @@hostname, @@port, @@character_set_server")
            ver, hostname, mport, charset = cur.fetchone()
            _ok(f"握手成功 · server={hostname}:{mport} · ver={ver} · charset={charset}")
            cur.execute(
                "SELECT User, Host FROM mysql.user WHERE User=%s ORDER BY Host",
                (user,),
            )
            rows = cur.fetchall()
            print(f"         ↳ 账号 '{user}' 的授权 Host：{[r[1] for r in rows]}")
            cur.execute("SHOW VARIABLES LIKE 'max_connections'")
            mc = cur.fetchone()[1]
            _ok(f"max_connections = {mc}")
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema=%s",
                (database,),
            )
            cnt = cur.fetchone()[0]
            _ok(f"数据库 '{database}' 含 {cnt} 张表")
        conn.close()
        return True
    except Exception as e:
        _fail("pymysql 握手失败", f"{type(e).__name__}: {e}")
        print("\n  → 常见原因：")
        print("     · 1045 Access denied：账号或密码错误，或该 User@Host 未授权")
        print("     · 1049 Unknown database：DB_NAME 写错，或没建库")
        print("     · 1130 Host 'x' is not allowed：MySQL 没给当前机器授权")
        print("     · 2003 Can't connect：服务未启 / 端口不通 / 防火墙")
        return False


# ---------- 4) aiomysql ----------

async def _check_aiomysql(host: str, port: int, user: str, password: str, database: str) -> bool:
    _section(f"4) aiomysql 异步握手（项目实际路径）")
    try:
        import aiomysql
    except ImportError:
        _fail("aiomysql 未安装", "pip install aiomysql")
        return False
    try:
        conn = await aiomysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            db=database,
            autocommit=True,
            connect_timeout=8,
        )
        async with conn.cursor() as cur:
            await cur.execute("SELECT VERSION()")
            row = await cur.fetchone()
            _ok(f"aiomysql 握手成功 · ver={row[0]}")
            await cur.execute("SELECT @@wait_timeout, @@interactive_timeout")
            wt, it = await cur.fetchone()
            _ok(f"wait_timeout={wt}s, interactive_timeout={it}s")
        conn.close()
        return True
    except Exception as e:
        _fail("aiomysql 握手失败", f"{type(e).__name__}: {e}")
        print("\n  → 项目代码会通过 database.init_schema() 走相同路径，失败会导致服务启动直接报错。")
        return False


# ---------- 5) pool + 并发查询 ----------

async def _check_pool(host: str, port: int, user: str, password: str, database: str, mn: int, mx: int) -> bool:
    _section(f"5) aiomysql 连接池 (min={mn}, max={mx}) + 并发 20 个 SELECT 1")
    try:
        import aiomysql
    except ImportError:
        _fail("aiomysql 未安装", "")
        return False
    try:
        pool = await aiomysql.create_pool(
            host=host, port=port, user=user, password=password, db=database,
            minsize=mn, maxsize=mx, autocommit=True, connect_timeout=8,
        )
        async def _q():
            async with pool.acquire() as c:
                async with c.cursor() as cur:
                    await cur.execute("SELECT 1")
                    await cur.fetchone()

        t0 = time.time()
        await asyncio.gather(*[_q() for _ in range(20)])
        _ok(f"20 个并发 SELECT 1 完成 ({int((time.time()-t0)*1000)} ms)")
        pool.close()
        await pool.wait_closed()
        return True
    except Exception as e:
        _fail("连接池 / 并发查询失败", f"{type(e).__name__}: {e}")
        if "max_connections" in str(e):
            print("         ↳ 服务端 max_connections 触顶，考虑调大或减少 DB_POOL_MAX")
        return False


# ---------- 6) 必需表 ----------

async def _check_required_tables(host: str, port: int, user: str, password: str, database: str) -> bool:
    _section("6) 业务表检查（personas / memories / memory_relations / memory_extract_logs / emotion_logs）")
    required = ["personas", "memories", "memory_relations", "memory_extract_logs", "emotion_logs"]
    try:
        import aiomysql
    except ImportError:
        _fail("aiomysql 未安装", "")
        return False
    try:
        conn = await aiomysql.connect(
            host=host, port=port, user=user, password=password, db=database,
            autocommit=True, connect_timeout=8,
        )
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema=%s",
                (database,),
            )
            existing = {r[0] for r in await cur.fetchall()}
        conn.close()
        missing = [t for t in required if t not in existing]
        if missing:
            _fail("缺失表", ", ".join(missing))
            print("         ↳ 服务启动时会自动创建；若持续缺失说明 init_schema() 没跑成功")
            return False
        _ok(f"6 张表全部存在: {', '.join(required)}")
        return True
    except Exception as e:
        _fail("表检查失败", f"{type(e).__name__}: {e}")
        return False


# ---------- main ----------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--name", default=None, help="database name")
    p.add_argument("--pool-min", type=int, default=None)
    p.add_argument("--pool-max", type=int, default=None)
    args = p.parse_args()

    # 优先用 CLI 参数；否则从 .env / config 读
    if args.host and args.port and args.user and args.password and args.name:
        cfg = {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "password": args.password,
            "name": args.name,
            "pool_min": args.pool_min or 1,
            "pool_max": args.pool_max or 10,
        }
    else:
        try:
            from config.config import get_settings
            s = get_settings().db
            cfg = {
                "host": s.host,
                "port": s.port,
                "user": s.user,
                "password": s.password,
                "name": s.name,
                "pool_min": s.pool_min,
                "pool_max": s.pool_max,
            }
        except Exception as e:
            print("无法读取配置（既没传 CLI 参数，配置加载也失败）:", e)
            print("请显式传入：--host --port --user --password --name")
            return 2

    print(f"\n目标: {cfg['user']}@{cfg['host']}:{cfg['port']}/{cfg['name']}")
    print(f"连接池: min={cfg['pool_min']}, max={cfg['pool_max']}")

    results = []
    results.append(check_dns(cfg["host"]))
    results.append(check_tcp(cfg["host"], cfg["port"]))
    if not all(results[-2:]):
        print("\n[结论] 网络层失败，后续步骤跳过。")
        return 1

    results.append(check_pymysql(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["name"]))
    if not results[-1]:
        print("\n[结论] 协议层失败，后续步骤跳过。")
        return 1

    results.append(asyncio.run(_check_aiomysql(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["name"])))
    results.append(asyncio.run(_check_pool(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["name"],
                                            cfg["pool_min"], cfg["pool_max"])))
    results.append(asyncio.run(_check_required_tables(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["name"])))

    print("\n" + "=" * 64)
    print(f" 总结: {sum(results)}/{len(results)} 通过")
    print("=" * 64)
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)