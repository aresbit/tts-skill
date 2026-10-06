#!/usr/bin/env python3
"""tts —— 用已克隆的音色把一段文字念成音频。**完全自包含。**

    .claude/skills/tts/scripts/tts.py "要念的话"
    .claude/skills/tts/scripts/tts.py --file 稿子.txt --play

自包含到什么程度
================

这个 skill **不依赖本仓库的任何东西**：不 import ``podcast`` 包，不找
``scripts/make_episode.py``，不用 ``.venv-gpu``，不读 ``ref/``。把整个
``.claude/skills/tts/`` 目录拷到任何地方（哪怕不在这个仓库里）都能跑。

它需要的两样东西第一次运行时自己长出来，之后不再联网：

  ``.venv/``                     独立 Python 环境。用 uv 建，依赖见 requirements.txt。
  ``.cache/huggingface/``        模型缓存。设了 HF_HOME，模型落在这而不是全局缓存。

所以「拷过去就能跑」成立的前提只有两个：机器上有 **uv**，以及第一次跑时**能联网**
（拉依赖 + 拉模型，约 1–2 GB）。这两样没法塞进目录里，别的都可以。

为什么入口要 re-exec 一次
=========================

建完 venv 之后，当前进程用的还是外部的解释器，缺 mlx。所以入口会把参数原样
交给 ``.venv/bin/python`` 再跑一遍自己。用一个环境变量当标记判断「已经在里面了」——
不要用 ``sys.executable`` 比较，venv 的 python 是指向基础解释器的符号链接，
``.resolve()`` 之后两边一模一样，会误判成已在 venv 里，然后 import 失败。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
VENV = SKILL / ".venv"
HF_HOME = SKILL / ".cache" / "huggingface"
REQUIREMENTS = SKILL / "requirements.txt"
DEFAULT_VOICE_DIR = SKILL / "assets" / "voice"
DEFAULT_VOICE = "bili-voice3"

#: 标记「已经跑在 skill 自己的 venv 里」。见模块 docstring。
_MARK = "TTS_SKILL_VENV"

#: 上一次装依赖时 requirements.txt 的指纹，用来判断要不要补装。
_STAMP = VENV / ".requirements.sha256"


def _venv_python() -> Path:
    return VENV / "bin" / "python"


def _need_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(
            "首次运行需要 uv 来建独立环境，但没找到 uv。\n"
            "  装：brew install uv\n"
            "  或：curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    return uv


def _install(py: Path) -> None:
    subprocess.run(
        [_need_uv(), "pip", "install", "--python", str(py), "-r", str(REQUIREMENTS)],
        check=True,
    )


def _ensure_venv() -> Path:
    """有且依赖没变就用现成的；否则建或补装。

    依赖清单的指纹必须存下来比对，不能只看 ``.venv/`` 在不在——否则改了
    requirements.txt 之后，老 venv 会被一直复用，缺的包要等到真正 import
    它的那一刻才炸（``sounddevice`` 就是这么暴露出来的）。
    """
    py = _venv_python()
    want = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()

    if py.exists():
        got = _STAMP.read_text().strip() if _STAMP.exists() else ""
        if got == want:
            return py
        print("依赖清单变了，正在补装…")
        _install(py)
        _STAMP.write_text(want)
        print("环境已更新。\n")
        return py

    uv = _need_uv()
    print(f"首次运行：正在建独立环境 {VENV}")
    subprocess.run([uv, "venv", str(VENV), "--python", "3.13"], check=True)
    print("正在装依赖（约 1 GB，只这一次）…")
    _install(py)
    _STAMP.write_text(want)
    print("环境就绪。\n")
    return py


def _reenter(py: Path) -> int:
    """把参数原样交给 venv 里的解释器，再跑一次本脚本。"""
    env = os.environ.copy()
    env[_MARK] = str(VENV)
    env["HF_HOME"] = str(HF_HOME)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return subprocess.run(
        [str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env=env
    ).returncode


def _slug(text: str, n: int = 16) -> str:
    s = re.sub(r"\s+", "-", text.strip())
    s = re.sub(r"[/\\:*?\"<>|\n\r\t]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:n].strip("-") or "tts"


def _resolve_voice(name: str, voice_dir: Path) -> tuple[Path, Path, str]:
    """音色必须成对。不成对就早失败——引擎拿到错配的参考音**不会报错**。"""
    ref = voice_dir / f"{name}.wav"
    ref_txt = voice_dir / f"{name}.txt"
    missing = [p for p in (ref, ref_txt) if not p.exists()]
    if missing:
        have = sorted(p.stem for p in voice_dir.glob("*.wav")) if voice_dir.exists() else []
        raise SystemExit(
            f"音色不完整：{name}\n"
            f"  缺：{', '.join(p.name for p in missing)}\n"
            f"  引擎需要 wav + txt 成对——wav 是声音，txt 是它念的字。\n"
            f"  音色库 {voice_dir} 现有：{', '.join(have) or '(空)'}"
            f"\n  加新音色：放 <名>.wav（≤30s、24 kHz、单人、无配乐）+ 逐字转写 <名>.txt"
        )
    return ref, ref_txt, ref_txt.read_text(encoding="utf-8").strip()


def _read_text(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.file and args.text:
        raise SystemExit("给了文字又给了 --file，二选一。")
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("没有稿子：positional text、--file 或 stdin 至少给一个。")
    if not text.strip():
        raise SystemExit("稿子是空的。")
    return text


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="用已克隆的音色把文字念成音频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("text", nargs="?", help="要念的文字（不给就读 --file 或 stdin）")
    p.add_argument("--file", type=Path, help="从文件读稿子（UTF-8）")
    p.add_argument(
        "--md",
        action="store_true",
        help="按 Markdown 剥壳再念（.md/.markdown 结尾的 --file 会自动启用）",
    )
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"音色名（默认 {DEFAULT_VOICE}）")
    p.add_argument("--voice-dir", type=Path, default=DEFAULT_VOICE_DIR, help="音色库目录")
    p.add_argument("--list-voices", action="store_true", help="列出音色后退出")
    p.add_argument("-o", "--out", type=Path, help="输出 wav（默认 ./tts-out/<时间戳>.wav）")
    p.add_argument("--seed", type=int, default=1, help="随机种子；同种子出同一份音频")
    p.add_argument("--max-chars", type=int, default=None, help="每段字数上限（默认 150）")
    p.add_argument("--keep", action="store_true", help="保留分段音频，便于单独重做某段")
    p.add_argument("--play", action="store_true", help="合成完直接播放")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.list_voices:
        d = args.voice_dir
        names = sorted(p.stem for p in d.glob("*.wav")) if d.exists() else []
        print(f"音色库：{d}")
        for n in names:
            ok = "✓" if (d / f"{n}.txt").exists() else "✗ 缺转写"
            print(f"  {n}  {ok}")
        return 0

    # 建环境；然后把自己交给 venv 里的解释器再跑一遍。
    py = _ensure_venv()
    if os.environ.get(_MARK) != str(VENV):
        return _reenter(py)
    os.environ.setdefault("HF_HOME", str(HF_HOME))

    from _synth import strip_markdown, synthesize  # 与本文件同目录

    text = _read_text(args, parser)
    if args.md or (args.file is not None and args.file.suffix.lower() in (".md", ".markdown")):
        raw = text
        text = strip_markdown(text)
        if not text.strip():
            raise SystemExit("Markdown 剥完是空的——稿子里只有排版、没有正文？")
        saved = len(raw) - len(text)
        print(f"Markdown 剥壳：{len(raw)} → {len(text)} 字（去掉 {saved} 个排版字符）")
    ref, ref_txt, ref_text = _resolve_voice(args.voice, args.voice_dir)

    out = args.out
    if out is None:
        out = Path.cwd() / "tts-out" / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(text)}.wav"
    out = Path(out)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # 稿子留一份，便于复现和改稿重跑
    script_file = out.with_name(out.stem + ".script.txt")
    script_file.write_text(text, encoding="utf-8")

    kwargs = {"seed": args.seed, "keep": args.keep}
    if args.max_chars is not None:
        kwargs["max_chars"] = args.max_chars

    print(f"音色　：{args.voice}  ({ref})")
    print(f"稿子　：{len(text.strip())} 字  ({script_file})")
    print(f"输出　：{out}\n")

    started = time.perf_counter()
    rate, duration, n = synthesize(text, out, ref, ref_text, **kwargs)
    print(f"\n好了：{out}")
    print(f"  {duration:.1f} 秒 · {rate} Hz · {n} 段 · 耗时 {time.perf_counter() - started:.0f} 秒")
    print(f"  听：afplay {out}")
    if args.play:
        subprocess.run(["afplay", str(out)])
    return 0


if __name__ == "__main__":
    # _synth.py 与本文件同目录，直接跑时也得能 import 到它。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
