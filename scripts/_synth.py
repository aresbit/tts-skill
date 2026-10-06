#!/usr/bin/env python3
"""克隆音色合成的核心实现。

自包含：只依赖 requirements.txt 里声明的包，**不 import 本仓库的任何模块**，
也不读仓库里的任何文件。这意味着分块策略在这里有一份独立实现——它和仓库
``domain/text.py`` 的那份是两个会各自漂移的真相。这是「自包含」换来的代价，
心里有数就行；改了这边记得那边也对一遍。

三条硬约束都体现在下面的常量里，来源是实测（不是猜的）：
  150 字   超过安全区分块上限的表现是**静默跳字**：引擎不报错、不截断，
           少念一大段然后正常返回。所以宁可多切一段。
  货币符号  ``¥1500`` 会丢「元」，``$5`` 会念成「五千」。这不归这里管——
           写稿时就得写成汉字，因为脚本去展开它们反而会引入新的错。
  30 秒    引擎对参考音是**先按 30 秒截断、再裁静音**，所以头尾静音会吃预算。
"""

from __future__ import annotations

import re
from pathlib import Path

#: 与仓库 scripts/make_episode.py 用同一个模型，避免同一台机器上两份缓存。
MODEL = "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16"

#: 实测安全上限是 200 字，150 是留了余量的取值。
MAX_CHARS = 150

#: 段与段之间的静音。太短听起来赶，太长像断线。
GAP_SECONDS = 0.35

_SENT_END = "。！？!?；;…"
_CLAUSE_END = "，,、：:"


def strip_markdown(text: str) -> str:
    """把 Markdown 剥成适合**朗读**的纯文本。

    阅读和朗读是两种消费方式。排版记号对眼睛是信息，对耳朵是噪声：
    ``## 标题`` 念出来是「井号井号 标题」，``|---|`` 念出来是「竖线横线横线
    竖线」，``[文字](https://...)`` 会把整条 URL 念完。所以这里把记号剥掉，
    只留下该说的字。

    刻意不做的：不把表格「翻译」成自然语言（那要理解语义，会猜错），
    只把竖线换成顿号，让一行念下来不至于卡住。要更好的效果就改稿子本身——
    录播客本来也不该念表格。
    """
    t = text
    t = re.sub(r"```.*?```", "", t, flags=re.S)          # 代码块：念不出来，整块丢掉
    t = re.sub(r"`([^`]*)`", r"\1", t)                   # 行内代码：留内容
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)           # 图片：丢掉
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)       # 链接：留文字，丢 URL
    # 表格必须**先删分隔行、再换竖线**。反过来的话，`|---|---|` 会先变成
    # 一串顿号，分隔行就再也认不出来了。
    t = re.sub(r"^[ \t]*\|?[ \t:|-]*\|[ \t:|-]*$", "", t, flags=re.M)
    t = re.sub(r"[ \t]*\|[ \t]*", "，", t)

    heading_or_item = re.compile(r"^\s{0,3}(#{1,6}\s|>\s?|[-*+]\s+|\d+[.)]\s+)")
    rule = re.compile(r"^\s*[-*_]{3,}\s*$")

    out: list[str] = []
    for raw_line in t.splitlines():
        line = raw_line.strip()
        if not line:
            # 保留**单个**空行当段落边界。全删掉段落就会黏成一段——
            # 这正是「依赖去哪了mlx-audio 有两个…」那种读法的来源。
            if out and out[-1] != "":
                out.append("")
            continue

        line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)      # 粗体
        line = re.sub(r"~~(.+?)~~", r"\1", line)             # 删除线
        line = re.sub(r"(?<!\w)(\*|_)(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", line)  # 斜体

        if rule.match(line):
            continue

        if heading_or_item.match(line):
            line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
            line = re.sub(r"^\s{0,3}>\s?", "", line)
            line = re.sub(r"^\s*[-*+]\s+", "", line)
            line = re.sub(r"^\s*\d+[.)]\s+", "", line)
            line = re.sub(r"[ \t]+", " ", line).strip()
            # 标题和列表项各自成句。不给句末标点的话，它们会和下一行连读，
            # 听感上就像一句话没说完就换了话题。
            if line and line[-1] not in _SENT_END + "，,：:、；！？":
                line += "。"

        for a, b in (("→", "，"), ("←", "，"), ("↔", "，"), ("✓", ""), ("✗", "")):
            line = line.replace(a, b)
        line = re.sub(r"([，、])\s*\1+", r"\1", line)     # 表格拼出来的连续顿号
        line = re.sub(r"[ \t]+", " ", line).strip(" ，、")
        if line:
            out.append(line)

    while out and out[-1] == "":
        out.pop()
    return "\n".join(out).strip()


def plan_chunks(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """把稿子切成每段不超过 max_chars 的块。**段落边界绝不跨越。**

    先按空行分段落，段内按句末标点断句，再把整句贪心装进块里；单句超长时
    退回子句（，、：）再切，最后才硬切。顺序不能反——先硬切会把句子切碎，
    引擎在碎句子上的韵律会明显变差。
    """
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = re.sub(r"\s*\n\s*", "", para).strip()
        if not para:
            continue
        buf = ""
        for sentence in _split_sentences(para):
            for piece in _split_long(sentence, max_chars):
                if buf and len(buf) + len(piece) > max_chars:
                    chunks.append(buf)
                    buf = ""
                buf += piece
                if len(buf) >= max_chars:
                    chunks.append(buf)
                    buf = ""
        if buf:
            chunks.append(buf)
    return chunks


def _split_sentences(para: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for ch in para:
        buf += ch
        if ch in _SENT_END:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """超长单句先按子句切，仍超长才硬切。"""
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    buf = ""
    for ch in sentence:
        buf += ch
        if ch in _CLAUSE_END and len(buf) >= max_chars // 2:
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)

    out: list[str] = []
    for p in parts:
        while len(p) > max_chars:
            out.append(p[:max_chars])
            p = p[max_chars:]
        if p:
            out.append(p)
    return out


def synth_chunk(text: str, wav: Path, ref_audio: Path, ref_text: str, seed: int) -> None:
    """合成一段。

    ``mx.random.seed`` 必须在 ``generate_audio`` **之前**：引擎自己没有 seed
    入口，能设的只有这个全局状态。不设的话同一进程里从第二段起会落进不动点，
    听不出问题，但也没了多样性。
    """
    import mlx.core as mx
    from mlx_audio.tts.generate import generate_audio

    mx.random.seed(seed)
    wav.unlink(missing_ok=True)
    generate_audio(
        text=text,
        model=MODEL,
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        file_prefix=str(wav.with_suffix("")),
        join_audio=True,
        verbose=False,
        play=False,
    )
    if not wav.exists():
        raise RuntimeError(f"合成没有落盘：{wav.name}（磁盘满？模型没下完？）")


def synthesize(
    text: str,
    out: Path,
    ref_audio: Path,
    ref_text: str,
    *,
    seed: int = 1,
    max_chars: int = MAX_CHARS,
    keep: bool = False,
    work_dir: Path | None = None,
) -> tuple[int, float, int]:
    """稿子 → 一集音频。返回（采样率，总秒数，段数）。"""
    import numpy as np
    import soundfile as sf

    out = Path(out)
    chunks = plan_chunks(text, max_chars)
    if not chunks:
        raise ValueError("稿子是空的。")

    parts = Path(work_dir) if work_dir else out.parent / (out.stem + ".parts")
    parts.mkdir(parents=True, exist_ok=True)

    pieces: list[Path] = []
    for i, chunk in enumerate(chunks):
        wav = parts / f"seg{i:03d}.wav"
        head = chunk[:24] + ("…" if len(chunk) > 24 else "")
        print(f"  [{i + 1}/{len(chunks)}] {len(chunk):>3}字  {head}", flush=True)
        synth_chunk(chunk, wav, ref_audio, ref_text, seed + i)
        pieces.append(wav)

    waves: list = []
    rate = 0
    for p in pieces:
        data, rate = sf.read(str(p), dtype="float32")
        waves.append(np.asarray(data).ravel())

    silence = np.zeros(int(rate * GAP_SECONDS), dtype=np.float32)
    merged = waves[0]
    for wave in waves[1:]:
        merged = np.concatenate([merged, silence, wave])
    sf.write(str(out), merged, rate, subtype="PCM_16")

    if not keep:
        for p in pieces:
            p.unlink(missing_ok=True)
        try:
            parts.rmdir()
        except OSError:
            pass
    return rate, len(merged) / rate, len(chunks)
