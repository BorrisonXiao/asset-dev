#!/usr/bin/env python3
"""Analyze linguistic patterns in the complete AISHELL-1 auxiliary-loss replication.

The only additional dependency is pypinyin==0.55.0. This is a CPU transcript
audit. All categories and denominators use character alignments after removing
word separators.
"""

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path
import re
from statistics import mean, stdev

from pypinyin import Style, __version__, lazy_pinyin

from analyze_aishell_table3_errors import ARMS, DEFAULT_OUTPUT, SEEDS, load_replication, score_run, sha256


PINYIN_VERSION = "0.55.0"
PRONUNCIATION_CLASSES = ("same_syllable_and_tone", "same_syllable_different_tone",
                         "different_syllable", "unscored")
SPAN_TYPES = ("number", "person", "place", "organization")
# Closed lexicon curated from the reference corpus. Longest textual matches win.
NAME_LEXICON = {
    "person": "李娜 陈凯琳 谢贤 周云露 朱茵 马云 王菲 朱军 杨扬 孙楠 刘翔 林丹 姚明 李宗伟 秦虹 王颖春 郎平 郭晶晶 丁俊晖 章子怡 刘德华 周杰伦 奥巴马 普京 梅西 费德勒 莎拉波娃 刘国梁 伍兹 施瓦辛格 郑嘉颖 陈嘉宝 李斯达".split(),
    "place": "中国 北京 温州 美国 日本 上海 广州 深圳 香港 台湾 天津 重庆 韩国 英国 德国 法国 俄罗斯 澳大利亚 加拿大 新加坡 印度 巴西 杭州 南京 沈阳 张家口 江苏 武汉 四川 唐山 郑州 张家界 河北 山东 广东 陕西 浙江 河南 湖南 湖北 福建 青岛 大连 成都 厦门 澳门 苏州 西安 南宁 昆明 纽约 伦敦 巴黎 东京 柏林 莫斯科 北上广深".split(),
    "organization": "搜狐 国家发改委 发改委 住房城乡建设部 教育部 财政部 商务部 证监会 央行 人民银行 中国银行 工商银行 建设银行 农业银行 招商银行 新华社 人民日报 经济日报 中央电视台 联合国 国际奥委会 百度 腾讯 阿里巴巴 微软 谷歌 高通 三星 苹果公司 苹果电脑公司 联想 华为 中国移动 中国联通 中国电信".split(),
}
NAME_KIND = {name: kind for kind, names in NAME_LEXICON.items() for name in names}
NAMES = re.compile("|".join(map(re.escape, sorted(NAME_KIND, key=lambda name: (-len(name), name)))))
NUMERIC_CHARS = "0-9０-９零〇一二两兩三四五六七八九十百千万萬亿億"
NUMBER = re.compile(rf"(?:百分之)?[{NUMERIC_CHARS}]+(?:[点點.．][{NUMERIC_CHARS}]+)?")
NUMBER_UNIT = re.compile(r"^(?:多?[年月日号岁点时分秒元块个名人位件家项套次批辆米千米公里吨克斤升层届轮倍成%％]|多米)")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U000323af]+")
SYLLABLE = re.compile(r"[a-züêv]+[1-5]", flags=re.ASCII)
LENGTH_LABELS = ("3–10", "11–15", "16–20", "21+")
EXAMPLE_SELECTION = (
    ("same_syllable_and_tone", "asset", "BAC009S0769W0411", "Same pronunciation", "ASSET, the fixed grid and the phone oracle write 它 for 他. Both are tā. The character oracle preserves the reference pronoun."),
    ("same_syllable_different_tone", "asset", "BAC009S0904W0308", "Same syllable, different tone", "ASSET changes 亲子沟通 (parent–child communication) to 亲自沟通 (communicate in person). The three baselines preserve 亲子."),
    ("number", "phone", "BAC009S0912W0428", "Changed time", "The phone oracle writes 12:15 for 19:26. The other three systems preserve the time."),
    ("person", "character", "BAC009S0764W0153", "Changed person name", "The character oracle writes 秦红 for 秦虹. Both name spellings have the same Pinyin reading. ASSET preserves the reference spelling."),
    ("place", "character", "BAC009S0766W0350", "Changed place name", "The character oracle changes 北京 (Beijing) to 本经. ASSET and the fixed grid match the full reference."),
    ("organization", "asset", "BAC009S0766W0188", "Changed organization", "ASSET changes 人民银行 (People’s Bank) to 民营银行 (private bank). The fixed grid matches the full reference."),
    ("repetition", "asset", "BAC009S0764W0289", "New adjacent repetition", "ASSET repeats 芯片市场的 (chip market’s) back-to-back. The phone oracle matches the reference."),
)


def length_label(length):
    if length <= 10:
        return LENGTH_LABELS[0]
    if length <= 15:
        return LENGTH_LABELS[1]
    if length <= 20:
        return LENGTH_LABELS[2]
    return LENGTH_LABELS[3]


@lru_cache(maxsize=65536)
def readings(text):
    result = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors=list)
    if len(result) != len(text):
        raise ValueError(f"Pinyin output does not preserve character positions: {text!r}")
    return tuple(result)


def pronunciation_class(ref, hyp):
    if not SYLLABLE.fullmatch(ref) or not SYLLABLE.fullmatch(hyp):
        return "unscored"
    if ref == hyp:
        return "same_syllable_and_tone"
    if ref[:-1] == hyp[:-1]:
        return "same_syllable_different_tone"
    return "different_syllable"


def reference_spans(words):
    text = "".join(words)
    names = [(NAME_KIND[match[0]], *match.span()) for match in NAMES.finditer(text)
             if match[0] != "三星" or not text[match.end():].startswith("期")]
    boundaries = {0}
    offset = 0
    for word in words:
        offset += len(word)
        boundaries.add(offset)
    numbers = []
    for match in NUMBER.finditer(text):
        start, end = match.span()
        if any(start < name_end and end > name_start for _, name_start, name_end in names):
            continue
        # Accept explicit digit/percent expressions and multi-character numeric
        # tokens. Single Chinese numeral characters need a following measure.
        explicit = match[0].startswith("百分之") or any(char.isdecimal() for char in match[0])
        bounded = start in boundaries and (end in boundaries or NUMBER_UNIT.match(text[end:]))
        if explicit or (bounded and (end - start > 1 or NUMBER_UNIT.match(text[end:]))):
            numbers.append(("number", start, end))
    return names + numbers


def novel_repetitions(ref, hyp):
    """Return units newly repeated back-to-back in the decoded string."""
    units = set()
    for match in HAN.finditer(hyp):
        text = match[0]
        for start in range(len(text) - 3):
            for width in range(2, min(8, (len(text) - start) // 2) + 1):
                unit = text[start:start + width]
                repeated = text[start:start + 2 * width]
                if repeated != unit * 2 or repeated in ref:
                    continue
                if any(width % period == 0 and unit == unit[:period] * (width // period)
                       for period in range(1, width)):
                    continue
                units.add(unit)
    return sorted(units)


def inspect_utterance(row, spans):
    ref, hyp = row["ref"], row["hyp"]
    pronunciations, insertions, changed = [], defaultdict(list), set()
    if row["sub"]:
        ref_pinyin, hyp_pinyin = readings(ref), readings(hyp)
    hypothesis_index = 0
    for kind, reference_index, before, after in row["ops"]:
        if kind == "sub":
            ref_sound = ref_pinyin[reference_index]
            hyp_sound = hyp_pinyin[hypothesis_index]
            pronunciations.append((pronunciation_class(ref_sound, hyp_sound), before, after, ref_sound, hyp_sound))
        if kind in {"sub", "del"}:
            changed.add(reference_index)
        if kind == "ins":
            insertions[reference_index].append(after)
        if kind != "del":
            hypothesis_index += 1
    assert hypothesis_index == len(hyp)
    flags = []
    for kind, start, end in spans:
        mismatch = any(index in changed for index in range(start, end))
        mismatch |= any(index in insertions for index in range(start + 1, end))
        if kind == "number":
            mismatch |= any(NUMBER.fullmatch(char) for edge in (start, end) for char in insertions.get(edge, ()))
        flags.append((kind, ref[start:end], mismatch))
    return pronunciations, flags, novel_repetitions(ref, hyp) if row["errors"] else []


def rate(errors, count):
    return 100 * errors / count if count else None


def mean_sd(values):
    return {"mean": mean(values), "sample_sd": stdev(values)}


def measure(root, output):
    if __version__ != PINYIN_VERSION:
        raise ValueError(f"Expected pypinyin {PINYIN_VERSION}, found {__version__}")
    raw, paths, _, manifest_hash = load_replication(root)
    base_audit = output / "analysis.json"
    audit = json.loads(base_audit.read_text())
    if audit["test_manifest_sha256"] != manifest_hash:
        raise ValueError("Existing character audit uses a different manifest")
    with (root / "runs/zh/manifests/test.csv").open() as handle:
        speakers = {row["ID"]: row["spk_id"] for row in csv.DictReader(handle)}
    canonical = raw["fixed", SEEDS[0]]
    spans = {uid: reference_spans(row["ref_words"]) for uid, row in canonical.items()}
    targets = Counter(kind for annotated in spans.values() for kind, _, _ in annotated)
    per_seed, speakers_by_arm, lengths_by_arm = defaultdict(list), defaultdict(dict), defaultdict(dict)
    details_for_examples, sample_candidates = {}, defaultdict(list)
    all_pairs = defaultdict(Counter)

    for (arm, seed), rows in raw.items():
        score, details = score_run(rows, "")
        expected = next(row for row in audit["conventions"]["characters_without_spaces"]["per_seed"][arm]
                        if row["seed"] == seed)
        for key in ("errors", "ref_characters", "sub", "dele", "ins"):
            if score[key] != expected[key]:
                raise ValueError(f"Character scoring disagrees with prior audit: {arm} {seed} {key}")
        pronunciation_counts, span_errors, repeat_units = Counter(), Counter(), Counter()
        speaker_rows, length_rows = defaultdict(Counter), defaultdict(Counter)
        repeat_utterances = 0
        for uid, row in details.items():
            pronunciations, flags, units = inspect_utterance(row, spans[uid])
            for event in pronunciations:
                pronunciation_counts[event[0]] += 1
                all_pairs[arm][event] += 1
            span_errors.update(kind for kind, _, mismatch in flags if mismatch)
            repeat_utterances += bool(units)
            repeat_units.update(units)
            for counter in (speaker_rows[speakers[uid]], length_rows[length_label(row["n"])]):
                counter.update(errors=row["errors"], characters=row["n"], utterances=1)
            if seed == SEEDS[0]:
                details_for_examples[arm, uid] = (row["hyp"], row["errors"])
                for event in pronunciations:
                    if event[0] in PRONUNCIATION_CLASSES[:2]:
                        sample_candidates[event[0]].append((arm, uid, list(event[1:])))
                for kind, text, mismatch in flags:
                    if mismatch:
                        sample_candidates[kind].append((arm, uid, text))
                if units:
                    sample_candidates["repetition"].append((arm, uid, units))
        assert sum(pronunciation_counts.values()) == score["sub"]
        scored = score["sub"] - pronunciation_counts["unscored"]
        per_seed[arm].append({
            "seed": seed, "reference_characters": score["ref_characters"],
            "substitutions": score["sub"], "pronunciation_counts": dict(pronunciation_counts),
            "pronunciation_rates": {key: rate(pronunciation_counts[key], score["ref_characters"])
                                    for key in PRONUNCIATION_CLASSES},
            "same_tone_share_of_scored_substitutions": rate(pronunciation_counts["same_syllable_and_tone"], scored),
            "span_errors": {key: span_errors[key] for key in SPAN_TYPES},
            "span_error_rates": {key: rate(span_errors[key], targets[key]) for key in SPAN_TYPES},
            "utterances_with_novel_repetition": repeat_utterances,
            "novel_repetition_utterance_rate": rate(repeat_utterances, len(rows)),
            "top_repeated_units": repeat_units.most_common(10),
        })
        speakers_by_arm[arm][seed], lengths_by_arm[arm][seed] = speaker_rows, length_rows

    def stratify(grouped, group_ids):
        return [{"group": group, "utterances": grouped["fixed"][SEEDS[0]][group]["utterances"],
                 "characters": grouped["fixed"][SEEDS[0]][group]["characters"],
                 "systems": {arm: mean_sd([rate(grouped[arm][seed][group]["errors"],
                                                 grouped[arm][seed][group]["characters"]) for seed in SEEDS])
                             for arm in ARMS}} for group in group_ids]

    speaker_results = stratify(speakers_by_arm, sorted(set(speakers.values())))
    length_results = stratify(lengths_by_arm, LENGTH_LABELS)
    result = {
        "status": "complete", "rate_channel": "aux", "pypinyin_version": __version__, "gpus": 0,
        "seeds": SEEDS, "test_utterances": len(canonical), "test_manifest_sha256": manifest_hash,
        "analysis_sha256": sha256(base_audit), "source_files": audit["source_files"],
        "reference_spans_per_run": {key: targets[key] for key in SPAN_TYPES},
        "name_lexicon": NAME_LEXICON, "per_seed": per_seed,
        "speaker_groups": speaker_results, "length_groups": length_results,
        "top_pronunciation_pairs": {arm: [{"class": key[0], "reference": key[1], "hypothesis": key[2],
                                           "reference_pinyin": key[3], "hypothesis_pinyin": key[4], "count": count}
                                          for key, count in all_pairs[arm].most_common(25)] for arm in ARMS},
    }
    examples = {kind: [{"arm": arm, "id": uid, "tag": tag, "reference": "".join(canonical[uid]["ref_words"]),
                        "hypotheses": {system: details_for_examples[system, uid] for system in ARMS}}
                       for arm, uid, tag in entries] for kind, entries in sample_candidates.items()}
    selected_examples = []
    for category, arm, uid, title, comment in EXAMPLE_SELECTION:
        case = next(case for case in examples[category] if case["arm"] == arm and case["id"] == uid)
        selected_examples.append({**case, "category": category, "title": title, "comment": comment})
    result["illustrative_examples"] = selected_examples
    output.mkdir(parents=True, exist_ok=True)
    (output / "linguistic_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    # Keep the full candidate pool local. The report carries the selected cases.
    (output / "linguistic_candidates.json").write_text(json.dumps(examples, ensure_ascii=False) + "\n")
    print(json.dumps({"test_utterances": len(canonical), "speakers": len(speaker_results),
                      "annotated_reference_spans": result["reference_spans_per_run"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    measure(args.replication_root, args.output)
