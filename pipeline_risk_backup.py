from __future__ import annotations

from pipeline_features import *

def get_signal(score: float) -> str:
    if pd.isna(score):
        return "無法判斷，分數為空值"
    if score >= 0.70:
        return "投資吸引力較高"
    if score >= 0.55:
        return "中性偏多"
    if score >= 0.45:
        return "中性"
    if score >= 0.30:
        return "偏弱，建議觀察"
    return "風險偏高，不建議積極進場"


def score_to_signal(score: float) -> str:
    return get_signal(score)


def generate_decision_explanation(row) -> dict:
    score = row.get("predicted_target_score", np.nan)
    signal = row.get("signal", score_to_signal(score) if pd.notna(score) else "無法判斷")

    if pd.notna(score) and score >= 0.70:
        score_text = "模型分數偏高"
    elif pd.notna(score) and score >= 0.55:
        score_text = "模型分數中性偏多"
    elif pd.notna(score) and score >= 0.45:
        score_text = "模型分數接近中性"
    elif pd.notna(score) and score >= 0.30:
        score_text = "模型分數偏弱"
    else:
        score_text = "模型分數偏低"

    technical_parts = []

    rsi = row.get("RSI14", np.nan)
    if pd.notna(rsi):
        if rsi >= 70:
            technical_parts.append("RSI 偏高，短線有過熱疑慮")
        elif rsi >= 55:
            technical_parts.append("RSI 呈中性偏強")
        elif rsi <= 30:
            technical_parts.append("RSI 偏低，動能較弱但可能接近超賣")
        elif rsi <= 45:
            technical_parts.append("RSI 呈中性偏弱")

    macd = row.get("MACD", np.nan)
    if pd.notna(macd):
        technical_parts.append("MACD 偏多" if macd > 0 else "MACD 偏弱")

    k_value = row.get("K", np.nan)
    d_value = row.get("D", np.nan)
    if pd.notna(k_value) and pd.notna(d_value):
        technical_parts.append("K 值高於 D 值，短線動能偏多" if k_value > d_value else "K 值低於 D 值，短線動能偏弱")

    bias24 = row.get("BIAS24", np.nan)
    if pd.notna(bias24):
        if bias24 > 8:
            technical_parts.append("BIAS24 偏高，需留意追高風險")
        elif bias24 < -8:
            technical_parts.append("BIAS24 偏低，價格相對均線偏離較大")

    fib_zone = row.get("fib_zone", None)
    if pd.notna(fib_zone) and str(fib_zone) not in {"", "none", "nan"}:
        technical_parts.append(f"Fib 位置為 {fib_zone}")

    resistance_distance = row.get("resistance_distance", np.nan)
    if pd.notna(resistance_distance) and resistance_distance < 0.03:
        technical_parts.append("上方壓力距離較近")

    support_distance = row.get("support_distance", np.nan)
    if pd.notna(support_distance) and support_distance < 0.03:
        technical_parts.append("價格接近支撐區")

    technical_reason = "；".join(technical_parts) if technical_parts else "技術指標未出現明顯單邊訊號。"

    risk_parts = []

    atr_pct = row.get("atr_pct", np.nan)
    if pd.notna(atr_pct):
        if atr_pct >= 0.05:
            risk_parts.append("ATR 波動偏高")
        elif atr_pct <= 0.02:
            risk_parts.append("ATR 波動相對溫和")

    risk_stoploss_pct = row.get("risk_stoploss_pct", np.nan)
    if pd.notna(risk_stoploss_pct):
        if risk_stoploss_pct <= 0:
            risk_parts.append("停損距離異常，需檢查風險欄位")
        elif risk_stoploss_pct < 0.03:
            risk_parts.append("停損距離較小，風險較容易控制")
        elif risk_stoploss_pct < 0.06:
            risk_parts.append("停損距離中等，需控制部位")
        else:
            risk_parts.append("停損距離偏大，風險成本較高")

    risk_atr_distance = row.get("risk_atr_stop_distance_pct", np.nan)
    if pd.notna(risk_atr_distance) and risk_atr_distance > 0.06:
        risk_parts.append("ATR 停損距離偏遠")

    risk_stop_distance = row.get("risk_stoploss_distance_pct", np.nan)
    if pd.notna(risk_stop_distance) and risk_stop_distance > 0.06:
        risk_parts.append("整體停損距離偏遠")

    risk_valid = row.get("risk_valid", np.nan)
    if pd.notna(risk_valid) and int(risk_valid) == 0:
        risk_parts.append("目前停損設定無效")

    risk_reason = "；".join(risk_parts) if risk_parts else "風險欄位未顯示明顯異常，但仍需設定停損。"

    if pd.notna(score) and score >= 0.70:
        action_suggestion = "可列入觀察名單，等待量價或支撐確認。"
    elif pd.notna(score) and score >= 0.55:
        action_suggestion = "可保守觀察，等待回測支撐或突破確認。"
    elif pd.notna(score) and score >= 0.45:
        action_suggestion = "模型接近中性，建議觀察，不宜單靠模型積極進場。"
    elif pd.notna(score) and score >= 0.30:
        action_suggestion = "暫不建議追高，等待動能轉強或風險下降。"
    else:
        action_suggestion = "風險偏高，建議降低部位或暫時觀望。"

    if pd.notna(risk_stoploss_pct) and risk_stoploss_pct >= 0.06:
        action_suggestion = "暫不建議追高，等待回測支撐或風險下降。"
    if pd.notna(risk_valid) and int(risk_valid) == 0:
        action_suggestion = "風險欄位尚未有效，建議暫時觀望。"

    decision_summary = f"{score_text}，模型判斷為{signal}，但仍需搭配技術面與風險控管確認。"

    return {
        "decision_summary": decision_summary,
        "technical_reason": technical_reason,
        "risk_reason": risk_reason,
        "action_suggestion": action_suggestion,
    }

def positive_return_rate(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return np.nan
    return float((valid > 0).mean())


def calculate_risk_adjusted_return_10d_capped(
    future_return_10d: pd.Series,
    abs_future_drawdown_10d: pd.Series,
) -> pd.Series:
    future_return = pd.to_numeric(future_return_10d, errors="coerce")
    denominator = pd.to_numeric(abs_future_drawdown_10d, errors="coerce").abs()
    denominator = denominator.where(denominator.isna() | (denominator >= 0.01), 0.01)
    return (
        future_return
        .divide(denominator)
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=-5, upper=5)
    )


def normalize_text_for_report(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s*；\s*", "；", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_diagnostic_text(value):
    return normalize_text_for_report(value)


def format_decision_score(value) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "NA"
    return f"{numeric_value:.2f}"


def format_decision_pct(value) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "NA"
    return f"{numeric_value * 100:.2f}%"


def format_report_date(value) -> str:
    date_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(date_value):
        return "NA"
    return date_value.strftime("%Y-%m-%d")


def format_report_price(value) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "NA"
    return f"{numeric_value:.2f}"


def format_report_value(value, decimals: int = 2, pct: bool = False) -> str:
    if pd.isna(value):
        return "NA"
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return normalize_text_for_report(value)
    if pct:
        return f"{numeric_value * 100:.{decimals}f}%"
    return f"{numeric_value:.{decimals}f}"


def format_bias_value(value, decimals: int = 2) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "NA"
    return f"{numeric_value:.{decimals}f}%"


FINAL_SCORE_WEIGHT_PROFILE = "low_model_weight_v1"
FINAL_SCORE_WEIGHTS = {
    "safety_score": 0.30,
    "return_score": 0.20,
    "rule_filter_score": 0.25,
    "risk_reward_score": 0.25,
}
FINAL_SCORE_FORMULA_TEXT = (
    "final_decision_score = 0.30 * safety_score + 0.20 * return_score "
    "+ 0.25 * rule_filter_score + 0.25 * risk_reward_score"
)
FINAL_SCORE_FORMULA_EXPLANATION = (
    "由於第一層模型在 walk-forward 驗證中未穩定優於 baseline，因此本系統降低模型分數權重，"
    "並提高第二層規則式過濾與第三層風險報酬比之權重，使最終決策更偏向風險控管與交易可行性。"
)
MIN_PRACTICAL_STOP_PCT = 0.012
MIN_PREFERRED_STOP_PCT = 0.015


def validate_final_score_weights() -> None:
    weight_sum = sum(FINAL_SCORE_WEIGHTS.values())
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"FINAL_SCORE_WEIGHTS must sum to 1.0; got {weight_sum:.6f}")


ENHANCED_DECISION_COLUMNS = [
    "rule_filter_score",
    "rule_penalty_points",
    "technical_filter_result",
    "rule_filter_notes",
    "entry_price",
    "stop_loss_price_adjusted",
    "stop_loss_pct",
    "take_profit_price",
    "risk_per_share",
    "stop_loss_too_close_flag",
    "practical_risk_per_share",
    "reward_per_share",
    "risk_reward_ratio",
    "practical_risk_reward_ratio",
    "risk_reward_score",
    "practical_risk_reward_score",
    "trade_veto_flag",
    "trade_veto_reason",
    "trade_plan_notes",
    "final_decision_score",
    "final_score_weight_profile",
    "score_band",
    "decision_action",
    "risk_per_trade_pct",
    "max_position_pct",
    "suggested_position_pct",
    "suggested_shares",
    "position_sizing_label",
    "position_sizing_reason",
    "support_entry_price",
    "support_entry_zone_low",
    "support_entry_zone_high",
    "breakout_entry_price",
    "breakout_entry_zone_low",
    "breakout_entry_zone_high",
    "entry_stop_loss_price",
    "entry_stop_loss_pct",
    "entry_take_profit_1",
    "entry_take_profit_2",
    "entry_rr_1",
    "entry_rr_2",
    "entry_condition_text",
    "entry_support_source",
    "support_quality_warning",
    "entry_plan_override",
    "entry_plan_override_reason",
]

ENHANCED_DECISION_TEXT_COLUMNS = [
    "enhanced_decision_summary_text",
    "enhanced_decision_reason_text",
    "enhanced_suggested_action_text",
    "enhanced_decision_report_text",
]


def _row_number(row: pd.Series, key: str, default=np.nan) -> float:
    value = row.get(key, default)
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return numeric_value


def _clip01(value, default=np.nan):
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return default
    return float(np.clip(numeric_value, 0, 1))


def _join_notes(notes: list[str]) -> str:
    clean_notes = [str(note).strip() for note in notes if str(note).strip()]
    return "; ".join(clean_notes)


def _has_material_warning(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = normalize_text_for_report(str(value)).strip()
    if not text:
        return False
    lower_text = text.lower()
    if lower_text in {"na", "nan", "none", "no warning", "no material warning", "no obvious risk"}:
        return False
    if "no obvious" in lower_text or "no material" in lower_text:
        return False
    if "無" in text and ("風險" in text or "警示" in text):
        return False
    return True


def translate_veto_reason(reason) -> str:
    if reason is None:
        return ""
    try:
        if pd.isna(reason):
            return ""
    except (TypeError, ValueError):
        pass

    reason_text = normalize_text_for_report(str(reason))
    if not reason_text:
        return ""

    translations = {
        "rr_too_low": "風險報酬比低於 1.5",
        "risk/reward below 1.5": "風險報酬比低於 1.5",
        "stop_loss_too_far": "停損距離過遠",
        "stop loss wider than 8%": "停損距離過遠",
        "invalid_stop_loss": "無有效停損價",
        "no valid stop within 8%": "無有效停損價",
        "invalid_risk_reward": "風險報酬結構無效",
        "invalid risk or reward": "風險報酬結構無效",
        "invalid_reward": "上方報酬空間不足",
        "invalid_risk": "停損風險設定無效",
        "invalid entry price": "無有效進場價",
        "risk plan failed": "交易風險計畫未通過",
    }

    translated_parts = []
    for part in re.split(r"[;；]+", reason_text):
        clean_part = normalize_text_for_report(part)
        if not clean_part:
            continue
        translated_parts.append(translations.get(clean_part.lower(), clean_part))

    return "；".join(dict.fromkeys(translated_parts))


def translate_decision_label(label) -> str:
    if label is None:
        return ""
    try:
        if pd.isna(label):
            return ""
    except (TypeError, ValueError):
        pass
    text = normalize_text_for_report(str(label))
    mapping = {
        "Veto": "否決交易",
        "否決交易": "否決交易",
        "No Score": "無分數",
        "Watchlist": "中性觀察",
        "Caution": "保守觀察",
        "Avoid": "避免進場",
        "High Quality Opportunity": "高品質機會，可條件式進場",
        "Bullish Watch": "穩健偏多，優先觀察",
        "Neutral Bullish": "中性偏多，等待確認",
        "Strong Candidate": "強勢候選",
        "Qualified Candidate": "條件式候選",
        "中性觀察": "中性觀察",
        "等待回測進場": "等待回測進場",
    }
    return mapping.get(text, text)


def translate_decision_action(action) -> str:
    if action is None:
        return ""
    try:
        if pd.isna(action):
            return ""
    except (TypeError, ValueError):
        pass
    text = normalize_text_for_report(str(action))
    mapping = {
        "No trade": "不建議進場",
        "不建議進場": "不建議進場",
        "No action": "暫無明確操作建議",
        "Wait for confirmation": "等待確認，不建議直接追價",
        "Observe only": "觀察為主",
        "Priority watch": "優先觀察，等待確認",
        "Do not enter": "不建議進場",
        "Consider entry with normal risk control": "可條件式進場",
        "Consider small to normal position": "可條件式進場",
        "不追收盤價，等待價格回測至支撐區間後進場": "不追收盤價，等待價格回測至支撐區間後進場",
    }
    if text in mapping:
        return mapping[text]
    lower_text = text.lower()
    if any(keyword in lower_text for keyword in ["buy", "entry allowed", "consider entry", "enter"]):
        return "可條件式進場"
    return text


def translate_technical_filter_result(result) -> str:
    if result is None:
        return ""
    try:
        if pd.isna(result):
            return ""
    except (TypeError, ValueError):
        pass
    text = normalize_text_for_report(str(result))
    mapping = {
        "pass": "技術條件良好",
        "watch": "技術條件中性，需觀察",
        "fail": "技術條件偏弱",
        "技術條件良好": "技術條件良好",
        "技術條件中性，需觀察": "技術條件中性，需觀察",
        "技術條件偏弱": "技術條件偏弱",
    }
    return mapping.get(text.lower(), mapping.get(text, text))


def _translate_note_parts(notes, translations: dict[str, str]) -> str:
    if notes is None:
        return ""
    try:
        if pd.isna(notes):
            return ""
    except (TypeError, ValueError):
        pass
    text = normalize_text_for_report(str(notes))
    if not text:
        return ""

    translated = []
    for part in re.split(r"[;；]+", text):
        clean_part = normalize_text_for_report(part)
        if not clean_part:
            continue
        lower_part = clean_part.lower()
        replacement = translations.get(clean_part) or translations.get(lower_part)
        if replacement is None:
            for key, value in translations.items():
                if key.lower() in lower_part:
                    replacement = value
                    break
        translated.append(replacement or clean_part)
    return "；".join(dict.fromkeys(translated))


def translate_rule_filter_notes(notes) -> str:
    translations = {
        "RSI14 in constructive neutral range": "RSI 位於健康中性區間",
        "RSI14 positive but near overheat": "RSI 接近過熱",
        "RSI14 extremely overbought": "RSI 明顯過熱",
        "RSI14 overbought": "RSI 過熱",
        "RSI14 oversold or weak": "RSI 偏弱或接近超賣",
        "RSI14 weak": "RSI 偏弱",
        "MACD positive": "MACD 偏多",
        "MACD negative": "MACD 偏弱",
        "DIF above DEA": "DIF 高於 DEA",
        "DIF below DEA": "DIF 低於 DEA",
        "KDJ bullish crossover without overheat": "KDJ 短線偏多且尚未過熱",
        "KDJ not yet bullish": "KDJ 尚未轉強",
        "KDJ J value overheated": "J 值過熱",
        "KDJ J value weak": "J 值偏弱",
        "BIAS24 in moderate range": "BIAS 乖離適中",
        "BIAS24 too extended": "BIAS 偏離過大",
        "BIAS24 elevated": "BIAS 偏高",
        "BIAS24 very weak": "BIAS 偏弱",
        "balanced": "VR 量能平衡",
        "overheated": "量能或動能過熱",
        "weak volume confirmation": "量能確認偏弱",
        "fib_position in lower half of range": "價格位於近期區間下半部",
        "fib_position in middle range": "價格位於近期區間中段",
        "fib_position near range high": "價格接近近期高位",
        "fib_position elevated": "價格位置偏高",
        "price close to support": "價格接近支撐",
        "price far from support": "距離支撐偏遠",
        "support distance is wide": "距離支撐偏遠",
        "resistance too close": "上方壓力接近",
        "limited upside before resistance": "上方空間有限",
        "resistance distance leaves upside room": "上方仍有空間",
        "ATR percent in manageable range": "ATR 波動仍可控",
        "ATR percent too high": "ATR 波動過高",
        "ATR percent elevated": "ATR 波動偏高",
        "ATR percent very low": "ATR 波動過低",
        "risk warning requires protection": "風險警示需優先控管",
        "model confidence note requires caution": "模型信心提示需保守看待",
        "no rule signal": "無明顯規則訊號",
    }
    return _translate_note_parts(notes, translations)


def translate_trade_plan_notes(notes) -> str:
    translations = {
        "trade plan cannot be built without a valid close price": "缺少有效收盤價，無法建立交易計畫",
        "selected support_stop within 1.5%-6% stop range": "停損採用支撐位，位於 1.5%-6% 合理區間",
        "selected atr_stop within 1.5%-6% stop range": "停損採用 ATR 停損，位於 1.5%-6% 合理區間",
        "selected old_stop within 1.5%-6% stop range": "停損採用原始風險欄位，位於 1.5%-6% 合理區間",
        "selected support_stop outside preferred stop range": "停損採用支撐位，但不在偏好停損區間",
        "selected atr_stop outside preferred stop range": "停損採用 ATR 停損，但不在偏好停損區間",
        "selected old_stop outside preferred stop range": "停損採用原始風險欄位，但不在偏好停損區間",
        "stop is very close": "停損過近",
        "whipsaw risk is high": "容易受到正常波動洗出",
        "stop is very close; whipsaw risk is high": "停損過近，易受震盪掃出",
        "stop loss too close; whipsaw risk": "停損距離過近，容易受到正常波動洗出",
        "all available stops are invalid or wider than 8%": "停損候選無效或超過 8%",
        "no support, ATR, or legacy stop candidate available": "未取得有效支撐、ATR 或原始停損候選",
        "take profit uses nearest resistance": "目標價採用最近壓力位",
        "no valid resistance; take profit uses 2R": "未偵測到有效壓力位，目標價由 2R 推估，需人工確認上方空間。",
        "no valid resistance": "未偵測到有效壓力位，目標價由 2R 推估，需人工確認上方空間。",
        "take profit uses 2R": "未偵測到有效壓力位，目標價由 2R 推估，需人工確認上方空間。",
        "risk/reward below 2.0": "風險報酬比低於 2.0",
        "wide stop": "停損距離偏大，需降低部位",
        "position size should be reduced": "停損距離偏大，需降低部位",
        "wide stop; position size should be reduced": "停損距離偏大，需降低部位",
        "stop is too close for normal volatility": "停損相對正常波動過近",
        "停損距離過近，容易受到正常波動洗出": "停損距離過近，容易受到正常波動洗出",
        "nearest resistance is close": "上方壓力接近",
    }
    return _translate_note_parts(notes, translations)


def build_risk_warning(row: pd.Series) -> str:
    warnings = []

    atr_pct = _row_number(row, "atr_pct")
    if pd.notna(atr_pct) and atr_pct > 0.05:
        warnings.append("ATR 波動過高，短線風險成本偏大")

    fib_position = _row_number(row, "fib_position")
    if pd.notna(fib_position) and fib_position > 0.786:
        warnings.append("價格接近近期高位，追高風險較高")

    j_value = _row_number(row, "J")
    if pd.notna(j_value) and j_value > 90:
        warnings.append("J 值過熱，短線可能震盪")

    support_distance = _row_number(row, "support_distance")
    if pd.notna(support_distance) and support_distance > 0.05:
        warnings.append("距離支撐偏遠，若回落修正空間較大")

    stop_loss_pct = _row_number(row, "stop_loss_pct")
    if pd.notna(stop_loss_pct) and stop_loss_pct > 0.05:
        warnings.append("停損距離偏大，需降低部位")
    stop_loss_too_close_flag = str(row.get("stop_loss_too_close_flag", False)).strip().lower() in {"true", "1", "yes", "y"}
    if stop_loss_too_close_flag or (pd.notna(stop_loss_pct) and stop_loss_pct < MIN_PRACTICAL_STOP_PCT):
        warnings.append("停損距離過近，容易受到正常波動洗出")

    if "stop_loss_price_adjusted" in row.index:
        stop_loss_price_adjusted = _row_number(row, "stop_loss_price_adjusted")
        veto_reason_text = normalize_text_for_report(row.get("trade_veto_reason", ""))
        if pd.isna(stop_loss_price_adjusted) or "invalid_stop_loss" in veto_reason_text or "無有效停損價" in veto_reason_text:
            warnings.append("無有效停損價，交易計畫無法成立")

    if "risk_reward_ratio" in row.index:
        risk_reward_ratio = _row_number(row, "risk_reward_ratio")
        trade_veto_value = row.get("trade_veto_flag", False)
        try:
            trade_veto_flag = False if pd.isna(trade_veto_value) else bool(trade_veto_value)
        except (TypeError, ValueError):
            trade_veto_flag = bool(trade_veto_value)
        if pd.isna(risk_reward_ratio) and trade_veto_flag:
            warnings.append("風險報酬比無法有效計算")

    nearest_resistance_level = _row_number(row, "nearest_resistance_level")
    if pd.isna(nearest_resistance_level):
        warnings.append("未偵測到有效壓力位，目標價需人工確認")

    rule_filter_score = _row_number(row, "rule_filter_score")
    if pd.notna(rule_filter_score) and rule_filter_score < 0.50:
        warnings.append("第二層技術過濾偏弱")

    macd = _row_number(row, "MACD")
    if pd.notna(macd) and macd < 0:
        warnings.append("趨勢動能偏弱")

    return clean_diagnostic_text("；".join(dict.fromkeys(warnings)) if warnings else "無明顯風險警示")


def _safe_rate(numerator, denominator) -> float:
    numerator = pd.to_numeric(pd.Series([numerator]), errors="coerce").iloc[0]
    denominator = pd.to_numeric(pd.Series([denominator]), errors="coerce").iloc[0]
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def apply_second_layer_rule_filter(row: pd.Series) -> pd.Series:
    score = 0.50
    penalty_points = 0.0
    notes = []

    def add_rule(delta: float, note: str) -> None:
        nonlocal score, penalty_points
        score += delta
        if delta < 0:
            penalty_points += abs(delta)
        notes.append(note)

    rsi14 = _row_number(row, "RSI14")
    if pd.notna(rsi14):
        if 45 <= rsi14 <= 65:
            add_rule(0.08, "RSI14 in constructive neutral range")
        elif 65 < rsi14 <= 70:
            add_rule(0.03, "RSI14 positive but near overheat")
        elif rsi14 > 75:
            add_rule(-0.12, "RSI14 extremely overbought")
        elif rsi14 > 70:
            add_rule(-0.08, "RSI14 overbought")
        elif rsi14 < 30:
            add_rule(-0.10, "RSI14 oversold or weak")
        elif rsi14 < 40:
            add_rule(-0.05, "RSI14 weak")

    macd = _row_number(row, "MACD")
    dif = _row_number(row, "DIF")
    dea = _row_number(row, "DEA")
    if pd.notna(macd):
        add_rule(0.06 if macd > 0 else -0.06, "MACD positive" if macd > 0 else "MACD negative")
    if pd.notna(dif) and pd.notna(dea):
        add_rule(0.04 if dif > dea else -0.04, "DIF above DEA" if dif > dea else "DIF below DEA")

    k_value = _row_number(row, "K")
    d_value = _row_number(row, "D")
    j_value = _row_number(row, "J")
    if pd.notna(k_value) and pd.notna(d_value):
        if k_value > d_value and k_value < 80:
            add_rule(0.04, "KDJ bullish crossover without overheat")
        elif k_value <= d_value:
            add_rule(-0.04, "KDJ not yet bullish")
    if pd.notna(j_value):
        if j_value > 100:
            add_rule(-0.08, "KDJ J value overheated")
        elif j_value < 0:
            add_rule(-0.06, "KDJ J value weak")

    bias24 = _row_number(row, "BIAS24")
    if pd.notna(bias24):
        if -3 <= bias24 <= 6:
            add_rule(0.05, "BIAS24 in moderate range")
        elif bias24 > 12:
            add_rule(-0.10, "BIAS24 too extended")
        elif bias24 > 8:
            add_rule(-0.06, "BIAS24 elevated")
        elif bias24 < -10:
            add_rule(-0.05, "BIAS24 very weak")

    for vr_col in ["VR_14", "VR_26"]:
        vr_value = _row_number(row, vr_col)
        if pd.isna(vr_value):
            continue
        if 80 <= vr_value <= 180:
            add_rule(0.02, f"{vr_col} balanced")
        elif vr_value > 250:
            add_rule(-0.05, f"{vr_col} overheated")
        elif vr_value < 60:
            add_rule(-0.04, f"{vr_col} weak volume confirmation")

    fib_position = _row_number(row, "fib_position")
    if pd.notna(fib_position):
        if fib_position < 0.50:
            add_rule(0.06, "fib_position in lower half of range")
        elif fib_position <= 0.786:
            add_rule(0.03, "fib_position in middle range")
        elif fib_position > 0.90:
            add_rule(-0.08, "fib_position near range high")
        else:
            add_rule(-0.04, "fib_position elevated")

    support_distance = _row_number(row, "support_distance")
    if pd.notna(support_distance):
        if 0 <= support_distance < 0.03:
            add_rule(0.04, "price close to support")
        elif support_distance > 0.08:
            add_rule(-0.08, "price far from support")
        elif support_distance > 0.06:
            add_rule(-0.05, "support distance is wide")

    resistance_distance = _row_number(row, "resistance_distance")
    if pd.notna(resistance_distance):
        if 0 <= resistance_distance < 0.03:
            add_rule(-0.10, "resistance too close")
        elif resistance_distance < 0.06:
            add_rule(-0.05, "limited upside before resistance")
        elif resistance_distance > 0.10:
            add_rule(0.05, "resistance distance leaves upside room")

    atr_pct = _row_number(row, "atr_pct")
    if pd.notna(atr_pct):
        if 0.015 <= atr_pct <= 0.035:
            add_rule(0.03, "ATR percent in manageable range")
        elif atr_pct > 0.06:
            add_rule(-0.10, "ATR percent too high")
        elif atr_pct > 0.04:
            add_rule(-0.06, "ATR percent elevated")
        elif atr_pct < 0.015:
            add_rule(-0.03, "ATR percent very low")

    if _has_material_warning(row.get("risk_warning", "")):
        add_rule(-0.08, "risk warning requires protection")
    if _has_material_warning(row.get("model_confidence_note", "")):
        add_rule(-0.04, "model confidence note requires caution")

    score = _clip01(score, default=0.50)
    penalty_points = _clip01(penalty_points, default=0.0)
    if score >= 0.65:
        technical_filter_result = "pass"
    elif score >= 0.45:
        technical_filter_result = "watch"
    else:
        technical_filter_result = "fail"

    return pd.Series({
        "rule_filter_score": score,
        "rule_penalty_points": penalty_points,
        "technical_filter_result": technical_filter_result,
        "rule_filter_notes": translate_rule_filter_notes(_join_notes(notes) if notes else "no rule signal"),
    })


def build_trade_plan(row: pd.Series) -> pd.Series:
    entry_price = _row_number(row, "close")
    if pd.isna(entry_price) or entry_price <= 0:
        return pd.Series({
            "entry_price": entry_price,
            "stop_loss_price_adjusted": np.nan,
            "stop_loss_pct": np.nan,
            "take_profit_price": np.nan,
            "risk_per_share": np.nan,
            "stop_loss_too_close_flag": False,
            "practical_risk_per_share": np.nan,
            "reward_per_share": np.nan,
            "risk_reward_ratio": np.nan,
            "practical_risk_reward_ratio": np.nan,
            "risk_reward_score": 0.0,
            "practical_risk_reward_score": 0.0,
            "trade_veto_flag": True,
            "trade_veto_reason": translate_veto_reason("invalid entry price"),
            "trade_plan_notes": translate_trade_plan_notes("trade plan cannot be built without a valid close price"),
        })

    candidates: list[tuple[str, float, float]] = []

    support_stop = _row_number(row, "nearest_support_level")
    support_distance = _row_number(row, "support_distance")
    if (pd.isna(support_stop) or support_stop <= 0) and pd.notna(support_distance) and support_distance > 0:
        support_stop = entry_price / (1 + support_distance)
    if pd.notna(support_stop) and 0 < support_stop < entry_price:
        candidates.append(("support_stop", float(support_stop), float((entry_price - support_stop) / entry_price)))

    risk_atr_stop = _row_number(row, "risk_atr_stop")
    atr_14 = _row_number(row, "atr_14")
    atr_stop = risk_atr_stop
    if pd.isna(atr_stop) and pd.notna(atr_14) and atr_14 > 0:
        atr_stop = entry_price - 2 * atr_14
    if pd.notna(atr_stop) and 0 < atr_stop < entry_price:
        candidates.append(("atr_stop", float(atr_stop), float((entry_price - atr_stop) / entry_price)))

    old_stop = _row_number(row, "risk_stoploss_price")
    old_stop_pct = _row_number(row, "risk_stoploss_pct")
    if (pd.isna(old_stop) or old_stop <= 0) and pd.notna(old_stop_pct) and old_stop_pct > 0:
        old_stop = entry_price * (1 - old_stop_pct)
    if pd.notna(old_stop) and 0 < old_stop < entry_price:
        candidates.append(("old_stop", float(old_stop), float((entry_price - old_stop) / entry_price)))

    notes = []
    veto_reasons = []
    valid_candidates = [item for item in candidates if 0 < item[2] <= 0.08]
    preferred_candidates = [item for item in valid_candidates if MIN_PREFERRED_STOP_PCT <= item[2] <= 0.06]

    if preferred_candidates:
        chosen_name, stop_loss_price, stop_loss_pct = min(
            preferred_candidates,
            key=lambda item: abs(item[2] - 0.04),
        )
        notes.append(f"selected {chosen_name} within 1.5%-6% stop range")
    elif valid_candidates:
        chosen_name, stop_loss_price, stop_loss_pct = min(
            valid_candidates,
            key=lambda item: abs(item[2] - 0.04),
        )
        notes.append(f"selected {chosen_name} outside preferred stop range")
        if stop_loss_pct < MIN_PREFERRED_STOP_PCT:
            notes.append("stop is very close; whipsaw risk is high")
    else:
        stop_loss_price = np.nan
        stop_loss_pct = np.nan
        veto_reasons.append("invalid_stop_loss")
        if candidates:
            notes.append("all available stops are invalid or wider than 8%")
        else:
            notes.append("no support, ATR, or legacy stop candidate available")

    risk_per_share = entry_price - stop_loss_price if pd.notna(stop_loss_price) else np.nan

    resistance_price = _row_number(row, "nearest_resistance_level")
    resistance_distance = _row_number(row, "resistance_distance")
    if (pd.isna(resistance_price) or resistance_price <= entry_price) and pd.notna(resistance_distance) and resistance_distance > 0:
        resistance_price = entry_price * (1 + resistance_distance)

    if pd.notna(risk_per_share) and risk_per_share > 0:
        if pd.notna(resistance_price) and resistance_price > entry_price:
            take_profit_price = float(resistance_price)
            notes.append("take profit uses nearest resistance")
        else:
            take_profit_price = float(entry_price + 2 * risk_per_share)
            notes.append("no valid resistance; take profit uses 2R")
    else:
        take_profit_price = np.nan

    reward_per_share = take_profit_price - entry_price if pd.notna(take_profit_price) else np.nan
    if pd.notna(risk_per_share) and risk_per_share > 0 and pd.notna(reward_per_share):
        risk_reward_ratio = float(reward_per_share / risk_per_share)
    else:
        risk_reward_ratio = np.nan

    stop_loss_too_close_flag = bool(pd.notna(stop_loss_pct) and stop_loss_pct < MIN_PRACTICAL_STOP_PCT)
    if stop_loss_too_close_flag:
        notes.append("停損距離過近，容易受到正常波動洗出")

    if stop_loss_too_close_flag:
        minimum_practical_risk = entry_price * MIN_PREFERRED_STOP_PCT
        practical_risk_per_share = max(risk_per_share, minimum_practical_risk) if pd.notna(risk_per_share) else minimum_practical_risk
    else:
        practical_risk_per_share = risk_per_share

    if pd.notna(practical_risk_per_share) and practical_risk_per_share > 0 and pd.notna(reward_per_share) and reward_per_share > 0:
        practical_risk_reward_ratio = float(reward_per_share / practical_risk_per_share)
    else:
        practical_risk_reward_ratio = np.nan

    if pd.isna(risk_per_share) or risk_per_share <= 0:
        veto_reasons.append("invalid_risk")
    if pd.isna(reward_per_share) or reward_per_share <= 0:
        veto_reasons.append("invalid_reward")
    if pd.notna(stop_loss_pct) and stop_loss_pct > 0.08:
        veto_reasons.append("stop_loss_too_far")
    if pd.notna(practical_risk_reward_ratio) and practical_risk_reward_ratio < 1.5:
        veto_reasons.append("rr_too_low")

    if pd.notna(practical_risk_reward_ratio):
        if practical_risk_reward_ratio < 1.5:
            practical_risk_reward_score = 0.0
        elif practical_risk_reward_ratio < 2.0:
            practical_risk_reward_score = 0.4
            notes.append("risk/reward below 2.0")
        else:
            practical_risk_reward_score = _clip01(0.7 + 0.2 * (practical_risk_reward_ratio - 2.0), default=0.7)
    else:
        practical_risk_reward_score = 0.0
    if stop_loss_too_close_flag:
        practical_risk_reward_score = min(practical_risk_reward_score, 0.40)
    risk_reward_score = practical_risk_reward_score

    if pd.notna(stop_loss_pct):
        if stop_loss_pct > 0.06:
            notes.append("wide stop; position size should be reduced")
        elif stop_loss_pct < MIN_PREFERRED_STOP_PCT:
            notes.append("stop is too close for normal volatility")
    if pd.notna(resistance_distance) and resistance_distance < 0.03:
        notes.append("nearest resistance is close")

    veto_reasons = list(dict.fromkeys(veto_reasons))
    trade_veto_flag = bool(veto_reasons)

    return pd.Series({
        "entry_price": entry_price,
        "stop_loss_price_adjusted": stop_loss_price,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_price": take_profit_price,
        "risk_per_share": risk_per_share,
        "stop_loss_too_close_flag": stop_loss_too_close_flag,
        "practical_risk_per_share": practical_risk_per_share,
        "reward_per_share": reward_per_share,
        "risk_reward_ratio": risk_reward_ratio,
        "practical_risk_reward_ratio": practical_risk_reward_ratio,
        "risk_reward_score": risk_reward_score,
        "practical_risk_reward_score": practical_risk_reward_score,
        "trade_veto_flag": trade_veto_flag,
        "trade_veto_reason": translate_veto_reason(_join_notes(veto_reasons)),
        "trade_plan_notes": translate_trade_plan_notes(_join_notes(notes)),
    })


def build_entry_plan(row: pd.Series) -> pd.Series:
    _nan = np.nan
    close = _row_number(row, "close")

    _empty = pd.Series({
        "support_entry_price": _nan,
        "support_entry_zone_low": _nan,
        "support_entry_zone_high": _nan,
        "breakout_entry_price": _nan,
        "breakout_entry_zone_low": _nan,
        "breakout_entry_zone_high": _nan,
        "entry_stop_loss_price": _nan,
        "entry_stop_loss_pct": _nan,
        "entry_take_profit_1": _nan,
        "entry_take_profit_2": _nan,
        "entry_rr_1": _nan,
        "entry_rr_2": _nan,
        "entry_condition_text": "",
        "entry_support_source": "",
    })

    if pd.isna(close) or close <= 0:
        return _empty

    # ── 一、回測支撐進場點 ────────────────────────────────────────────
    support_candidates: list[tuple[str, float]] = []
    for key in ["nearest_support_level", "ma20", "ma5", "fib_0618_price"]:
        val = _row_number(row, key)
        if pd.notna(val) and 0 < val < close:
            support_candidates.append((key, float(val)))

    if support_candidates:
        _, support_entry_price = min(support_candidates, key=lambda x: close - x[1])
        support_entry_zone_low = round(support_entry_price * 0.995, 4)
        support_entry_zone_high = round(support_entry_price * 1.005, 4)
    else:
        support_entry_price = _nan
        support_entry_zone_low = _nan
        support_entry_zone_high = _nan

    # ── 突破確認進場點 ─────────────────────────────────────────────────
    breakout_candidates: list[tuple[str, float]] = []
    for key in ["nearest_resistance_level", "ma60"]:
        val = _row_number(row, key)
        if pd.notna(val) and val > close:
            breakout_candidates.append((key, float(val)))

    if breakout_candidates:
        _, breakout_entry_price = min(breakout_candidates, key=lambda x: x[1] - close)
        breakout_entry_zone_low = round(breakout_entry_price * 1.003, 4)
        breakout_entry_zone_high = round(breakout_entry_price * 1.010, 4)
    else:
        breakout_entry_price = _nan
        breakout_entry_zone_low = _nan
        breakout_entry_zone_high = _nan

    # 止損/止盈以支撐進場點為基準；若無則用突破進場點；若皆無則用收盤價
    ref_entry = (
        support_entry_price if pd.notna(support_entry_price)
        else breakout_entry_price if pd.notna(breakout_entry_price)
        else close
    )

    # ── 二、止損點（有效範圍：距進場點 1.2% ~ 8%）─────────────────────
    close_to_20d_low = _row_number(row, "close_to_20d_low")
    low_20d = close / (1 + close_to_20d_low) if (pd.notna(close_to_20d_low) and close_to_20d_low > -1) else _nan

    _stop_source_labels = {
        "support_stop":    "支撐位",
        "atr_stop":        "ATR 止損",
        "ma20_stop":       "20日均線",
        "low_20d_stop":    "近20日最低點",
        "vol_high1_stop":  "最大量日低點",
        "vol_high2_stop":  "次大量日低點",
        "pivot_low_stop":  "Pivot Low",
        "fvg_lower_stop":  "FVG 下緣",
    }
    stop_candidates: list[tuple[str, float, float]] = []
    for key, val in [
        ("support_stop",   _row_number(row, "nearest_support_level")),
        ("atr_stop",       _row_number(row, "risk_atr_stop")),
        ("ma20_stop",      _row_number(row, "ma20")),
        ("low_20d_stop",   low_20d),
        ("vol_high1_stop", _row_number(row, "vol_high1_low")),
        ("vol_high2_stop", _row_number(row, "vol_high2_low")),
        ("pivot_low_stop", _row_number(row, "pivot_low")),
        ("fvg_lower_stop", _row_number(row, "recent_active_gap_lower")),
    ]:
        if pd.notna(val) and 0 < val < ref_entry:
            pct = (ref_entry - val) / ref_entry
            if 0.012 <= pct <= 0.08:
                stop_candidates.append((key, float(val), float(pct)))

    if stop_candidates:
        stop_source, stop_loss_price, stop_loss_pct = min(stop_candidates, key=lambda x: x[2])
        entry_support_source = _stop_source_labels.get(stop_source, stop_source)
    else:
        # 所有候選均不在 1.2%~8% 範圍內，啟用 2% 保底止損
        stop_loss_price = float(ref_entry * (1 - 0.02))
        stop_loss_pct = 0.02
        entry_support_source = "2%保底"

    # ── 三、止盈點與 R/R ─────────────────────────────────────────────
    risk = ref_entry - stop_loss_price

    resistance_level = _row_number(row, "nearest_resistance_level")
    if pd.notna(resistance_level) and resistance_level > ref_entry:
        tp1 = float(resistance_level)
    elif risk > 0:
        tp1 = float(ref_entry + 1.5 * risk)
    else:
        tp1 = _nan

    resistance_zone_high = _row_number(row, "nearest_resistance_zone_high")
    if pd.notna(tp1) and pd.notna(resistance_zone_high) and resistance_zone_high > tp1:
        tp2 = float(resistance_zone_high)
    elif risk > 0:
        tp2 = float(ref_entry + 2.5 * risk)
    else:
        tp2 = _nan

    # 強制第二止盈必須大於第一止盈
    if pd.notna(tp2) and pd.notna(tp1) and tp2 <= tp1:
        tp2 = float(ref_entry + 2.5 * risk) if risk > 0 else _nan

    rr1 = float((tp1 - ref_entry) / risk) if (pd.notna(tp1) and risk > 0) else _nan
    rr2 = float((tp2 - ref_entry) / risk) if (pd.notna(tp2) and risk > 0) else _nan

    # ── 四、進場條件說明 ─────────────────────────────────────────────
    rsi14          = _row_number(row, "RSI14")
    macd           = _row_number(row, "MACD")
    k_val          = _row_number(row, "K")
    d_val          = _row_number(row, "D")
    fib_position   = _row_number(row, "fib_position")
    vol_ratio_20   = _row_number(row, "volume_ratio_20")
    res_distance   = _row_number(row, "resistance_distance")
    atr_pct        = _row_number(row, "atr_pct")

    conditions: list[str] = []
    if pd.notna(rsi14):
        if rsi14 > 70:
            conditions.append("需等待 RSI 回落至 70 以下再進場")
        elif rsi14 < 30:
            conditions.append("RSI 已接近超賣，可留意反彈訊號")
    if pd.notna(macd) and macd < 0:
        conditions.append("MACD 仍在零軸下方，建議等待 MACD 翻正確認")
    if pd.notna(k_val) and pd.notna(d_val) and k_val > d_val and k_val < 80:
        conditions.append("KDJ 短線偏多，動能支撐進場")
    if pd.notna(fib_position):
        if fib_position > 0.786:
            conditions.append("價格位於近期高位，回測進場優先於追價")
        elif fib_position < 0.5:
            conditions.append("價格位於區間低位，支撐進場勝率較高")
    if pd.notna(vol_ratio_20) and vol_ratio_20 > 1.5:
        conditions.append("今日成交量放大，需確認量能方向再進場")
    if pd.notna(res_distance) and res_distance < 0.03:
        conditions.append("上方壓力接近，突破進場需等待有效突破確認")
    if pd.notna(atr_pct) and atr_pct > 0.04:
        conditions.append("波動偏高，建議縮小部位或等待波動收斂")

    sqs = _row_number(row, "support_quality_score")
    if pd.isna(sqs):
        support_quality_warning = "支撐品質資料不足"
    elif sqs < 0.4:
        support_quality_warning = "注意：當前支撐位歷史有效率偏低，建議縮小部位或等待量能確認後再進場"
    elif sqs < 0.7:
        support_quality_warning = "支撐位可信度中等，建議搭配量能確認"
    else:
        support_quality_warning = "支撐位歷史有效率高，可信度較佳"

    return pd.Series({
        "support_entry_price":       support_entry_price,
        "support_entry_zone_low":    support_entry_zone_low,
        "support_entry_zone_high":   support_entry_zone_high,
        "breakout_entry_price":      breakout_entry_price,
        "breakout_entry_zone_low":   breakout_entry_zone_low,
        "breakout_entry_zone_high":  breakout_entry_zone_high,
        "entry_stop_loss_price":     stop_loss_price,
        "entry_stop_loss_pct":       stop_loss_pct,
        "entry_take_profit_1":       tp1,
        "entry_take_profit_2":       tp2,
        "entry_rr_1":                rr1,
        "entry_rr_2":                rr2,
        "entry_condition_text":      "；".join(conditions),
        "entry_support_source":      entry_support_source,
        "support_quality_warning":   support_quality_warning,
    })


def get_score_band(
    final_decision_score,
    trade_veto_flag: bool = False,
    entry_plan_override: bool = False,
) -> tuple[str, str]:
    if trade_veto_flag:
        return "Veto", "No trade"
    score = pd.to_numeric(pd.Series([final_decision_score]), errors="coerce").iloc[0]
    if pd.isna(score):
        return "No Score", "No action"
    if entry_plan_override and score >= 0.45:
        return "等待回測進場", "不追收盤價，等待價格回測至支撐區間後進場"
    if score >= 0.80:
        return "High Quality Opportunity", "Consider entry with normal risk control"
    if score >= 0.70:
        return "Bullish Watch", "Priority watch"
    if score >= 0.60:
        return "Neutral Bullish", "Wait for confirmation"
    if score >= 0.50:
        return "Watchlist", "Wait for confirmation"
    if score >= 0.40:
        return "Caution", "Observe only"
    return "Avoid", "Do not enter"


def build_final_decision(row: pd.Series) -> pd.Series:
    validate_final_score_weights()
    combined_score = _row_number(row, "combined_decision_score")
    safety_score = _row_number(row, "safety_score")
    return_score = _row_number(row, "return_score")
    rule_filter_score = _row_number(row, "rule_filter_score")
    risk_reward_score = _row_number(row, "risk_reward_score")
    penalty_points = _row_number(row, "rule_penalty_points")

    if pd.isna(safety_score):
        safety_score = combined_score if pd.notna(combined_score) else 0.50
    if pd.isna(return_score):
        return_score = combined_score if pd.notna(combined_score) else 0.50
    if pd.isna(rule_filter_score):
        rule_filter_score = 0.50
    if pd.isna(risk_reward_score):
        risk_reward_score = 0.0
    if pd.isna(penalty_points):
        penalty_points = 0.0

    final_decision_score = (
        FINAL_SCORE_WEIGHTS["safety_score"] * safety_score
        + FINAL_SCORE_WEIGHTS["return_score"] * return_score
        + FINAL_SCORE_WEIGHTS["rule_filter_score"] * rule_filter_score
        + FINAL_SCORE_WEIGHTS["risk_reward_score"] * risk_reward_score
    )
    final_decision_score -= min(0.10, penalty_points * 0.10)
    final_decision_score = _clip01(final_decision_score, default=0.0)

    caps = []
    technical_filter_result = str(row.get("technical_filter_result", "")).lower()
    if technical_filter_result == "fail" or rule_filter_score < 0.35:
        caps.append(0.49)
    stop_loss_pct = _row_number(row, "stop_loss_pct")
    if pd.notna(stop_loss_pct) and stop_loss_pct > 0.06:
        caps.append(0.69)
    practical_risk_reward_ratio = _row_number(row, "practical_risk_reward_ratio")
    risk_reward_ratio = practical_risk_reward_ratio if pd.notna(practical_risk_reward_ratio) else _row_number(row, "risk_reward_ratio")
    if pd.notna(risk_reward_ratio) and risk_reward_ratio < 2.0:
        caps.append(0.69)
    stop_loss_too_close_flag = str(row.get("stop_loss_too_close_flag", False)).strip().lower() in {"true", "1", "yes", "y"}
    if stop_loss_too_close_flag:
        caps.append(0.59)
    nearest_resistance_level = _row_number(row, "nearest_resistance_level")
    has_valid_resistance = pd.notna(nearest_resistance_level) and nearest_resistance_level > 0
    if not has_valid_resistance:
        caps.append(0.69)
    if _has_material_warning(row.get("risk_warning", "")):
        caps.append(0.69)

    trade_veto_flag = bool(row.get("trade_veto_flag", False))
    entry_plan_override = bool(row.get("entry_plan_override", False))
    if trade_veto_flag:
        caps.append(0.39)
    if caps:
        final_decision_score = min(final_decision_score, min(caps))
    final_decision_score = _clip01(final_decision_score, default=0.0)

    score_band_raw, decision_action_raw = get_score_band(
        final_decision_score,
        trade_veto_flag=trade_veto_flag,
        entry_plan_override=entry_plan_override,
    )
    if not trade_veto_flag and not has_valid_resistance:
        action_text = translate_decision_action(decision_action_raw)
        if action_text == "可條件式進場":
            score_band_raw = "Watchlist"
            decision_action_raw = "Wait for confirmation"
    score_band = translate_decision_label(score_band_raw)
    decision_action = translate_decision_action(decision_action_raw)

    if entry_plan_override:
        sqs = _row_number(row, "support_quality_score")
        if pd.notna(sqs) and sqs < 0.4:
            decision_action = "等待回測支撐且需量能確認後再進場"

    trade_veto_reason = translate_veto_reason(row.get("trade_veto_reason", ""))
    if trade_veto_flag:
        reason_text = trade_veto_reason or translate_veto_reason("risk plan failed")
        suggested_action_text = f"不建議進場。否決原因：{reason_text}。"
    else:
        suggested_action_text = decision_action

    return pd.Series({
        "final_decision_score": final_decision_score,
        "final_score_weight_profile": FINAL_SCORE_WEIGHT_PROFILE,
        "score_band": score_band,
        "decision_action": decision_action,
        "final_decision_label": score_band,
        "suggested_action_text": suggested_action_text,
    })


def build_position_sizing(row: pd.Series, account_equity=None) -> pd.Series:
    score = _row_number(row, "final_decision_score")
    practical_rr = _row_number(row, "practical_risk_reward_ratio")
    rr = practical_rr if pd.notna(practical_rr) else _row_number(row, "risk_reward_ratio")
    stop_loss_pct = _row_number(row, "stop_loss_pct")
    stop_loss_price = _row_number(row, "stop_loss_price_adjusted")
    risk_per_share = _row_number(row, "risk_per_share")
    entry_price = _row_number(row, "entry_price")
    rule_filter_score = _row_number(row, "rule_filter_score")
    safety_score = _row_number(row, "safety_score")
    trade_veto_flag = bool(row.get("trade_veto_flag", False))
    stop_loss_too_close_flag = str(row.get("stop_loss_too_close_flag", False)).strip().lower() in {"true", "1", "yes", "y"}
    atr_pct = _row_number(row, "atr_pct")

    def _result(
        risk_per_trade_pct: float,
        max_position_pct: float,
        suggested_position_pct: float,
        position_sizing_label: str,
        position_sizing_reason: str,
    ) -> pd.Series:
        suggested_position_pct = _clip01(suggested_position_pct, default=0.0)
        if suggested_position_pct <= 0:
            suggested_shares = 0
        elif account_equity is None or pd.isna(entry_price) or entry_price <= 0:
            suggested_shares = np.nan
        else:
            equity_value = pd.to_numeric(pd.Series([account_equity]), errors="coerce").iloc[0]
            if pd.isna(equity_value):
                suggested_shares = np.nan
            else:
                shares_by_position = np.floor(equity_value * suggested_position_pct / entry_price)
                if risk_per_trade_pct > 0 and pd.notna(risk_per_share) and risk_per_share > 0:
                    shares_by_risk = np.floor(equity_value * risk_per_trade_pct / risk_per_share)
                    suggested_shares = max(0, min(shares_by_position, shares_by_risk))
                else:
                    suggested_shares = max(0, shares_by_position)

        return pd.Series({
            "risk_per_trade_pct": risk_per_trade_pct,
            "max_position_pct": max_position_pct,
            "suggested_position_pct": suggested_position_pct,
            "suggested_shares": suggested_shares,
            "position_sizing_label": position_sizing_label,
            "position_sizing_reason": position_sizing_reason,
        })

    def _zero(label: str, reason: str) -> pd.Series:
        return _result(0.0, 0.0, 0.0, label, reason)

    def _sized(
        risk_per_trade_pct: float,
        max_position_pct: float,
        default_position_pct: float,
        label: str,
        reason: str,
    ) -> pd.Series:
        risk_budget_position_pct = (
            risk_per_trade_pct / stop_loss_pct
            if pd.notna(stop_loss_pct) and stop_loss_pct > 0
            else 0.0
        )
        suggested_position_pct = min(max_position_pct, default_position_pct, risk_budget_position_pct)
        return _result(risk_per_trade_pct, max_position_pct, suggested_position_pct, label, reason)

    if trade_veto_flag:
        return _zero("不配置倉位", "trade_veto_flag=True")
    if pd.isna(rr):
        return _zero("不配置倉位", "risk_reward_ratio 無法有效計算")
    if rr < 2.0:
        return _zero("不配置倉位", "risk_reward_ratio 低於 2.0")
    if pd.isna(stop_loss_price):
        return _zero("不配置倉位", "stop_loss_price_adjusted 無有效停損價")
    if pd.isna(stop_loss_pct):
        return _zero("不配置倉位", "stop_loss_pct 無法有效計算")
    if stop_loss_pct > 0.08:
        return _zero("不配置倉位", "stop_loss_pct 過高")
    if pd.isna(risk_per_share) or risk_per_share <= 0:
        return _zero("不配置倉位", "risk_per_share 小於等於 0")
    if stop_loss_too_close_flag and pd.notna(atr_pct) and atr_pct > 0.04:
        return _zero(
            "保守觀察，不配置實際倉位",
            "停損距離過近且 ATR 波動偏高，R/R 不採信",
        )
    if pd.isna(score) or score < 0.50:
        return _zero("不配置倉位", "final_decision_score 低於 0.50")

    if 0.50 <= score < 0.60:
        return _zero(
            "保守觀察，不配置實際倉位",
            "final_decision_score 介於 0.50~0.60",
        )

    if 0.60 <= score < 0.70:
        missing_reasons = []
        if pd.isna(rule_filter_score) or rule_filter_score < 0.50:
            missing_reasons.append("rule_filter_score 不足")
        if pd.isna(safety_score) or safety_score < 0.50:
            missing_reasons.append("safety_score 不足")
        if stop_loss_pct > 0.06:
            missing_reasons.append("stop_loss_pct 過高")
        if missing_reasons:
            return _zero("條件不足，暫不配置倉位", "；".join(missing_reasons))
        return _sized(0.0025, 0.05, 0.05, "小倉試探", "符合小倉試探條件")

    if 0.70 <= score < 0.80:
        missing_reasons = []
        if rr < 2.5:
            missing_reasons.append("risk_reward_ratio 低於 2.5")
        if pd.isna(rule_filter_score) or rule_filter_score < 0.55:
            missing_reasons.append("rule_filter_score 不足")
        if pd.isna(safety_score) or safety_score < 0.55:
            missing_reasons.append("safety_score 不足")
        if stop_loss_pct > 0.06:
            missing_reasons.append("stop_loss_pct 過高")
        if missing_reasons:
            return _zero("條件不足，暫不配置倉位", "；".join(missing_reasons))
        return _sized(0.005, 0.08, 0.08, "標準小倉位", "符合標準小倉位條件")

    if score >= 0.80:
        missing_reasons = []
        if rr < 3.0:
            missing_reasons.append("risk_reward_ratio 低於 3.0")
        if pd.isna(rule_filter_score) or rule_filter_score < 0.65:
            missing_reasons.append("rule_filter_score 不足")
        if pd.isna(safety_score) or safety_score < 0.60:
            missing_reasons.append("safety_score 不足")
        if stop_loss_pct > 0.05:
            missing_reasons.append("stop_loss_pct 過高")
        if missing_reasons:
            return _zero("條件不足，暫不配置倉位", "；".join(missing_reasons))
        return _sized(0.01, 0.15, 0.15, "高品質機會倉位", "符合高品質機會倉位條件")

    return _zero("條件不足，暫不配置倉位", "條件不足，暫不配置倉位")


def build_enhanced_decision_text(row: pd.Series) -> pd.Series:
    score_band = translate_decision_label(row.get("score_band", "No Score"))
    decision_action = translate_decision_action(row.get("decision_action", "No action"))
    veto_reason = translate_veto_reason(row.get("trade_veto_reason", ""))
    trade_veto_flag = bool(row.get("trade_veto_flag", False))
    nearest_resistance_level = _row_number(row, "nearest_resistance_level")
    has_valid_resistance = pd.notna(nearest_resistance_level) and nearest_resistance_level > 0
    rule_notes_text = translate_rule_filter_notes(row.get("rule_filter_notes", ""))
    trade_notes_text = translate_trade_plan_notes(row.get("trade_plan_notes", ""))
    technical_filter_text = translate_technical_filter_result(row.get("technical_filter_result", "NA"))

    summary = (
        f"最終決策：{score_band}。"
        f"final_decision_score={format_decision_score(row.get('final_decision_score'))}；"
        f"decision_action={decision_action}。"
    )
    reason = (
        f"第一層模型：safety_score={format_decision_score(row.get('safety_score'))}，"
        f"return_score={format_decision_score(row.get('return_score'))}，"
        f"combined_decision_score={format_decision_score(row.get('combined_decision_score'))}。"
        f"第二層規則式過濾：rule_filter_score={format_decision_score(row.get('rule_filter_score'))}，"
        f"technical_filter_result={technical_filter_text}，"
        f"rule_penalty_points={format_report_value(row.get('rule_penalty_points'))}。"
        f"交易可行性與風險報酬比：risk_reward_ratio={format_report_value(row.get('risk_reward_ratio'))}，"
        f"practical_risk_reward_ratio={format_report_value(row.get('practical_risk_reward_ratio'))}，"
        f"risk_reward_score={format_decision_score(row.get('risk_reward_score'))}，"
        f"trade_veto_flag={bool(row.get('trade_veto_flag', False))}，"
        f"trade_veto_reason={veto_reason or '無'}。"
        f"最終決策分數公式：{FINAL_SCORE_FORMULA_TEXT}。"
        f"公式說明：{FINAL_SCORE_FORMULA_EXPLANATION}"
    )
    action = (
        f"{decision_action}。entry_price={format_report_price(row.get('entry_price'))}，"
        f"stop_loss_price_adjusted={format_report_price(row.get('stop_loss_price_adjusted'))}"
        f"({format_report_value(row.get('stop_loss_pct'), pct=True)}), "
        f"take_profit_price={format_report_price(row.get('take_profit_price'))}，"
        f"suggested_position_pct={format_report_value(row.get('suggested_position_pct'), pct=True)}。"
    )
    if trade_veto_flag:
        reason_text = veto_reason or translate_veto_reason("risk plan failed")
        action = f"不建議進場。否決原因：{reason_text}。"

    report_lines = [
        "強化投資決策報告",
        "=" * 70,
        f"symbol: {row.get('symbol', 'NA')}",
        f"date: {format_report_date(row.get('date'))}",
        f"close: {format_report_price(row.get('close'))}",
        "",
        summary,
        "",
        "第一層模型綜合分數",
        f"combined_decision_score: {format_decision_score(row.get('combined_decision_score'))}",
        f"safety_score: {format_decision_score(row.get('safety_score'))}",
        f"return_score: {format_decision_score(row.get('return_score'))}",
        "",
        "最終決策",
        f"final_decision_score: {format_decision_score(row.get('final_decision_score'))}",
        f"final_score_weight_profile: {row.get('final_score_weight_profile', FINAL_SCORE_WEIGHT_PROFILE)}",
        f"final_decision_label: {score_band}",
        f"decision_action: {decision_action}",
        f"suggested_action_text: {normalize_text_for_report(action)}",
        f"final_decision_score_formula: {FINAL_SCORE_FORMULA_TEXT}",
        f"formula_note: {FINAL_SCORE_FORMULA_EXPLANATION}",
        "",
        "第二層規則式過濾",
        f"rule_filter_score: {format_decision_score(row.get('rule_filter_score'))}",
        f"technical_filter_result: {technical_filter_text}",
        f"rule_penalty_points: {format_report_value(row.get('rule_penalty_points'))}",
        f"rule_filter_notes: {rule_notes_text}",
        "",
        "交易可行性與風險報酬比",
        f"entry_price: {format_report_price(row.get('entry_price'))}",
        f"stop_loss_price_adjusted: {format_report_price(row.get('stop_loss_price_adjusted'))}",
        f"stop_loss_pct: {format_report_value(row.get('stop_loss_pct'), pct=True)}",
        f"take_profit_price: {format_report_price(row.get('take_profit_price'))}",
        f"risk_per_share: {format_report_price(row.get('risk_per_share'))}",
        f"stop_loss_too_close_flag: {bool(row.get('stop_loss_too_close_flag', False))}",
        f"practical_risk_per_share: {format_report_price(row.get('practical_risk_per_share'))}",
        f"reward_per_share: {format_report_price(row.get('reward_per_share'))}",
        f"risk_reward_ratio: {format_report_value(row.get('risk_reward_ratio'))}",
        f"practical_risk_reward_ratio: {format_report_value(row.get('practical_risk_reward_ratio'))}",
        f"risk_reward_score: {format_decision_score(row.get('risk_reward_score'))}",
        f"practical_risk_reward_score: {format_decision_score(row.get('practical_risk_reward_score'))}",
        "practical_risk_reward_note: 原始 R/R 可能因停損過近而被放大，因此系統採用 practical R/R 作為風險報酬判斷依據。",
        f"trade_veto_flag: {bool(row.get('trade_veto_flag', False))}",
        f"trade_veto_reason: {veto_reason or '無'}",
        f"trade_plan_notes: {trade_notes_text}",
        "",
        "倉位建議",
        f"risk_per_trade_pct: {format_report_value(row.get('risk_per_trade_pct'), pct=True)}",
        f"max_position_pct: {format_report_value(row.get('max_position_pct'), pct=True)}",
        f"suggested_position_pct: {format_report_value(row.get('suggested_position_pct'), pct=True)}",
        f"suggested_shares: {format_report_value(row.get('suggested_shares'), decimals=0)}",
        f"position_sizing_label: {normalize_text_for_report(row.get('position_sizing_label', ''))}",
        f"position_sizing_reason: {normalize_text_for_report(row.get('position_sizing_reason', ''))}",
        "",
        reason,
        action,
    ]
    if not has_valid_resistance:
        report_lines.append("目標價說明：未偵測到有效壓力位，目標價由 2R 推估，並非由實際壓力位得出，需人工確認上方空間。")
    if veto_reason:
        report_lines.append(f"否決原因：{veto_reason}")
    if _has_material_warning(row.get("risk_warning", "")):
        report_lines.append(f"風險警示：{normalize_text_for_report(row.get('risk_warning'))}")

    return pd.Series({
        "enhanced_decision_summary_text": normalize_text_for_report(summary),
        "enhanced_decision_reason_text": normalize_text_for_report(reason),
        "enhanced_suggested_action_text": normalize_text_for_report(action),
        "enhanced_decision_report_text": "\n".join(report_lines),
    })


def _apply_entry_plan_override_row(row: pd.Series) -> pd.Series:
    trade_veto_flag = bool(row.get("trade_veto_flag", False))
    trade_veto_reason = normalize_text_for_report(str(row.get("trade_veto_reason", "")))
    entry_rr_1 = _row_number(row, "entry_rr_1")
    entry_rr_2 = _row_number(row, "entry_rr_2")
    current_rr_score = _row_number(row, "risk_reward_score")

    if pd.isna(current_rr_score):
        current_rr_score = 0.0

    override = False
    override_reason = ""
    new_veto_flag = trade_veto_flag
    new_veto_reason = trade_veto_reason
    new_rr_score = current_rr_score

    support_quality_score = _row_number(row, "support_quality_score")

    if (
        trade_veto_flag
        and trade_veto_reason == "風險報酬比低於 1.5"
        and pd.notna(entry_rr_1) and entry_rr_1 >= 1.5
        and pd.notna(entry_rr_2) and entry_rr_2 >= 2.0
    ):
        override = True
        new_veto_flag = False
        new_veto_reason = ""
        if entry_rr_1 < 2.0:
            new_rr_score = 0.4
        else:
            new_rr_score = _clip01(0.7 + 0.2 * (entry_rr_1 - 2.0), default=0.7)

        if pd.notna(support_quality_score) and support_quality_score < 0.4:
            override_reason = "回測支撐進場 R/R 達標，但支撐品質偏低，建議等待量能確認後再進場"
        else:
            override_reason = "回測支撐進場 R/R 達標，建議等待回測再進場"

    return pd.Series({
        "trade_veto_flag": new_veto_flag,
        "trade_veto_reason": new_veto_reason,
        "risk_reward_score": new_rr_score,
        "entry_plan_override": override,
        "entry_plan_override_reason": override_reason,
    })


def _concat_overwrite(base_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
    overlap_cols = [col for col in add_df.columns if col in base_df.columns]
    if overlap_cols:
        base_df = base_df.drop(columns=overlap_cols)
    return pd.concat([base_df, add_df], axis=1)


def apply_enhanced_decision_layers(output_df: pd.DataFrame, account_equity=None) -> pd.DataFrame:
    out = output_df.copy()
    trade_plan_df = out.apply(build_trade_plan, axis=1, result_type="expand")
    out = _concat_overwrite(out, trade_plan_df)
    out["risk_warning"] = out.apply(build_risk_warning, axis=1)

    rule_filter_df = out.apply(apply_second_layer_rule_filter, axis=1, result_type="expand")
    out = _concat_overwrite(out, rule_filter_df)

    entry_plan_df = out.apply(build_entry_plan, axis=1, result_type="expand")
    out = _concat_overwrite(out, entry_plan_df)

    override_df = out.apply(_apply_entry_plan_override_row, axis=1, result_type="expand")
    out = _concat_overwrite(out, override_df)

    final_decision_df = out.apply(build_final_decision, axis=1, result_type="expand")
    out = _concat_overwrite(out, final_decision_df)
    if "technical_filter_result" in out.columns:
        out["technical_filter_result"] = out["technical_filter_result"].apply(translate_technical_filter_result)

    position_df = out.apply(
        lambda row: build_position_sizing(row, account_equity=account_equity),
        axis=1,
        result_type="expand",
    )
    out = _concat_overwrite(out, position_df)

    enhanced_text_df = out.apply(build_enhanced_decision_text, axis=1, result_type="expand")
    out = _concat_overwrite(out, enhanced_text_df)
    return out


def assign_score_band(value) -> str:
    score = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(score):
        return "NA"
    if score >= 0.80:
        return "0.80-1.00"
    if score >= 0.70:
        return "0.70-0.79"
    if score >= 0.60:
        return "0.60-0.69"
    if score >= 0.50:
        return "0.50-0.59"
    if score >= 0.40:
        return "0.40-0.49"
    return "0.00-0.39"


def _prepare_decision_summary_work_df(output_df: pd.DataFrame) -> pd.DataFrame:
    work = output_df.copy()
    work["_row_count"] = 1
    if "actual_target" in work.columns:
        work["_actual_target_for_summary"] = pd.to_numeric(work["actual_target"], errors="coerce")
    elif "actual_return_target" in work.columns:
        work["_actual_target_for_summary"] = pd.to_numeric(work["actual_return_target"], errors="coerce")
    else:
        work["_actual_target_for_summary"] = np.nan
    if "trade_veto_flag" in work.columns:
        work["_veto_int"] = work["trade_veto_flag"].fillna(False).astype(bool).astype(int)
    else:
        work["_veto_int"] = 0
    return work


def build_enhanced_decision_backtest_summary(output_df: pd.DataFrame, config) -> pd.DataFrame:
    if output_df.empty:
        return pd.DataFrame()
    work = _prepare_decision_summary_work_df(output_df)
    if "final_decision_label" not in work.columns:
        work["final_decision_label"] = work.get("score_band", "NA")
    if "decision_action" not in work.columns:
        work["decision_action"] = "NA"

    summary = (
        work
        .groupby(["final_decision_label", "decision_action"], dropna=False)
        .agg(
            count=("_row_count", "sum"),
            actual_target_mean=("_actual_target_for_summary", "mean"),
            future_return_10d_mean=("future_return_10d", "mean"),
            future_return_10d_median=("future_return_10d", "median"),
            win_rate=("future_return_10d", positive_return_rate),
            future_drawdown_10d_mean=("future_drawdown_10d", "mean"),
            abs_future_drawdown_10d_mean=("abs_future_drawdown_10d", "mean"),
            risk_adjusted_return_10d_mean=("risk_adjusted_return_10d", "mean"),
            avg_risk_reward_ratio=("risk_reward_ratio", "mean"),
            median_stop_loss_pct=("stop_loss_pct", "median"),
            veto_count=("_veto_int", "sum"),
        )
        .reset_index()
    )
    summary["veto_rate"] = summary.apply(lambda row: _safe_rate(row["veto_count"], row["count"]), axis=1)
    summary["market"] = getattr(config, "market", "")
    summary["symbol"] = getattr(config, "stock_no", "")
    return summary.sort_values(["final_decision_label", "decision_action"]).reset_index(drop=True)


def build_score_band_summary(output_df: pd.DataFrame, score_col: str, band_col_name: str) -> pd.DataFrame:
    if output_df.empty or score_col not in output_df.columns:
        return pd.DataFrame()
    work = _prepare_decision_summary_work_df(output_df)
    work[band_col_name] = work[score_col].apply(assign_score_band)
    summary = (
        work
        .groupby(band_col_name, dropna=False)
        .agg(
            count=("_row_count", "sum"),
            score_mean=(score_col, "mean"),
            actual_target_mean=("_actual_target_for_summary", "mean"),
            future_return_10d_mean=("future_return_10d", "mean"),
            win_rate=("future_return_10d", positive_return_rate),
            avg_risk_reward_ratio=("risk_reward_ratio", "mean"),
            veto_count=("_veto_int", "sum"),
        )
        .reset_index()
    )
    summary["veto_rate"] = summary.apply(lambda row: _safe_rate(row["veto_count"], row["count"]), axis=1)
    return summary


def build_veto_summary(output_df: pd.DataFrame) -> pd.DataFrame:
    if output_df.empty:
        return pd.DataFrame()
    work = _prepare_decision_summary_work_df(output_df)
    if "trade_veto_reason" not in work.columns:
        work["trade_veto_reason"] = ""
    total_count = len(work)
    summary = (
        work
        .groupby(["trade_veto_flag", "trade_veto_reason"], dropna=False)
        .agg(
            count=("_row_count", "sum"),
            final_decision_score_mean=("final_decision_score", "mean"),
            risk_reward_ratio_mean=("risk_reward_ratio", "mean"),
            stop_loss_pct_median=("stop_loss_pct", "median"),
        )
        .reset_index()
    )
    summary["rate"] = summary["count"].apply(lambda count: _safe_rate(count, total_count))
    return summary.sort_values(["trade_veto_flag", "count"], ascending=[False, False]).reset_index(drop=True)


def build_feature_and_rule_config(config) -> pd.DataFrame:
    rows = [
        ("model", "market", getattr(config, "market", "")),
        ("model", "symbol", getattr(config, "stock_no", "")),
        ("final_score_weight_profile", "profile", FINAL_SCORE_WEIGHT_PROFILE),
        ("final_score_weight", "safety_score", FINAL_SCORE_WEIGHTS["safety_score"]),
        ("final_score_weight", "return_score", FINAL_SCORE_WEIGHTS["return_score"]),
        ("final_score_weight", "rule_filter_score", FINAL_SCORE_WEIGHTS["rule_filter_score"]),
        ("final_score_weight", "risk_reward_score", FINAL_SCORE_WEIGHTS["risk_reward_score"]),
        ("final_score_formula", "formula", FINAL_SCORE_FORMULA_TEXT),
        ("final_score_formula", "explanation", FINAL_SCORE_FORMULA_EXPLANATION),
        ("trade_plan", "preferred_stop_loss_pct_min", 0.015),
        ("trade_plan", "preferred_stop_loss_pct_max", 0.060),
        ("trade_plan", "veto_stop_loss_pct_max", 0.080),
        ("trade_plan", "veto_min_risk_reward_ratio", 1.50),
        ("score_band", "high_quality_opportunity_min", 0.80),
        ("score_band", "bullish_watch_min", 0.70),
        ("score_band", "neutral_bullish_min", 0.60),
        ("score_band", "watchlist_min", 0.50),
        ("score_band", "caution_min", 0.40),
        ("rule_filter", "start_score", 0.50),
        ("rule_filter", "score_range", "0-1"),
        ("note", "hundred_point_score", "not used"),
    ]
    return pd.DataFrame(rows, columns=["section", "item", "setting"])


def build_decision_text(row: pd.Series) -> pd.Series:
    final_label = translate_decision_label(row.get("final_decision_label", "中性觀察"))
    risk_warning = normalize_text_for_report(row.get("risk_warning", "無明顯風險警示"))
    confidence_note = normalize_text_for_report(row.get("model_confidence_note", ""))

    safety_score = pd.to_numeric(pd.Series([row.get("safety_score", np.nan)]), errors="coerce").iloc[0]
    return_score = pd.to_numeric(pd.Series([row.get("return_score", np.nan)]), errors="coerce").iloc[0]
    combined_score = pd.to_numeric(pd.Series([row.get("combined_decision_score", np.nan)]), errors="coerce").iloc[0]

    if final_label == "安全與報酬分數偏高，仍需確認":
        decision_summary_text = (
            "模型判斷此股票同時具備較高安全性與報酬機會，可列入優先觀察名單，"
            "但仍需搭配技術面與風險控管確認。"
        )
    elif final_label == "穩健偏多，優先觀察":
        decision_summary_text = (
            "模型判斷此股票下行風險相對可控，整體偏向穩健觀察，"
            "適合等待較佳進場位置。"
        )
    elif final_label == "短線機會型，但需控管風險":
        decision_summary_text = (
            "模型判斷此股票可能具有短線報酬機會，但安全分數不足，"
            "進場前需特別控管停損與部位。"
        )
    elif final_label == "分數偏弱，暫以觀察為主":
        decision_summary_text = (
            "模型判斷此股票目前風險偏高或分數偏弱，暫不適合積極進場，"
            "建議以觀察為主。"
        )
    else:
        decision_summary_text = (
            "模型判斷此股票目前屬於中性觀察，尚未出現足夠明確的積極或避險訊號。"
        )

    if pd.notna(safety_score) and safety_score >= 0.65:
        safety_description = "安全分數偏高，下行風險相對較低"
    elif pd.notna(safety_score) and safety_score < 0.45:
        safety_description = "安全分數偏低，下行風險需提高警覺"
    else:
        safety_description = "安全分數接近中性"

    if pd.notna(return_score) and return_score >= 0.55:
        return_description = "報酬分數偏高，具備一定上行機會"
    elif pd.notna(return_score) and return_score < 0.45:
        return_description = "報酬分數偏低，上行機會尚不明確"
    else:
        return_description = "報酬分數接近中性"

    if not risk_warning or risk_warning == "無明顯風險警示":
        risk_text = "目前未出現明顯風險警示"
    else:
        risk_text = f"風險警示：{risk_warning}"

    reason_parts = [
        (
            f"安全分數為 {format_decision_score(safety_score)}，"
            f"報酬分數為 {format_decision_score(return_score)}，"
            f"第一層模型綜合分數為 {format_decision_score(combined_score)}。"
        ),
        f"{safety_description}；{return_description}。",
        f"{risk_text}。",
    ]
    if confidence_note:
        reason_parts.append(f"模型信心說明：{confidence_note}。")
    if pd.notna(pd.to_numeric(pd.Series([row.get("atr_pct", np.nan)]), errors="coerce").iloc[0]):
        reason_parts.append(f"ATR 波動率為 {format_decision_pct(row.get('atr_pct'))}。")
    if any(
        pd.notna(pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0])
        for col in ["future_return_10d", "future_drawdown_10d", "abs_future_drawdown_10d"]
    ):
        reason_parts.append(
            "回測對應 10 日未來報酬為 "
            f"{format_decision_pct(row.get('future_return_10d'))}，"
            f"未來回撤為 {format_decision_pct(row.get('future_drawdown_10d'))}，"
            f"絕對回撤為 {format_decision_pct(row.get('abs_future_drawdown_10d'))}。"
        )
    decision_reason_text = "".join(reason_parts)

    if "安全與報酬分數偏高" in str(final_label):
        suggested_action_text = (
            "建議列入優先觀察名單，可等待量價確認或回測支撐後分批評估，不建議忽略停損。"
        )
    elif "穩健偏多" in str(final_label):
        suggested_action_text = (
            "建議保守觀察，等待回測支撐、波動下降或技術指標轉強後再考慮進場。"
        )
    elif "短線機會型" in str(final_label):
        suggested_action_text = (
            "若要操作，應降低部位並設定明確停損，避免因高波動造成過大虧損。"
        )
    elif "分數偏弱" in str(final_label):
        suggested_action_text = (
            "建議暫不追高，等待風險警示下降、價格回到合理支撐區後再重新評估。"
        )
    else:
        suggested_action_text = (
            "建議維持觀察，不宜單靠模型分數進場，可等待更明確的安全分數、"
            "報酬分數或技術面訊號。"
        )
    if risk_warning and risk_warning != "無明顯風險警示":
        suggested_action_text += "由於目前存在風險警示，操作上應優先考慮風險控管。"

    if pd.notna(safety_score) and safety_score >= 0.65:
        report_safety_reason = "安全分數偏高，代表下行風險相對較低。"
    elif pd.notna(safety_score) and safety_score >= 0.55:
        report_safety_reason = "安全分數接近中性偏高，代表下行風險相對可控。"
    elif pd.notna(safety_score) and safety_score < 0.45:
        report_safety_reason = "安全分數偏低，代表下行風險需提高警覺。"
    else:
        report_safety_reason = "安全分數接近中性，代表下行風險尚未出現明確優勢。"

    if pd.notna(return_score) and return_score >= 0.55:
        report_return_reason = "報酬分數偏高，代表未來第 10 個交易日具備一定上行機會。"
    elif pd.notna(return_score) and return_score < 0.45:
        report_return_reason = "報酬分數偏低，代表未來第 10 個交易日的上行機會尚不明確。"
    else:
        report_return_reason = "報酬分數接近中性，代表未來第 10 個交易日的上行機會尚未明確。"

    if risk_warning and risk_warning != "無明顯風險警示":
        report_risk_reason = f"目前風險警示：{risk_warning}。"
        report_risk_action = "由於目前存在風險警示，操作上應優先考慮風險控管。"
    else:
        report_risk_reason = "目前未出現明顯風險警示。"
        report_risk_action = ""

    if "安全與報酬分數偏高" in str(final_label):
        report_action_lines = [
            "建議列入優先觀察名單，可等待量價確認或回測支撐後分批評估。",
            "不建議忽略停損，仍應搭配技術面與風險控管確認。",
        ]
    elif "穩健偏多" in str(final_label):
        report_action_lines = [
            "建議保守觀察，等待較佳進場位置。",
            "可等待回測支撐、波動下降或技術指標轉強後再考慮進場。",
        ]
    elif "短線機會型" in str(final_label):
        report_action_lines = [
            "若要操作，應降低部位並設定明確停損。",
            "避免因高波動造成過大虧損。",
        ]
    elif "分數偏弱" in str(final_label):
        report_action_lines = [
            "建議暫不追高，等待風險警示下降後再重新評估。",
            "可等待價格回到合理支撐區、波動下降或技術面轉強。",
        ]
    else:
        report_action_lines = [
            "建議維持觀察，不宜單靠模型分數進場。",
            "若要操作，應等待價格回測支撐、波動下降或技術面轉強後再評估。",
        ]
    if report_risk_action:
        report_action_lines.append(report_risk_action)

    decision_report_text = "\n".join([
        "=" * 70,
        "未來第 10 個交易日投資決策建議",
        "=" * 70,
        "",
        f"股票代碼：{row.get('symbol', 'NA')}",
        f"資料日期：{format_report_date(row.get('date'))}",
        f"目前收盤價：{format_report_price(row.get('close'))}",
        "預測目標：未來第 10 個交易日",
        "",
        "-" * 70,
        "一、模型綜合判斷",
        "-" * 70,
        f"決策結論：{final_label}",
        f"第一層模型綜合分數：{format_decision_score(combined_score)}",
        f"安全分數：{format_decision_score(safety_score)}",
        f"報酬機會分數：{format_decision_score(return_score)}",
        "",
        "系統判斷：",
        normalize_text_for_report(decision_summary_text),
        "",
        "-" * 70,
        "二、投資理由",
        "-" * 70,
        normalize_text_for_report(report_safety_reason),
        normalize_text_for_report(report_return_reason),
        normalize_text_for_report(report_risk_reason),
        "",
        "-" * 70,
        "三、操作建議",
        "-" * 70,
        *[normalize_text_for_report(line) for line in report_action_lines],
        "",
        "-" * 70,
        "四、風險提醒",
        "-" * 70,
        "本系統為投資決策支援工具，並非保證獲利模型。",
        "模型結果應搭配基本面、籌碼面、市場環境與個人風險承受度共同判斷。",
    ])

    return pd.Series({
        "decision_summary_text": normalize_text_for_report(decision_summary_text),
        "decision_reason_text": normalize_text_for_report(decision_reason_text),
        "suggested_action_text": normalize_text_for_report(suggested_action_text),
        "decision_report_text": decision_report_text,
    })


def interpret_latest_indicators(row: pd.Series) -> str:
    lines = []

    rsi14 = pd.to_numeric(pd.Series([row.get("RSI14", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(rsi14):
        lines.append("RSI14：NA，資料不足，無法判斷動能強弱。")
    elif rsi14 >= 70:
        lines.append(f"RSI14：{rsi14:.2f}，RSI 偏高，短線可能過熱。")
    elif rsi14 >= 55:
        lines.append(f"RSI14：{rsi14:.2f}，RSI 中性偏強，買盤動能相對較佳。")
    elif rsi14 >= 45:
        lines.append(f"RSI14：{rsi14:.2f}，RSI 接近中性，多空尚未明確。")
    elif rsi14 >= 30:
        lines.append(f"RSI14：{rsi14:.2f}，RSI 中性偏弱，動能不足。")
    else:
        lines.append(f"RSI14：{rsi14:.2f}，RSI 偏低，可能弱勢或接近超賣。")

    macd = pd.to_numeric(pd.Series([row.get("MACD", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(macd):
        lines.append("MACD：NA，資料不足，無法判斷趨勢動能。")
    elif macd > 0:
        lines.append(f"MACD：{macd:.2f}，MACD 為正，趨勢動能偏多。")
    else:
        lines.append(f"MACD：{macd:.2f}，MACD 為負，趨勢動能偏弱。")

    k_value = pd.to_numeric(pd.Series([row.get("K", np.nan)]), errors="coerce").iloc[0]
    d_value = pd.to_numeric(pd.Series([row.get("D", np.nan)]), errors="coerce").iloc[0]
    j_value = pd.to_numeric(pd.Series([row.get("J", np.nan)]), errors="coerce").iloc[0]
    k_text = format_decision_score(k_value)
    d_text = format_decision_score(d_value)
    j_text = format_decision_score(j_value)
    if pd.isna(k_value) or pd.isna(d_value):
        lines.append(f"KDJ：K={k_text}，D={d_text}，J={j_text}，資料不足，無法判斷短線動能。")
    elif k_value > d_value:
        lines.append(f"KDJ：K={k_text}，D={d_text}，J={j_text}，K 值高於 D 值，短線動能偏多。")
    else:
        lines.append(f"KDJ：K={k_text}，D={d_text}，J={j_text}，K 值低於 D 值，短線動能偏弱。")

    atr_pct = pd.to_numeric(pd.Series([row.get("atr_pct", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(atr_pct):
        lines.append("ATR%：NA，資料不足，無法判斷短期波動。")
    elif atr_pct >= 0.05:
        lines.append(f"ATR%：{format_decision_pct(atr_pct)}，短期波動偏高，需提高停損與部位控管。")
    elif atr_pct >= 0.03:
        lines.append(f"ATR%：{format_decision_pct(atr_pct)}，短期波動中等，操作需留意震盪。")
    else:
        lines.append(f"ATR%：{format_decision_pct(atr_pct)}，短期波動相對溫和。")

    fib_position = pd.to_numeric(pd.Series([row.get("fib_position", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(fib_position):
        lines.append("Fib Position：NA，資料不足，無法判斷價格區間位置。")
    elif fib_position < 0.5:
        lines.append(f"Fib Position：{fib_position:.2f}，價格位於近期區間偏低位置。")
    elif fib_position <= 0.786:
        lines.append(f"Fib Position：{fib_position:.2f}，價格位於近期區間中段至偏高位置。")
    else:
        lines.append(f"Fib Position：{fib_position:.2f}，價格接近近期區間高位，追高風險較高。")

    support_distance = pd.to_numeric(pd.Series([row.get("support_distance", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(support_distance):
        lines.append("Support Distance：NA，資料不足，無法判斷支撐距離。")
    elif support_distance >= 0.06:
        lines.append(f"Support Distance：{format_decision_pct(support_distance)}，距離支撐較遠，若回落可能有較大修正空間。")
    elif support_distance >= 0.03:
        lines.append(f"Support Distance：{format_decision_pct(support_distance)}，距離支撐中等。")
    else:
        lines.append(f"Support Distance：{format_decision_pct(support_distance)}，價格接近支撐區。")

    resistance_distance = pd.to_numeric(pd.Series([row.get("resistance_distance", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(resistance_distance):
        lines.append("Resistance Distance：NA，資料不足，無法判斷壓力距離。")
    elif resistance_distance <= 0.03:
        lines.append(f"Resistance Distance：{format_decision_pct(resistance_distance)}，上方壓力接近。")
    elif resistance_distance <= 0.06:
        lines.append(f"Resistance Distance：{format_decision_pct(resistance_distance)}，上方壓力距離中等。")
    else:
        lines.append(f"Resistance Distance：{format_decision_pct(resistance_distance)}，上方壓力距離較遠。")

    risk_warning = normalize_text_for_report(row.get("risk_warning", "無明顯風險警示"))
    lines.append(f"Risk Warning：{risk_warning if risk_warning else '無明顯風險警示'}。")
    return "\n".join(normalize_text_for_report(line) for line in lines)


def build_combined_decision_explanation_text(row: pd.Series) -> str:
    final_label = translate_decision_label(row.get("final_decision_label", "中性觀察"))
    combined_score = format_decision_score(row.get("combined_decision_score"))
    final_score = format_decision_score(row.get("final_decision_score"))
    risk_warning = normalize_text_for_report(row.get("risk_warning", "無明顯風險警示"))
    if "中性觀察" in str(final_label):
        combined_explanation = "目前系統傾向中性觀察，不建議直接追高，建議等待支撐回測、風險下降或動能轉強後再評估。"
    elif "分數偏弱" in str(final_label):
        combined_explanation = "目前風險偏高，建議暫不積極進場，等待風險警示下降或價格回到合理支撐後再重新評估。"
    elif "穩健偏多" in str(final_label) or "安全與報酬分數偏高" in str(final_label):
        combined_explanation = "可列入優先觀察，但仍需等待量價或支撐確認，並以停損距離控制部位風險。"
    else:
        combined_explanation = "目前訊號尚未完全明確，建議搭配技術指標、風險警示與人工判斷共同使用。"
    return normalize_text_for_report(
        f"最終決策標籤為「{final_label}」，第一層模型綜合分數為 {combined_score}，"
        f"最終決策分數為 {final_score}，"
        f"目前風險警示為「{risk_warning if risk_warning else '無明顯風險警示'}」。"
        f"{combined_explanation}"
    )


def build_decision_guide_text(row: pd.Series, include_combined: bool = True) -> str:
    lines = []

    rsi14 = pd.to_numeric(pd.Series([row.get("RSI14", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(rsi14):
        rsi_state = "資料不足"
        rsi_impact = "RSI 資料不足，短線動能需搭配其他指標判斷。"
    elif rsi14 >= 70:
        rsi_state = "過熱"
        rsi_impact = "RSI 偏高，短線可能過熱，若追價需提高風險控管。"
    elif rsi14 >= 55:
        rsi_state = "中性偏強"
        rsi_impact = "RSI 中性偏強，買盤動能相對較佳，可作為偏多觀察依據。"
    elif rsi14 >= 45:
        rsi_state = "中性"
        rsi_impact = "RSI 接近中性，多空尚未明確，適合等待方向確認。"
    elif rsi14 >= 30:
        rsi_state = "中性偏弱"
        rsi_impact = "RSI 中性偏弱，動能不足，進場前需等待轉強訊號。"
    else:
        rsi_state = "偏弱或接近超賣"
        rsi_impact = "RSI 偏低，可能弱勢或接近超賣，不宜只因低檔就直接進場。"
    lines.append(f"1. RSI14｜{format_report_value(rsi14)}｜{rsi_state}：{rsi_impact}")

    macd = pd.to_numeric(pd.Series([row.get("MACD", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(macd):
        macd_state = "資料不足"
        macd_impact = "MACD 資料不足，趨勢動能需搭配其他技術指標。"
    elif macd > 0:
        macd_state = "偏多"
        macd_impact = "MACD 為正，趨勢動能偏多，有利於觀察上行延續。"
    else:
        macd_state = "偏弱"
        macd_impact = "MACD 為負，趨勢動能偏弱，操作上宜保守。"
    lines.append(f"2. MACD｜{format_report_value(macd)}｜{macd_state}：{macd_impact}")

    k_value = pd.to_numeric(pd.Series([row.get("K", np.nan)]), errors="coerce").iloc[0]
    d_value = pd.to_numeric(pd.Series([row.get("D", np.nan)]), errors="coerce").iloc[0]
    j_value = pd.to_numeric(pd.Series([row.get("J", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(k_value) or pd.isna(d_value):
        kdj_state = "資料不足"
        kdj_impact = "KDJ 資料不足，短線動能需等待資料補齊。"
    elif k_value > d_value:
        kdj_state = "短線偏多"
        kdj_impact = "K 值高於 D 值，短線動能偏多，可作為短線轉強參考。"
    else:
        kdj_state = "短線偏弱"
        kdj_impact = "K 值低於 D 值，短線動能偏弱，宜等待動能修復。"
    lines.append(
        "3. KDJ｜"
        f"K={format_report_value(k_value)}，D={format_report_value(d_value)}，J={format_report_value(j_value)}"
        f"｜{kdj_state}：{kdj_impact}"
    )

    fib_position = pd.to_numeric(pd.Series([row.get("fib_position", np.nan)]), errors="coerce").iloc[0]
    fib_zone = normalize_text_for_report(row.get("fib_zone", "NA"))
    if pd.isna(fib_position):
        fib_state = "資料不足"
        fib_impact = "Fib 位置資料不足，價格區間判斷需參考支撐壓力。"
    elif fib_position < 0.5:
        fib_state = "價格位於近期區間偏低"
        fib_impact = "價格相對不在高檔，若風險警示下降，可觀察支撐反應。"
    elif fib_position <= 0.786:
        fib_state = "價格位於中段至偏高區"
        fib_impact = "價格位置不算低，追價前需等待量價或支撐確認。"
    else:
        fib_state = "價格接近近期高位"
        fib_impact = "價格接近高位，追高風險較高，宜等待回測或突破確認。"
    lines.append(f"4. Fib Position｜{format_report_value(fib_position)} / {fib_zone}｜{fib_state}：{fib_impact}")

    support_level = pd.to_numeric(pd.Series([row.get("nearest_support_level", np.nan)]), errors="coerce").iloc[0]
    support_distance = pd.to_numeric(pd.Series([row.get("support_distance", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(support_distance):
        support_state = "資料不足"
        support_impact = "支撐距離資料不足，停損位置需人工確認。"
    elif support_distance < 0.03:
        support_state = "接近支撐"
        support_impact = "價格接近支撐區，若支撐有效，可作為觀察進場位置。"
    elif support_distance < 0.06:
        support_state = "距離支撐中等"
        support_impact = "距離支撐仍有一定空間，進場前需留意回落風險。"
    else:
        support_state = "距離支撐較遠"
        support_impact = "距離支撐較遠，若回落可能有較大修正空間，不宜忽略停損。"
    lines.append(
        f"5. Support｜{format_report_price(support_level)}，距離 {format_report_value(support_distance, pct=True)}"
        f"｜{support_state}：{support_impact}"
    )

    resistance_level = pd.to_numeric(pd.Series([row.get("nearest_resistance_level", np.nan)]), errors="coerce").iloc[0]
    resistance_distance = pd.to_numeric(pd.Series([row.get("resistance_distance", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(resistance_distance):
        resistance_state = "資料不足"
        resistance_impact = "壓力距離資料不足，上檔空間需人工確認。"
    elif resistance_distance < 0.03:
        resistance_state = "上方壓力近"
        resistance_impact = "上方壓力接近，短線上檔空間受限，追價需保守。"
    elif resistance_distance < 0.06:
        resistance_state = "上方壓力中等"
        resistance_impact = "上方壓力距離中等，可等待突破確認再提高信心。"
    else:
        resistance_state = "上方壓力較遠"
        resistance_impact = "上方壓力較遠，若動能配合，仍有一定上行觀察空間。"
    lines.append(
        f"6. Resistance｜{format_report_price(resistance_level)}，距離 {format_report_value(resistance_distance, pct=True)}"
        f"｜{resistance_state}：{resistance_impact}"
    )

    stop_loss_pct = pd.to_numeric(pd.Series([row.get("stop_loss_pct", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(stop_loss_pct):
        stop_state = "資料不足"
        stop_impact = "停損成本資料不足，實際操作前需重新估算風險。"
    elif stop_loss_pct < 0.03:
        stop_state = "風險成本低"
        stop_impact = "停損成本相對低，若其他條件配合，部位規劃較有彈性。"
    elif stop_loss_pct < 0.06:
        stop_state = "風險成本中等"
        stop_impact = "停損成本中等，進場需控制部位並保留震盪空間。"
    else:
        stop_state = "風險成本偏高"
        stop_impact = "停損成本偏高，若要操作應降低部位並嚴格控管風險。"
    lines.append(f"7. Stoploss Risk｜{format_report_value(stop_loss_pct, pct=True)}｜{stop_state}：{stop_impact}")

    ob_type = normalize_text_for_report(row.get("recent_2w_ob_type", "none"))
    ob_fallback = pd.to_numeric(pd.Series([row.get("recent_2w_ob_is_fallback", np.nan)]), errors="coerce").iloc[0]
    if ob_type == "fallback_range" or ob_fallback == 1:
        ob_impact = "沒有明確 OB，目前使用近 10 日區間替代，結構支撐可信度較低，操作需更依賴支撐與停損。"
    else:
        ob_impact = f"目前 OB 類型為 {ob_type if ob_type else 'none'}，可作為近期供需結構的輔助參考。"
    lines.append(f"8. Order Block｜{ob_type}，fallback={format_report_value(ob_fallback, decimals=0)}：{ob_impact}")

    gap_type = normalize_text_for_report(row.get("recent_active_gap_type", "none"))
    if gap_type == "bullish":
        gap_impact = "近期仍有 bullish gap 結構，代表多方缺口支撐仍可納入觀察。"
    elif gap_type == "bearish":
        gap_impact = "近期仍有 bearish gap 結構，代表上方或下行壓力需提高警覺。"
    else:
        gap_impact = "目前沒有 active gap，缺口結構對本次判斷影響較低。"
    lines.append(f"9. Gap｜{gap_type}：{gap_impact}")

    if include_combined:
        lines.extend([
            "",
            "綜合決策說明：",
            build_combined_decision_explanation_text(row),
        ])
    return "\n".join(normalize_text_for_report(line) for line in lines)


__all__ = [name for name in globals() if not name.startswith('__')]
