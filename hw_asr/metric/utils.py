# Don't forget to support cases when target_text == ''
import editdistance


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    return editdistance.distance(target_words, predicted_words) / max(1, len(target_words))


def calc_cer(target_text, predicted_text) -> float:
    return editdistance.distance(target_text, predicted_text) / max(1, len(target_text))
