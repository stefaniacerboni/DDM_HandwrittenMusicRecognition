def cer_wer(decoded_sequence, ground_truth_sequence):
    S = sum(1 for x, y in zip(decoded_sequence, ground_truth_sequence) if x != y)
    D = abs(len(decoded_sequence) - len(ground_truth_sequence))
    I = abs(len(decoded_sequence) - len(ground_truth_sequence))
    N = len(ground_truth_sequence)
    cer = (S + D + I) / N
    return cer
