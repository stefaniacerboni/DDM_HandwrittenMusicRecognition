from torch.utils.data import random_split, DataLoader, ConcatDataset


def cer_wer(decoded_sequence, ground_truth_sequence):
    S = sum(1 for x, y in zip(decoded_sequence, ground_truth_sequence) if x != y)
    D = abs(len(decoded_sequence) - len(ground_truth_sequence))
    I = abs(len(decoded_sequence) - len(ground_truth_sequence))
    N = len(ground_truth_sequence)
    cer = (S + D + I) / N
    return cer


def get_curriculum_learning_loader(synthetic_dataset, historical_dataset, collate_fn, batch_size, historical_rate):
    total_historical_samples = len(historical_dataset)
    historical_size = int(historical_rate * total_historical_samples)

    # Create data loaders for the initial proportions
    synthetic_data, _ = random_split(
        synthetic_dataset,
        [total_historical_samples - historical_size,
         len(synthetic_dataset) - (total_historical_samples - historical_size)]
    )
    historical_data, _ = random_split(
        total_historical_samples,
        [historical_size, historical_size - historical_size]
    )
    data_loader = DataLoader(ConcatDataset([synthetic_data, historical_data]),
                             batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return data_loader
