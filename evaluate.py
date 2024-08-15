import os
import sys

def parse_intervals(line):
    # Remove unnecessary spaces and brackets
    line = line.strip()[1:-1]
    intervals = line.split('][')
    parsed_intervals = []
    for interval in intervals:
        numbers = interval.split(',')
        if len(numbers) == 2 and all(num.strip().isdigit() for num in numbers):
            parsed_intervals.append(tuple(map(int, numbers)))
        else:
            continue
    return parsed_intervals

def read_intervals(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line for line in file.readlines()]
    return [parse_intervals(line) for line in data]

def calculate_metrics(gold_data, pred_data):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    true_positives_total = 0
    false_positives_total = 0
    false_negatives_total = 0

    for gold, pred in zip(gold_data, pred_data):
        gold_set = set(gold)
        pred_set = set(pred)
        
        true_positives = len(gold_set & pred_set)
        false_positives = len(pred_set - gold_set)
        false_negatives = len(gold_set - pred_set)

        true_positives_total += true_positives
        false_positives_total += false_positives
        false_negatives_total += false_negatives

        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            precision_scores.append(precision)
        else:
            precision = 0
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
            recall_scores.append(recall)
        else:
            recall = 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
        else:
            f1 = 0

    average_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    overall_precision = true_positives_total / (true_positives_total + false_positives_total) if true_positives_total + false_positives_total > 0 else 0
    overall_recall = true_positives_total / (true_positives_total + false_negatives_total) if true_positives_total + false_negatives_total > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    return average_precision, average_recall, average_f1, overall_precision, overall_recall, overall_f1

def main(file_path):
    gold_data = read_intervals(os.path.join(file_path, 'stem_gold.txt'))
    pred_data = read_intervals(os.path.join(file_path, 'stem_predict.txt'))

    average_precision, average_recall, average_f1, overall_precision, overall_recall, overall_f1 = calculate_metrics(gold_data, pred_data)

    print(f'Average Precision: {average_precision:.3f}')
    print(f'Average Recall: {average_recall:.3f}')
    print(f'Average F1 Score: {average_f1:.3f}')
    print(f'Overall Precision: {overall_precision:.3f}')
    print(f'Overall Recall: {overall_recall:.3f}')
    print(f'Overall F1 Score: {overall_f1:.3f}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        main(sys.argv[1])
