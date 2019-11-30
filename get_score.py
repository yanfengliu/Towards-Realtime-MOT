import eval
import pickle


evaluator = eval.BboxTrackEvaluator(0.5)
for i in range(10):
    print(f'Evaluating sequence {i}')
    dt_result_txt_path = f'temp/{i}/results.txt'
    dt_sequence = evaluator.gen_dt_target_sequence(dt_result_txt_path)
    gt_sequence_path = f'C:/Users/yliu60/Documents/GitHub/embedding_tracking/dataset/6_shapes/test/seq_{i}/sequence.pickle'
    with open(gt_sequence_path, 'rb') as handle:
        gt_sequence = pickle.load(handle)
    evaluator.eval_on_sequence(dt_sequence, gt_sequence)
strsummary = evaluator.summarize()
txt_path = f'summary/scores.txt'
with open(txt_path, "a") as f:
    f.write(f'{strsummary} \n')