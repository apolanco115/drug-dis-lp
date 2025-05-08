import argparse
import json
import datetime
import time
from bipartite_link_prediction.bipartite_network import BipartiteNetwork
from bipartite_link_prediction.evaluation.evaluator import Evaluator
from bipartite_link_prediction.evaluation.train_test_split import TrainTestSplit
from bipartite_link_prediction.predictors.sbm_predictor import SBMPredictor
from bipartite_link_prediction.predictors.node2vec_predictor import N2VPredictor

def main():
    parser = argparse.ArgumentParser(description="Load config for Bipartite Network Prediction")
    parser.add_argument('config_file', help="Path to the configuration file")
    parser.add_argument('--trial', type=int, help="Run a specific trial number. If omitted, all trials are run.", required=False)

    args = parser.parse_args()
    # Correctly open the JSON configuration file and load its contents
    with open(args.config_file, 'r') as file:
        config_params = json.load(file)

    bipartite_config = config_params['BipartiteNetwork']
    train_test_config = config_params['TrainTestSplit']
    predictors_config = config_params['Predictors']

    bipartite_network = BipartiteNetwork(**bipartite_config)
    print('number of disease:', len(bipartite_network.top_nodes))
    print('number of drugs:', len(bipartite_network.bottom_nodes))
    print('number of edges:', bipartite_network.graph.num_edges())

    print('generating test data')
    train_test_split = TrainTestSplit(bipartite_network=bipartite_network, **train_test_config)

    for predictor_name, predictor_params in predictors_config.items():
        print('Using', predictor_name, 'predictor')
        trial_metrics = {}

        if args.trial is not None:
            if args.trial >= 0 and args.trial < train_test_split.num_trials:
                trial_range = [args.trial]
            else:
                raise ValueError("Trial number is out of range.")
        else:
            trial_range = range(train_test_split.num_trials)

        for t in trial_range:
            print('Begin trial', t)
            predictor_class = globals()[predictor_name]
            predictor = predictor_class(**predictor_params)
            print('getting test data')
            test_pairs, test_labels, num_test_edges = train_test_split.get_trial_data(t)
            print('there are:', len(train_test_split.non_edges), 'non edges being used for training')
            print('preparing data for fitting')
            data = predictor.prepare_data(bipartite_network, test_pairs[:num_test_edges])

            print('fitting data')
            predictor.fit(data)

            print('predicting links')
            predicted_scores = predictor.predict_links(test_pairs, test_labels, trial_num=t)
            if predicted_scores is not None:
                print('evaluating')
                fpr, tpr, roc_thresholds = Evaluator.calculate_roc(test_labels, predicted_scores)
                precision, recall, pr_thresholds = Evaluator.calculate_pr_curve(test_labels, predicted_scores)
                topk = Evaluator.top_k_precision(test_labels, predicted_scores, k=100)

                # Calculate AUC metrics
                roc_auc = Evaluator.calculate_aucroc(fpr, tpr)
                pr_auc = Evaluator.calculate_aucpr(precision, recall)

                base_fpr, interp_tpr = Evaluator.interpolate_curve(fpr, tpr, num_points=1000)
                base_recall, interp_prec = Evaluator.interpolate_curve(recall, precision, num_points=1000)

                # Store metrics for this fold
                trial_metrics[t] = {
                    'fpr': base_fpr,
                    'tpr': interp_tpr,
                    'roc_auc': roc_auc,
                    'precision': interp_prec,
                    'recall': base_recall,
                    'pr_auc': pr_auc,
                    'topk': topk
                }
                print(f"Trial {t} - ROC AUC: {roc_auc}, PR AUC: {pr_auc}, Top-k precision: {topk}")
            print(f"Trial {t} completed")
            print('----------------------------------------------------------------------------------')
        print("All trials completed")
        if trial_metrics:
            all_fpr_tpr = [(metrics['fpr'], metrics['tpr']) for metrics in trial_metrics.values()]
            all_re_prec = [(metrics['recall'], metrics['precision']) for metrics in trial_metrics.values()]

            all_aucroc = [metrics['roc_auc'] for metrics in trial_metrics.values()]
            all_aucpr = [metrics['pr_auc'] for metrics in trial_metrics.values()]
            all_topk = [metrics['topk'] for metrics in trial_metrics.values()]

            average_aucroc = Evaluator.calculate_average(all_aucroc)
            ste_aucroc = Evaluator.calculate_standard_error(all_aucroc)

            average_aucpr = Evaluator.calculate_average(all_aucpr)
            ste_aucpr = Evaluator.calculate_standard_error(all_aucpr)

            average_topk = Evaluator.calculate_average(all_topk)
            ste_topk = Evaluator.calculate_standard_error(all_topk)

            print("Average AUCROC", average_aucroc, "±", ste_aucroc)
            print("Average AUCPR", average_aucpr, "±", ste_aucpr)
            print("Average Top-k", average_topk, "±", ste_topk)

            average_fpr, average_tpr = Evaluator.calculate_average_curve(all_fpr_tpr)
            average_rec, average_prec = Evaluator.calculate_average_curve(all_re_prec)

            current_date = datetime.date.today()
            formatted_date = current_date.strftime('%y%m%d')

            Evaluator.write_curve(average_fpr, average_tpr, x_header='FPR', y_header='TPR',
                                  filename=predictor_name + "_average_roc_" + formatted_date + ".csv")
            Evaluator.write_curve(average_rec, average_prec, x_header='Recall', y_header='Precision',
                                  filename=predictor_name + "_average_pr_" + formatted_date + ".csv")
            print("Average curves written to disk")


if __name__ == "__main__":
    main()
