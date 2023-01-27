import matplotlib.pyplot as plt
import pandas as pd
from torch import load as torch_load

def load_metrics(topics_list, embedding_list, data_path, save_path):
    etm_tc, etm_td, etm_tq = {}, {}, {}
    for emb_name, _ in embedding_list.items():
        etm_tc[emb_name], etm_td[emb_name], etm_tq[emb_name] = [], [], []
        for topics in topics_list:
            model_file = 'ETM_' + data_path.split("/")[-1] + '_' + emb_name + '_' + str(topics) + '_parameters.pt'
            metrics = torch_load(save_path + '/' + model_file)
            etm_tc[emb_name].append(metrics['tc'])
            etm_td[emb_name].append(metrics['td'])
            etm_tq[emb_name].append(metrics['tc'] * metrics['td'])
            #print(model_file.ljust(70), round(metrics['tc']*metrics['td'],4))
    etm_tc = pd.DataFrame.from_dict(etm_tc)
    etm_td = pd.DataFrame.from_dict(etm_td)
    etm_tq = pd.DataFrame.from_dict(etm_tq)
    etm_tc.index = etm_td.index = etm_tq.index = topics_list
    return etm_tc, etm_td, etm_tq


def display_metric(metrics, name, show_table=False):
    fig, ax = plt.subplots(figsize=(12,6))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    for emb_name in metrics.columns:
        plt.plot(metrics.index, metrics[emb_name], label=emb_name, marker='o')
    plt.xlabel('number of topics', fontsize=16)
    plt.ylabel(name, fontsize=16)
    plt.xticks(metrics.index)
    plt.title('Topic model comparison', fontsize=16)
    plt.legend()
    if show_table:
        display(etm_tq)
    plt.show()