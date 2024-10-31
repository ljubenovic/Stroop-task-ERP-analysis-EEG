import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def read_rejected_components(S_id):

    csv_path = os.path.join('data','ICA components','S{}_components_to_reject.csv'.format(S_id))

    df = pd.read_csv(csv_path, header=None, on_bad_lines='skip').iloc[1:]
    
    df = df[df.count(axis=1) > 1]
    df.columns = ['component', 'corr']

    n_comp = df.iloc[:, 0].nunique()
    componens = df.iloc[:, 0].astype(int).unique()

    df_row = pd.Series([S_id, n_comp, componens], index=['S_id', 'n_comp', 'components'])

    return df_row


def hist_rejected_components(unique, counts):

    plt.figure(figsize=(8,4))
    plt.bar(unique, counts, color='black', zorder=3, alpha=0.9)
    plt.title('Broj odbačenih nezavisnih komponenti')
    plt.xlabel('broj komponenti')
    plt.xticks(unique)
    plt.ylabel('broj ispitanika')
    plt.yticks(np.arange(1,11,2))
    plt.grid(which='both', zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join('data','Results','num_of_rejected_components.png'))
    plt.close()

    return


def save_tabular_data(df_results):

    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_results.values, colLabels=['ID ispitanika','broj odbačenih komponenti','odbačene komponente'], cellLoc = 'center', loc='center')
    table.auto_set_column_width([0, 1, 2])
    table.scale(1, 2)
    plt.tight_layout()
    plt.savefig(os.path.join('data','Results','rejected_components.png'))
    plt.close()

    return


if __name__ == '__main__':

    df_results = pd.DataFrame(columns=['S_id', 'n_comp', 'components'])

    for i in range(0,21):

        S_id = i+1
        if S_id == 5:
            continue

        df_row = read_rejected_components(S_id)
        df_results = pd.concat([df_results, df_row.to_frame().T], ignore_index=False)
        

    S_ids = df_results['S_id'].to_numpy().astype(np.uint8)
    n_comps = df_results['n_comp'].to_numpy().astype(np.uint8)

    unique, counts = np.unique(n_comps, return_counts=True)
    hist_rejected_components(unique, counts)

    print('-Number of rejected components-')
    print('mean: ', np.mean(unique))
    print('std: ', np.std(unique))

    save_tabular_data(df_results)

